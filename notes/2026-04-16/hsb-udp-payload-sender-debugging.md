# Debugging `thor_send_payload.py` — Full Journey
**Date:** 2026-04-16

This document chronicles every problem, root cause, and fix applied while getting
`thor_send_payload.py` to successfully send a 32-bit payload to the HSB (Holoscan
Sensor Bridge) FPGA board over UDP.

---

## Table of Contents

1. [Background — What the Script Is Supposed to Do](#1-background)
2. [Problem 1 — Wrong Port Name in `add_flow`](#2-problem-1--wrong-port-name-in-add_flow)
3. [Problem 2 — Python Can't Feed `UdpTransmitterOp` a Tensor](#3-problem-2--python-cant-feed-udptransmitterop-a-tensor)
4. [Problem 3 — Decided to Write a C++ Operator (Build Journey)](#4-problem-3--decided-to-write-a-c-operator)
   - 4a. [Dynamic Linker Error — Missing `libpayload_generator_op.so`](#4a-dynamic-linker-error)
   - 4b. [Pybind11 Type Registry Mismatch — `unknown base type holoscan::Operator`](#4b-pybind11-type-registry-mismatch)
   - 4c. [First Fix Attempt — Import `holoscan.core` in the Pybind11 Module](#4c-first-fix-attempt)
   - 4d. [Second Fix Attempt — Pin Pybind11 to Version 2.13.6](#4d-second-fix-attempt)
   - 4e. [Root Cause Found — ABI Suffix in Internals Capsule Name](#4e-root-cause-found)
   - 4f. [Final Fix — Force Empty ABI Macros in CMake](#4f-final-fix)
5. [Problem 4 — Scheduler Deadlock (Transmitter Never Runs)](#5-problem-4--scheduler-deadlock)
   - 5a. [What the Greedy Scheduler Does](#5a-what-the-greedy-scheduler-does)
   - 5b. [Why It Deadlocks Here](#5b-why-it-deadlocks-here)
   - 5c. [Fix — Switch to `EventBasedScheduler`](#5c-fix--switch-to-eventbasedscheduler)
6. [Final Working Architecture](#6-final-working-architecture)
7. [Key Lessons](#7-key-lessons)

---

## 1. Background

The goal is to send a 32-bit value (`0xDEADBEEF` by default) from the AGX Thor host
to an FPGA on an HSB board using a Holoscan pipeline. The pipeline consists of two
operators chained together:

```
PayloadGeneratorOp  ──→  UdpTransmitterOp
   (produces data)          (sends UDP packet)
```

Holoscan is NVIDIA's GPU-accelerated pipeline framework. Operators in a pipeline
exchange typed messages through declared input/output ports, and a scheduler decides
the order in which operators execute.

The HSB library (`hololink`) provides a ready-made `UdpTransmitterOp` — a C++
operator that takes a GPU tensor on its input port and blasts the raw bytes over UDP
to a specified IP:port.

---

## 2. Problem 1 — Wrong Port Name in `add_flow`

### Symptom

```
ERROR 0.7188 udp_transmitter_op.cpp:131 compute tid=0xc1 -- Failed to receive message from port 'in'
```

### What happened

`add_flow` was called with:

```python
self.add_flow(payload_generator, udp_transmitter, {("output", "in")})
```

The error message from the C++ source said `port 'in'`, which looked like the internal
GXF name for the input port. A first-pass fix changed the port label to `"in"`.

### Why that was wrong

Holoscan's Python bindings expose the port with the label `"input"`, not `"in"`.
The C++ error string was a misleading internal log, not the Python-facing port name.
The user confirmed:

```
[error] [fragment.cpp:750] The downstream operator(udp_transmitter) does not have an
input port with label 'in'. It should be one of (input)
```

### Fix

```python
self.add_flow(payload_generator, udp_transmitter, {("output", "input")})
```

However, fixing the port name did not make the pipeline work. The pipeline appeared
to run but produced no UDP output — revealing a deeper problem.

---

## 3. Problem 2 — Python Can't Feed `UdpTransmitterOp` a Tensor

### Root cause investigation

Inspecting `udp_transmitter_op.cpp` revealed the operator's entire compute path:

```cpp
// Input port type
spec.input<std::shared_ptr<holoscan::Tensor>>(input_name);

// In compute():
auto maybe_tensor = op_input.receive<std::shared_ptr<holoscan::Tensor>>(input_name);
if (!maybe_tensor) {
    HOLOSCAN_LOG_ERROR("Failed to receive message from port 'in'");
    return;
}
auto& tensor = maybe_tensor.value();
// ... cudaMemcpyDeviceToHost ...
// ... sendto() ...
```

Critical points:
- The input type is `std::shared_ptr<holoscan::Tensor>` — a **GPU tensor** backed by
  device memory allocated through Holoscan's GXF memory system.
- It performs `cudaMemcpyDeviceToHost`, which means the tensor **must live in GPU
  memory** (not host/CPU memory).

The original Python `PayloadGeneratorOp` was emitting a NumPy array:

```python
data = np.array([self._payload_value], dtype=np.uint32)
op_output.emit(data, "output")
```

NumPy arrays live in **CPU memory**. Even if Holoscan's Python layer could somehow
wrap the array into a `holoscan::Tensor`, the tensor would point to host memory, and
`cudaMemcpyDeviceToHost` from a host pointer would silently produce garbage or crash.

### Intermediate fix attempt — CuPy

CuPy arrays live in GPU memory. The thinking was: if we hand CuPy data to the output
port, Holoscan might be able to wrap it as a valid GPU tensor.

```python
import cupy as cp
data = cp.array([self._payload_value], dtype=cp.uint32)
op_output.emit(cp.asnumpy(data).tobytes())  # wrong — converts back to CPU
```

This didn't work for two reasons:
1. The emit was being called with raw bytes (CPU), not a CuPy array.
2. Even if a CuPy array were passed, Holoscan's Python `OutputContext` doesn't know
   how to serialize an arbitrary CuPy array into the GXF message queue in a form that
   the C++ `receive<shared_ptr<holoscan::Tensor>>` call can deserialize.

### Why Python operators fundamentally can't satisfy `UdpTransmitterOp`

Holoscan's C++ `receive<T>()` goes through the `EmitterReceiverRegistry` — a type
registry that maps C++ types to GXF codecs (serialize/deserialize functions). For
`std::shared_ptr<holoscan::Tensor>` to be received correctly, the sender must have
used the matching GXF codec to emit it. Python operators emit via a different
mechanism (Python object → pybind11 → generic GXF message), which the C++ receiver
cannot decode as a `shared_ptr<holoscan::Tensor>`.

The only reliable way to produce a message that `UdpTransmitterOp` can consume is to
write a **C++ operator** that allocates a `nvidia::gxf::Tensor` in GPU memory and
wraps it properly.

---

## 4. Problem 3 — Decided to Write a C++ Operator

The approach: write `PayloadGeneratorOp` in C++ with pybind11 bindings so it can be
used from Python.

### What the C++ operator does

```cpp
// setup():
spec.output<std::shared_ptr<holoscan::Tensor>>("output");
spec.param(pool_, "pool", "Allocator", "Memory allocator");
spec.param(payload_value_, "payload_value", "Payload", "uint32 to send");

// compute():
// 1. Allocate a 4-byte tensor in GPU (device) memory
nvidia::gxf::Tensor gxf_tensor;
auto allocator = ...Handle<nvidia::gxf::Allocator>::Create(context, pool_->gxf_cid());
gxf_tensor.reshape<uint8_t>(
    nvidia::gxf::Shape{4},
    nvidia::gxf::MemoryStorageType::kDevice,   // ← GPU memory
    allocator.value()
);

// 2. Copy the 32-bit value from host to GPU
uint32_t val = payload_value_.get();
cudaMemcpy(gxf_tensor.pointer(), &val, sizeof(val), cudaMemcpyHostToDevice);

// 3. Wrap as holoscan::Tensor and emit
auto dl_ctx = std::make_shared<holoscan::DLManagedTensorContext>(...);
auto tensor = std::make_shared<holoscan::Tensor>(dl_ctx);
op_output.emit(tensor);   // ← same emit pattern as IQDecoderOp
```

This mirrors exactly how `AudioPacketizerOp` and `IQDecoderOp` (both confirmed-working
hololink operators) allocate and emit GPU tensors.

The C++ operator compiled cleanly. But importing it from Python opened a new series of
errors.

---

### 4a. Dynamic Linker Error

#### Symptom

```
ImportError: libpayload_generator_op.so: cannot open shared object file: No such file or directory
```

#### Cause

The initial `CMakeLists.txt` compiled the operator into two separate shared libraries:

```cmake
add_library(payload_generator_op SHARED ...)       # libpayload_generator_op.so
add_library(_payload_generator_op MODULE ...)       # the Python extension
target_link_libraries(_payload_generator_op PRIVATE payload_generator_op ...)
```

When Python loads `_payload_generator_op.so`, the dynamic linker tries to find
`libpayload_generator_op.so`. This file was not in any standard library search path,
and no `RPATH` was configured to point to the build directory.

#### Fix

Collapse both `.cpp` files into a single module — there is no need for a separate
intermediate library:

```cmake
add_library(_payload_generator_op MODULE
    payload_generator_op.cpp
    payload_generator_op_pybind.cpp)
```

Now there is only one `.so` file, and Python finds it via `sys.path`.

---

### 4b. Pybind11 Type Registry Mismatch

#### Symptom (appeared three separate times)

```
ImportError: generic_type: type "PayloadGeneratorOp" referenced unknown base type "holoscan::Operator"
```

#### What this error means

Pybind11 maintains an internal registry of all C++ types that have been exposed to
Python. When you write:

```cpp
py::class_<PayloadGeneratorOp, holoscan::Operator, ...>(m, "PayloadGeneratorOp")
```

pybind11 looks up `holoscan::Operator` in this registry. It must already be there
(registered by Holoscan's own pybind11 module) for the inheritance to work.

The registry is stored in a Python capsule whose name is a string like:

```
__pybind11_internals_v5__
```

**Every pybind11 module that is loaded into the same Python process must use the
exact same capsule name.** If the capsule names differ, the modules each maintain
separate, isolated type registries, and types registered by one module are invisible
to the other. That is why `holoscan::Operator` appeared as "unknown" to our module —
Holoscan registered it in registry A, but our module was looking in registry B.

---

### 4c. First Fix Attempt

#### Theory

Maybe `holoscan.core` (the module that registers `holoscan::Operator`) hadn't been
imported yet when our module tried to look up the base type. Force-import it first:

```cpp
PYBIND11_MODULE(_payload_generator_op, m) {
    py::module_::import("holoscan.core");   // ensure holoscan::Operator is registered
    py::class_<PayloadGeneratorOp, ...>(m, "PayloadGeneratorOp")
    ...
}
```

#### Why it didn't work

This addresses import ordering, not the capsule name mismatch. Even after importing
`holoscan.core`, both modules were still using different capsule names, so they still
had separate registries. The `holoscan::Operator` registration made by `holoscan.core`
into registry A was still invisible to our module in registry B.

---

### 4d. Second Fix Attempt

#### Theory

The capsule name includes a version number: `__pybind11_internals_v5__`. The `v5`
comes from `PYBIND11_INTERNALS_VERSION` in the pybind11 header. Holoscan was compiled
with pybind11 that used version 5. We were building with the latest pip pybind11
(version 3.0.3 at the time), which had bumped the version to `v11`. The capsule names
were `__pybind11_internals_v11__` vs `__pybind11_internals_v5__` — a trivial fix:
pin our pybind11 to a version that uses `v5`.

```bash
pip install 'pybind11==2.13.6'
```

Pybind11 2.13.6 uses `PYBIND11_INTERNALS_VERSION 5`. This should match.

#### Why it didn't work

After rebuilding, `strings _payload_generator_op.so | grep pybind11_internals` showed:

```
__pybind11_internals_v5_gcc_libstdcpp_cxxabi1018__
```

Holoscan's modules contained:

```
__pybind11_internals_v5__
```

The names still didn't match. The version number was right (`v5`), but a
**platform ABI suffix** was appended.

---

### 4e. Root Cause Found

Starting from pybind11 **2.12.0**, the capsule name was changed to include
compiler and ABI details:

```cpp
// pybind11/detail/internals.h (2.12+)
#define PYBIND11_INTERNALS_ID                           \
    "__pybind11_internals_v" PYBIND11_TOSTRING(PYBIND11_INTERNALS_VERSION) \
    PYBIND11_COMPILER_TYPE PYBIND11_STDLIB PYBIND11_BUILD_ABI "__"
```

Where on a typical Linux GCC system:

```cpp
#define PYBIND11_COMPILER_TYPE "_gcc"
#define PYBIND11_STDLIB        "_libstdcpp"
#define PYBIND11_BUILD_ABI     "_cxxabi1018"
```

Resulting in: `__pybind11_internals_v5_gcc_libstdcpp_cxxabi1018__`

Holoscan was compiled with a version of pybind11 **before 2.12** (or with patched
definitions that set all three macros to empty strings), producing the plain:
`__pybind11_internals_v5__`

---

### 4f. Final Fix

Force all three ABI macros to empty strings at compile time. In `CMakeLists.txt`:

```cmake
target_compile_definitions(_payload_generator_op PRIVATE
    "PYBIND11_COMPILER_TYPE=\"\""
    "PYBIND11_STDLIB=\"\""
    "PYBIND11_BUILD_ABI=\"\"")
```

After rebuilding:

```bash
strings payload_generator_op/_payload_generator_op.so | grep pybind11_internals
```

Output:

```
__pybind11_internals_v5__
```

This now matched Holoscan's modules exactly. The `ImportError` disappeared. The C++
operator registered `PayloadGeneratorOp` successfully inheriting from
`holoscan::Operator`.

---

## 5. Problem 4 — Scheduler Deadlock (Transmitter Never Runs)

With the import fixed, the pipeline ran — but the transmitter never executed:

```
[info] [payload_generator_op.cpp:75] PayloadGeneratorOp: emitting 0xDEADBEEF  bytes=efbeadde
[info] [greedy_scheduler.cpp:372] Scheduler stopped: Some entities are waiting for execution,
       but there are no periodic or async entities to get out of the deadlock.
```

The generator ran and emitted. The transmitter was described as "waiting for execution"
(i.e., ready and has data, but never actually scheduled). The pipeline ended without
the transmitter running at all.

### 5a. What the Greedy Scheduler Does

Holoscan's default `GreedyScheduler` runs a tight polling loop:

```
loop:
  1. Evaluate the scheduling condition of every entity.
  2. If any entity is READY → run the first one, go to 1.
  3. If no entity is READY:
       a. If some entities are in WAIT_EXECUTION (ready but unscheduled) AND
          there are no periodic/async entities that could change the state
          → declare DEADLOCK, stop.
       b. Otherwise → stop normally.
```

An entity's scheduling condition is composed of all the `SchedulingTerm`s attached to
it. For the generator, that's a `CountCondition`. For the transmitter, there's an
auto-attached `MessageAvailableSchedulingTerm` on its input port that becomes READY
when a message is in the queue.

### 5b. Why It Deadlocks Here

The sequence with `CountCondition(count=1)` on the generator:

| Step | Generator state | Transmitter state | Scheduler action |
|------|----------------|-------------------|-----------------|
| Start | READY (count=1) | WAIT (no data) | Run generator |
| After generator emits | **NEVER** (count=0) | WAIT_EXECUTION (has data!) | ??? |

After the generator runs once its `CountCondition` reaches zero. GXF marks it as
`kNever` — it will never be scheduled again. At this point the transmitter has a
message waiting in its input queue, so its `MessageAvailableSchedulingTerm` evaluates
to READY (`WAIT_EXECUTION`).

The Greedy Scheduler should run the transmitter next. But it doesn't — it fires the
deadlock check instead. The reason: the deadlock detector sees that the **only source
of data** for the transmitter (the generator) is in `kNever` state. It concludes that
while the transmitter can run *once more*, it will then starve permanently with no way
out. Rather than run it once and then face an unavoidable stall, the scheduler
preemptively declares deadlock and stops.

This is a known limitation of the `GreedyScheduler` with short, bounded pipelines:
it sacrifices the last remaining execution of downstream operators to avoid an
eventual stall.

### 5c. Fix — Switch to `EventBasedScheduler`

The `EventBasedScheduler` is fundamentally different. Instead of polling a loop, it
uses condition variables and an event queue:

- When an entity emits a message to a downstream port, an **event** is posted to the
  scheduler's event queue.
- Worker threads sleep until events arrive; on receiving an event, they wake and run
  the entity whose condition just became satisfied.
- The scheduler only stops when **all** work has been processed and no more events can
  arrive.

With this scheduler:

1. Generator runs → emits tensor → posts "MessageAvailable" event for transmitter.
2. Worker thread picks up the event → transmitter runs → sends UDP datagram.
3. No more events possible → scheduler shuts down cleanly.

The "generator is NEVER" situation is handled correctly because the scheduler doesn't
reason about long-term starvation — it only asks: *is there a pending event right
now?*

```python
self.scheduler(holoscan.schedulers.EventBasedScheduler(
    self,
    name="event_scheduler",
    worker_thread_number=2,
))
```

After this change:

```
INFO:root:PythonUdpTransmitterOp: UDP socket ready → 192.168.0.2:4791
INFO:root:PayloadGeneratorOp: emitting 0xDEADBEEF  bytes=efbeadde
INFO:root:PythonUdpTransmitterOp: sent 4 bytes to 192.168.0.2:4791
[info] [event_based_scheduler.cpp:684] Total execution time: 1.946334 ms
```

Both operators ran, the packet was sent, and the pipeline terminated cleanly.

---

## 6. Final Working Architecture

```
thor_send_payload.py
│
├── PayloadGeneratorOp  (Python operator)
│     • Emits a numpy uint32 array each tick
│     • Controlled by CountCondition(N) or PeriodicCondition(1 Hz)
│
├── PythonUdpTransmitterOp  (Python operator)
│     • Receives the array, calls tobytes(), sends via socket.sendto()
│     • Pure Python — no GPU tensors, no C++ library dependencies
│
└── EventBasedScheduler  (2 worker threads)
      • Event-driven: wakes transmitter the moment generator emits
      • Stops cleanly after all pending events are drained
```

The pure-Python transmitter (`PythonUdpTransmitterOp`) replaced the C++
`hololink.operators.UdpTransmitterOp`. This sidesteps the GPU tensor requirement
entirely — the FPGA receives raw UDP bytes regardless of whether the sender used
the C++ tensor path or a plain socket.

The C++ `PayloadGeneratorOp` and its pybind11 binding were fully built and debugged
(solving the dynamic linker and pybind11 internals problems), but the script ultimately
uses the simpler Python-only path, which is sufficient for plain-UDP sends on Thor
where a ConnectX NIC is not present.

---

## 7. Key Lessons

| # | Lesson |
|---|--------|
| 1 | **Port name mismatch between GXF internals and Python bindings.** C++ log messages often show internal GXF port names (e.g., `"in"`) that differ from the Python-facing names (e.g., `"input"`). Trust the Python API error, not the C++ log string. |
| 2 | **Python operators cannot produce C++ GPU tensors.** `UdpTransmitterOp` expects `std::shared_ptr<holoscan::Tensor>` allocated in device memory via GXF's allocator system. A Python `emit()` goes through a different codec path and produces a message the C++ receiver cannot deserialize. |
| 3 | **Pybind11's type registry is keyed on an exact capsule name string.** All modules in the same process must produce the exact same `__pybind11_internals_vN[_suffix]__` string. Version mismatches or ABI suffix differences silently create parallel registries, breaking class inheritance. Force empty ABI macros (`PYBIND11_COMPILER_TYPE=""`, `PYBIND11_STDLIB=""`, `PYBIND11_BUILD_ABI=""`) to match older SDK builds that predate pybind11 2.12's platform-suffix addition. |
| 4 | **`GreedyScheduler` can starve downstream operators in bounded pipelines.** When an upstream source is exhausted (`kNever`), the scheduler may stop before flushing remaining messages in downstream queues. Use `EventBasedScheduler` for pipelines where the data source runs fewer times than the consumer. |
| 5 | **Plain Python UDP sockets work fine for sending to the FPGA.** If the FPGA just needs raw bytes on port 4791, there is no requirement to use the C++ `UdpTransmitterOp` or GPU tensors. A `socket.sendto()` call produces an identical UDP datagram at the network level. |
