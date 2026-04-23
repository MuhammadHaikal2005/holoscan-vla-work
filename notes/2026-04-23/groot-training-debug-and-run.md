# GR00T N1.6 Training Debug Session — 3090 PC

**Date:** 2026-04-23  
**Goal:** Get `finetune_dual_usb.sh` running to completion on the 3090 PC and transfer the resulting checkpoint back to the Jetson AGX Thor.  
**Outcome:** Training completed successfully (2000 steps). Checkpoint transferred back to Thor via rsync over direct Ethernet.

---

## Starting state

Files had been transferred from the Thor the day before (see `notes/2026-04-22/transfer-training-to-3090.md`). The 3090 PC had:

- `~/holoscan-vla-work/` — project repo (GitHub clone name, not `hsb-groot-robot`)
- `~/Isaac-GR00T/` — training codebase, venv already built by `uv sync`
- `~/holoscan-vla-work/datasets/dual_cam_blue_only/` — dataset in v2.1 format
- `~/holoscan-vla-work/models/base/GR00T-N1.6-3B/` — base model weights
- `~/holoscan-vla-work/training/` — training scripts

The plan from the day before (see `notes/2026-04-22/run-training-on-3090.md`) laid out 7 steps. This session executed those steps and hit several unexpected issues along the way.

---

## Step 1 — Symlink: `~/hsb-groot-robot` → `~/holoscan-vla-work`

### Why this was needed

The training scripts (`finetune_dual_usb.sh` and `launch_finetune.py`) hardcode `$HOME/hsb-groot-robot` as the project root. That was the folder name on the Thor. On the 3090 PC the repo was cloned from GitHub and landed as `holoscan-vla-work`. Rather than editing every hardcoded path in multiple scripts, a symlink makes the OS transparently resolve `~/hsb-groot-robot` to `~/holoscan-vla-work`.

### Command

```bash
ln -s ~/holoscan-vla-work ~/hsb-groot-robot
```

### Verification

```bash
ls -la ~/ | grep hsb-groot-robot
# lrwxrwxrwx  1 u3090 u3090  29 Apr 23 10:48 hsb-groot-robot -> /home/u3090/holoscan-vla-work
```

Any path that starts with `~/hsb-groot-robot/` now resolves correctly. Scripts don't need to be edited.

---

## Step 2 — Remove the aarch64 torchcodec wheel check

### What the original code did

`finetune_dual_usb.sh` had a self-healing block written for the Thor (aarch64 / FFmpeg 6):

```bash
TC_WHEEL="$GROOT_DIR/wheels/torchcodec-0.10.0a0-cp312-cp312-linux_aarch64.whl"
TC_CORE6="$GROOT_DIR/.venv/lib/python3.12/site-packages/torchcodec/libtorchcodec_core6.so"
if [[ ! -f "$TC_CORE6" ]]; then
    echo "torchcodec: FFmpeg-6 build missing — reinstalling from cached wheel..."
    uv pip install --python "$VENV_PYTHON" --no-deps "$TC_WHEEL"
    echo "torchcodec: reinstalled OK"
fi
```

### Why it would fail on x86_64

The wheel filename (`linux_aarch64.whl`) is ARM64-only. On x86_64 the install would fail. The file `libtorchcodec_core6.so` also doesn't exist on x86_64 because x86_64 PyPI wheels link against a different FFmpeg version. The condition would always evaluate to true and then try to force-install an incompatible binary.

### The fix

Replaced the entire block with a plain import check:

```bash
# ── torchcodec: verify the x86_64 wheel loaded correctly ──────────────────────
if ! "$VENV_PYTHON" -c "import torchcodec" 2>/dev/null; then
    echo "ERROR: torchcodec failed to load. Try: sudo apt install ffmpeg"
    exit 1
fi
```

This is architecture-agnostic. It simply confirms torchcodec is importable (which it was — x86_64 wheels installed cleanly from PyPI during `uv sync`), and if not, gives a clear actionable error message.

---

## Pre-flight sanity checks

Before attempting to run training, confirmed:

```bash
~/Isaac-GR00T/.venv/bin/python3 -c "import torchcodec; print('torchcodec OK')"
# torchcodec OK

~/Isaac-GR00T/.venv/bin/python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# True
# NVIDIA GeForce RTX 3090
```

CUDA was visible and the correct GPU was detected.

---

## First run attempt — Terminal crash (SIGTERM / ble.sh)

### What happened

Running the training command directly in the Cursor IDE terminal:

```bash
cd ~/holoscan-vla-work
bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only
```

The terminal appeared to crash. Output just stopped and the prompt returned. Training never produced any loss values.

### Diagnosis

The output ended with:

```
[ble: EOF]
u3090@u3090-MS-7E24:~$
```

`ble.sh` is a bash line-editor enhancement (`bash-line-editor`) installed on this machine. It intercepts stdin/stdout to provide syntax highlighting and history in the shell. When the Cursor IDE terminal window lost focus or the PTY had an issue, `ble.sh` raised an EOF condition and terminated the entire shell session — which sent SIGHUP to the foreground process group, killing the training script.

This was **not** a training failure. The training process itself was starting correctly (the logs showed `Current global step: 0` and `Creating custom train dataloader`) but was killed externally by the dying shell.

### First attempted fix

Run inside `tmux`, which gives the training process its own PTY completely decoupled from the IDE terminal:

```bash
tmux new-session -s training
# inside tmux:
cd ~/holoscan-vla-work
bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only
```

---

## Second run — CUDA Out of Memory (batch size 16)

### What happened

Running inside tmux, training progressed further — workers started caching shards, the model loaded, and the first forward pass completed. But at the optimizer step it crashed:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 36.00 MiB.
GPU 0 has a total capacity of 23.55 GiB of which 39.00 MiB is free.
This process has 23.01 GiB memory in use. Of the allocated memory 21.52 GiB
is allocated by PyTorch.
```

The crash happened inside `adam.py` at:
```python
state["exp_avg"] = torch.zeros_like(...)
```

### Initial diagnosis (incomplete)

The first hypothesis was that batch size 16 was too large for the 3090's 24 GB. Batch size controls how many samples are in a forward/backward pass, which affects activation memory. Halving the batch size to 8 seemed like a reasonable first attempt.

Also added the allocator hint PyTorch's own error message suggested:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This tells PyTorch's CUDA memory allocator to use expandable memory segments, which reduces fragmentation and can make it possible to satisfy small allocations even when memory is tight.

### Fix applied

In `finetune_dual_usb.sh`:
```bash
--global-batch-size 8 \   # was 16
```

---

## Third run — CUDA Out of Memory again (batch size 8)

### What happened

Same traceback, same location in `adam.py`, but this time at `exp_avg_sq`:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.00 MiB.
GPU 0 has a total capacity of 23.55 GiB of which 46.00 MiB is free.
This process has 23.05 GiB memory in use. Of the allocated memory 22.41 GiB
is allocated by PyTorch.
```

Notably, with batch size 8 PyTorch was using **more** memory (22.41 GiB vs 21.52 GiB). This was the clue that batch size was not the actual problem.

### Root cause analysis

Inspecting `launch_finetune.py` revealed two key settings:

```python
config.model.backbone_trainable_params_fp32 = True   # backbone weights in float32
config.training.optim = "adamw_torch"                # standard Adam optimizer
config.training.gradient_checkpointing = False        # default, not set
```

**Memory breakdown for a 3B parameter model:**

| Component | Precision | Size |
|---|---|---|
| Model weights (backbone in fp32) | float32 | ~12 GB |
| Adam `exp_avg` (1st moment) | float32 | ~12 GB |
| Adam `exp_avg_sq` (2nd moment) | float32 | ~12 GB |
| Activations (batch size 8) | bfloat16 | ~1–2 GB |
| **Total** | | **~37–38 GB** |

The 3090 has 24 GB. No batch size reduction can fix a ~37 GB optimizer + model requirement. The batch size was never the real problem — the optimizer state alone exceeded GPU capacity.

Batch size affects **activation memory** (proportional to batch size × sequence length × hidden dim). It does **not** affect the size of model weights or optimizer states. Those scale with the number of trainable parameters, which is fixed regardless of batch size.

### Why the memory looked worse with batch size 8

With batch size 16, the first OOM hit at `exp_avg` allocation (~36 MiB requested). With batch size 8, PyTorch was slightly further into optimizer state initialization (22.41 GiB used, hit at `exp_avg_sq`) because smaller activations left fractionally more headroom before the catastrophic allocations began. The underlying issue was identical.

### The actual fix

**8-bit Adam via bitsandbytes.** Instead of storing optimizer moments in float32, `adamw_bnb_8bit` quantizes them to 8-bit integers with dynamic block-wise scaling. This reduces optimizer state memory by 4×:

| State tensor | Before (fp32) | After (8-bit) |
|---|---|---|
| `exp_avg` | ~12 GB | ~3 GB |
| `exp_avg_sq` | ~12 GB | ~3 GB |
| **Total optimizer** | ~24 GB | ~6 GB |

**Updated memory breakdown:**

| Component | Size |
|---|---|
| Model weights (fp32 backbone) | ~12 GB |
| 8-bit Adam states | ~6 GB |
| Activations + workspace | ~2–3 GB |
| **Total** | **~20–21 GB ✓** |

This fits comfortably within 24 GB.

**Gradient checkpointing** was also enabled as a secondary measure. Instead of holding all intermediate activations in VRAM for the backward pass, gradient checkpointing recomputes them on demand. This trades ~20–30% extra compute for a significant reduction in peak activation memory, giving more headroom for the optimizer state.

### Installation and changes

```bash
# Install bitsandbytes into the Isaac-GR00T venv
cd ~/Isaac-GR00T
uv pip install --python .venv/bin/python3 bitsandbytes
# Installed bitsandbytes==0.49.2
```

In `~/Isaac-GR00T/gr00t/experiment/launch_finetune.py`:

```python
# Before:
config.training.optim = "adamw_torch"

# After:
config.training.optim = "adamw_bnb_8bit"
config.training.gradient_checkpointing = True
```

---

## Final run — success

Inside a `tmux` session (`tmux new-session -s training`):

```bash
cd ~/holoscan-vla-work
bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only
```

Training started cleanly:

```
Current global step: 0
Creating custom train dataloader
  0%|          | 0/2000 ...
{'loss': ..., 'grad_norm': ..., 'learning_rate': ...}
  0%|          | 1/2000 [...]
```

Checkpoints saved every 500 steps to `~/holoscan-vla-work/checkpoints/dual_cam_blue_only/`.
Training completed at step 2000.

---

## Transferring the checkpoint back to the Thor

Re-established the direct Ethernet link (192.168.100.1 = Thor), then:

```bash
rsync -av --progress \
    ~/holoscan-vla-work/checkpoints/dual_cam_blue_only/checkpoint-2000/ \
    latticeapp@192.168.100.1:~/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-2000/
```

On the Thor, the inference server is pointed at:

```
/home/latticeapp/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-2000
```

---

## Summary of all changes made

| File | Change | Reason |
|---|---|---|
| `~/.bashrc` (via shell) | `ln -s ~/holoscan-vla-work ~/hsb-groot-robot` | Scripts hardcode `hsb-groot-robot` path |
| `training/finetune_dual_usb.sh` | Replaced aarch64 torchcodec wheel block with plain import check | ARM64 wheel incompatible with x86_64 |
| `training/finetune_dual_usb.sh` | `--global-batch-size 16` → `8` | Minor headroom improvement (not the primary fix) |
| `training/finetune_dual_usb.sh` | Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduces CUDA allocator fragmentation |
| `training/finetune_dual_usb.sh` | `--dataloader-num-workers 4` → `2` | Reduce RAM pressure during shard caching |
| `Isaac-GR00T/.venv` | Installed `bitsandbytes==0.49.2` | Required for 8-bit Adam optimizer |
| `Isaac-GR00T/gr00t/experiment/launch_finetune.py` | `optim = "adamw_bnb_8bit"` | Primary OOM fix — cuts optimizer state from ~24 GB to ~6 GB |
| `Isaac-GR00T/gr00t/experiment/launch_finetune.py` | `gradient_checkpointing = True` | Secondary OOM fix — reduces peak activation memory |

---

## Key lessons

1. **Batch size does not fix optimizer OOM.** Batch size only affects activation memory. If you're OOMing inside `optimizer.step()` (Adam moment tensors), batch size is the wrong knob. The fix is a memory-efficient optimizer.

2. **Adam is expensive for large models.** Standard fp32 Adam needs 2× the model's parameter count in float32 just for optimizer states. For a 3B model with fp32 backbone, that's ~24 GB before activations or other tensors are considered.

3. **8-bit Adam (bitsandbytes) is the correct fix.** `adamw_bnb_8bit` reduces optimizer state by 4× with minimal convergence impact. It is the standard approach for fine-tuning large models on consumer GPUs.

4. **`ble.sh` is incompatible with long-running training in Cursor's IDE terminal.** Always run training inside `tmux`. Detach with `Ctrl+B D`, reattach with `tmux attach -t training`.

5. **Architecture-specific code must be gated.** The aarch64 wheel check had no architecture guard. Always wrap platform-specific code in `[[ "$(uname -m)" == "aarch64" ]]` checks.
