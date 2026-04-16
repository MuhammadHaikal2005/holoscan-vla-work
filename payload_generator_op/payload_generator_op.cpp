// PayloadGeneratorOp implementation.
// Mirrors AudioPacketizerOp's tensor creation pattern exactly so that
// UdpTransmitterOp::compute can receive<shared_ptr<holoscan::Tensor>> it.

#include "payload_generator_op.hpp"

#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

// GXF tensor types are pulled in via holoscan/holoscan.hpp → gxf headers
#include <gxf/std/tensor.hpp>

namespace hsb_groot {

void PayloadGeneratorOp::setup(holoscan::OperatorSpec& spec)
{
    spec.output<std::shared_ptr<holoscan::Tensor>>("output");

    spec.param(payload_value_,
        "payload_value",
        "Payload Value",
        "32-bit value to send to the FPGA",
        static_cast<uint32_t>(0xDEADBEEF));

    spec.param(pool_,
        "pool",
        "Pool",
        "Allocator used to allocate the GPU tensor");
}

void PayloadGeneratorOp::compute(
    [[maybe_unused]] holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    // ── 1. Allocate a 4-byte tensor in device (GPU) memory ──────────────────
    nvidia::gxf::Tensor gxf_tensor;

    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), pool_->gxf_cid());

    if (!allocator) {
        HOLOSCAN_LOG_ERROR("PayloadGeneratorOp: failed to get allocator handle");
        return;
    }

    constexpr int payload_bytes = static_cast<int>(sizeof(uint32_t));

    auto reshape_result = gxf_tensor.reshape<uint8_t>(
        nvidia::gxf::Shape{payload_bytes},
        nvidia::gxf::MemoryStorageType::kDevice,
        allocator.value());

    if (!reshape_result) {
        HOLOSCAN_LOG_ERROR("PayloadGeneratorOp: failed to allocate GPU tensor");
        return;
    }

    // ── 2. Copy payload value to GPU ─────────────────────────────────────────
    uint32_t val = payload_value_.get();
    cudaError_t err = cudaMemcpy(
        gxf_tensor.pointer(),
        &val,
        sizeof(val),
        cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("PayloadGeneratorOp: cudaMemcpy failed: {}",
            cudaGetErrorString(err));
        return;
    }

    HOLOSCAN_LOG_INFO("PayloadGeneratorOp: emitting 0x{:08X}  bytes={:02x}{:02x}{:02x}{:02x}",
        val,
        static_cast<uint8_t>(val & 0xFF),
        static_cast<uint8_t>((val >> 8) & 0xFF),
        static_cast<uint8_t>((val >> 16) & 0xFF),
        static_cast<uint8_t>((val >> 24) & 0xFF));

    // ── 3. Wrap as holoscan::Tensor and emit ─────────────────────────────────
    auto maybe_dl_ctx = gxf_tensor.toDLManagedTensorContext();
    if (!maybe_dl_ctx) {
        HOLOSCAN_LOG_ERROR("PayloadGeneratorOp: toDLManagedTensorContext failed");
        return;
    }

    auto dl_ctx = std::make_shared<holoscan::DLManagedTensorContext>(
        std::move(*maybe_dl_ctx.value()));
    auto tensor = std::make_shared<holoscan::Tensor>(dl_ctx);

    op_output.emit(tensor);
}

} // namespace hsb_groot
