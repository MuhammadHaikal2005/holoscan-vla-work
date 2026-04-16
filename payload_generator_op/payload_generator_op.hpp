// PayloadGeneratorOp — emits a 32-bit value as a GPU-backed holoscan::Tensor
// so that UdpTransmitterOp (which calls receive<shared_ptr<Tensor>>) can receive it.
//
// Must be used with an UnboundedAllocator passed as the `pool` parameter.

#pragma once

#include <memory>

#include <holoscan/holoscan.hpp>

namespace hsb_groot {

class PayloadGeneratorOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(PayloadGeneratorOp)

    PayloadGeneratorOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;
    void compute(holoscan::InputContext& op_input,
                 holoscan::OutputContext& op_output,
                 holoscan::ExecutionContext& context) override;

private:
    holoscan::Parameter<uint32_t> payload_value_;
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> pool_;
};

} // namespace hsb_groot
