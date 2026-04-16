// pybind11 bindings for PayloadGeneratorOp.
// Mirrors the pattern used in AudioPacketizerOp's Python bindings.

#include "payload_generator_op.hpp"

#include <pybind11/pybind11.h>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

using pybind11::literals::operator""_a;
namespace py = pybind11;

namespace hsb_groot {

// Trampoline class so pybind11 can construct the operator with explicit args
// rather than requiring the variadic ArgList constructor directly from Python.
class PyPayloadGeneratorOp : public PayloadGeneratorOp {
public:
    using PayloadGeneratorOp::PayloadGeneratorOp;

    PyPayloadGeneratorOp(
        holoscan::Fragment* fragment,
        uint32_t payload_value,
        std::shared_ptr<holoscan::Allocator> pool,
        const std::string& name)
        : PayloadGeneratorOp(holoscan::ArgList{
            holoscan::Arg{"payload_value", payload_value},
            holoscan::Arg{"pool", pool}})
    {
        name_     = name;
        fragment_ = fragment;
        spec_     = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_payload_generator_op, m)
{
    m.doc() = "PayloadGeneratorOp — emits a uint32 GPU tensor for UdpTransmitterOp";

    // Force holoscan.core to initialise and register holoscan::Operator in
    // pybind11's type registry before we try to subclass it.
    py::module_::import("holoscan.core");

    py::class_<PayloadGeneratorOp,
               PyPayloadGeneratorOp,
               holoscan::Operator,
               std::shared_ptr<PayloadGeneratorOp>>(m, "PayloadGeneratorOp")
        .def(py::init<holoscan::Fragment*,
                      uint32_t,
                      std::shared_ptr<holoscan::Allocator>,
                      const std::string&>(),
            "fragment"_a,
            "payload_value"_a = static_cast<uint32_t>(0xDEADBEEF),
            "pool"_a,
            "name"_a = "payload_generator");
}

} // namespace hsb_groot
