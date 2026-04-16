#!/usr/bin/env bash
# Build PayloadGeneratorOp inside the hololink-demo Docker container.
# Run this from the repo root: bash payload_generator_op/build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Must use pybind11 2.13.x — same PYBIND11_INTERNALS_VERSION (5) as holoscan/hololink.
# pybind11 3.x uses internals_v11 which is incompatible with holoscan's type registry.
pip install -q 'pybind11==2.13.6'
PYBIND11_INCLUDE="$(python3 -c 'import pybind11; print(pybind11.get_include())')"
echo "Using pybind11 from: ${PYBIND11_INCLUDE}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${SCRIPT_DIR}" \
    -DPYBIND11_INCLUDE_DIR="${PYBIND11_INCLUDE}"

make -j"$(nproc)"
make install

echo ""
echo "Build complete. Shared libraries installed to:"
ls -lh "${SCRIPT_DIR}"/*.so 2>/dev/null || echo "  (no .so files found — check build output above)"
