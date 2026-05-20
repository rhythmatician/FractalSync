#!/usr/bin/env bash
set -euo pipefail

# Generate runtime_core Python stubs and write them to backend/stubs/runtime_core
# Usage: ./scripts/generate_runtime_core_stubs.sh

pushd runtime-core
maturin develop --release
popd

python -m pyo3_stubgen.generate runtime_core -o backend/stubs/runtime_core --package-name runtime_core || true

# Additionally attempt to produce stubs from metadata exported by the bindings
python scripts/generate_runtime_core_stubs_from_metadata.py -o backend/stubs/runtime_core || true

echo "Runtime core stubs generated to backend/stubs/runtime_core/"