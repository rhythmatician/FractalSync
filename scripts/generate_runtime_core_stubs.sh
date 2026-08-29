#!/usr/bin/env bash
set -euo pipefail

# Generate runtime_core Python stubs and write them to backend/stubs/runtime_core
# Usage: ./scripts/generate_runtime_core_stubs.sh

pushd runtime-core
maturin develop --release
popd

# Generate the stubs by introspecting the live module
python scripts/generate_runtime_core_stubs.py -o backend/stubs/runtime_core

echo "Runtime core stubs generated to backend/stubs/runtime_core/"
