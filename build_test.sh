#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cmake --preset default
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
