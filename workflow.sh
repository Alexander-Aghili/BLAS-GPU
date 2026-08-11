#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

kernel="${1:-dot}"

cmake --build build -j

mkdir -p reports
out="reports/${kernel}_$(date +%Y%m%d_%H%M%S)"

ncu_cmd=(/usr/local/cuda/bin/ncu)
if grep -q "RmProfilingAdminOnly: 1" /proc/driver/nvidia/params 2>/dev/null; then
    echo "profiling counters are admin-only (RmProfilingAdminOnly=1), running ncu with sudo"
    ncu_cmd=(sudo /usr/local/cuda/bin/ncu)
fi

"${ncu_cmd[@]}" --kernel-name "regex:${kernel}" --launch-skip 1 --launch-count 3 --set full -o "$out" ./build/bench

echo
echo "report written: ${out}.ncu-rep"
echo "open with:      ncu-ui ${out}.ncu-rep"
