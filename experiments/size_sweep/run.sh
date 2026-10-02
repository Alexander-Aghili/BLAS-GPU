#!/usr/bin/env bash
# Run the sweep (default 3 passes), logging GPU clocks/temperature/power, then
# aggregate to results.csv (median across passes of each pass's median).
set -euo pipefail
cd "$(dirname "$0")"
SWEEPS=${1:-3}
q=clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate,clocks_throttle_reasons.active
nvidia-smi --query-gpu=name,driver_version,$q --format=csv > clocks_before.csv
nvidia-smi --query-gpu=timestamp,$q --format=csv -lms 1000 > clocks_during.csv &
mon=$!
trap 'kill $mon 2>/dev/null || true' EXIT
./sweep 1 > warmup_raw.csv           # one discarded pass to warm the GPU
./sweep "$SWEEPS" > raw.csv
kill $mon
nvidia-smi --query-gpu=name,driver_version,$q --format=csv > clocks_after.csv
python3 aggregate.py raw.csv results.csv
