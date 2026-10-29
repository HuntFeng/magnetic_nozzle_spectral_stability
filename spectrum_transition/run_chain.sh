#!/bin/bash
# wait until the unmapped timing scan has been stopped, then run the mapped sweep, transonic scan, continuation
cd "$(dirname "$0")"
while pgrep -f scan_timing.py >/dev/null; do sleep 10; done
echo "=== sweep start $(date)" 
uv run python scan_sweep.py 0.5 0.8 0.9 0.95 0.98 0.99 0.995 0.999 1.001 1.005 1.01 1.02 1.05 1.1 1.3 1.5 > data/sweep.log 2>&1
echo "=== transonic start $(date)"
uv run python transonic_scan.py > data/transonic.log 2>&1
echo "=== continuation start $(date)"
uv run python continuation.py fixed_open > data/continuation_fixed_open.log 2>&1
uv run python continuation.py fixed_fixed > data/continuation_fixed_fixed.log 2>&1
echo "=== chain done $(date)"
