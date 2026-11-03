#!/usr/bin/env bash
set -euo pipefail
python -B /workspace/benchmarks/run_benchmark.py --dataset test23 --gpu "${PROTENIX_GPU:-6}" --output-dir "${TEST23_OUTPUT_ROOT:-/workspace/output/test23_retest}"
