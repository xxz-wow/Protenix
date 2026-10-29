#!/usr/bin/env bash
set -euo pipefail
python -B /workspace/benchmarks/run_benchmark.py --dataset eval21 --gpu "${PROTENIX_GPU:-6}" --output-dir "${EVAL21_OUTPUT_ROOT:-/workspace/output/eval21_retest}"
