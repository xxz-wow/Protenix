#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mcc="${MCC:-${MUSA_HOME:-/usr/local/musa}/bin/mcc}"
out="${1:-$script_dir/libv21_training_fast_p36q32_repro.so}"
mkdir -p "$script_dir/compiler_diagnostics"
"$mcc" \
  -O3 --use_fast_math -std=c++17 --cuda-gpu-arch=mp_31 \
  -DV21_BLOCK_M=64 -DV21_WARPS=4 \
  -DV21_QSTRIDE=32 -DV21_KSTRIDE=40 -DV21_PSTRIDE=36 \
  -DV21_SPECIAL512 -DV21_PRESCALED -DV21_NO_BIAS1 \
  -DV21_SHARED_P -DV21_DIRECT_OUT \
  -fPIC -shared -resource-usage -x musa \
  "$script_dir/tri_attn_musa_v21_training.mu" -o "$out" \
  2>&1 | tee "$script_dir/compiler_diagnostics/training-fast-p36q32-repro.log"
sha256sum "$script_dir/tri_attn_musa_v21_training.mu" "$out"
