#!/usr/bin/env bash
# 在 Linux C500/MACA 环境执行；产物默认为 build/libtri_attention_backward.so。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export PATH="${MACA_PATH}/tools/cu-bridge/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
CUCC="${MACA_PATH}/tools/cu-bridge/bin/cucc"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
if [[ ! -x "${CUCC}" ]]; then
  printf 'MACA compiler not found: %s\n' "${CUCC}" >&2
  exit 1
fi
mkdir -p "${BUILD_DIR}"
COMMON=(
  -O3 --use_fast_math -ffast-math -std=c++17 -arch=sm_80
  --expt-relaxed-constexpr --expt-extended-lambda -fPIC
  "-I${ROOT}/third_party/mctlass/include" "-I${ROOT}/src"
  "-I${MACA_PATH}/include" "-I${MACA_PATH}/include/mcr"
  "-I${MACA_PATH}/tools/cu-bridge/include"
  -DUSE_KERNEL_VERSIONED -DUSE_MACA -DNV_ARCH_A100 -D__FAST_HALF_CVT__
)
for UNIT in triangle_attention_forward triangle_attention_backward pair_bias_cache; do
  EXTRA=()
  if [[ "${UNIT}" == triangle_attention_forward ]]; then
    EXTRA=(-Xclang -menable-no-nans)
  fi
  printf 'Compiling %s.cu\n' "${UNIT}"
  "${CUCC}" "${COMMON[@]}" "${EXTRA[@]}" -c "${ROOT}/src/${UNIT}.cu" -o "${BUILD_DIR}/${UNIT}.o"
done
"${CUCC}" -shared \
  "${BUILD_DIR}/triangle_attention_forward.o" \
  "${BUILD_DIR}/triangle_attention_backward.o" \
  "${BUILD_DIR}/pair_bias_cache.o" \
  "-L${MACA_PATH}/lib" -L/opt/mxdriver/lib -lmcruntime \
  -o "${BUILD_DIR}/libtri_attention_backward.so"
printf 'Built %s/libtri_attention_backward.so\n' "${BUILD_DIR}"
