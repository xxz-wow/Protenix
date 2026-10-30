#pragma once
/**
 * triangle_attention_config.h — 前后向共用的 BF16 MMA、共享内存和拷贝布局
 *
 * 基于 fa2_minimal 的 CuTe/mctlass 架构,
 * 加入 Triangle Attention 的 Bias1/Bias2 支持.
 *
 * 不依赖 kernel_config.h (使用独立的 CuTe 类型体系)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>

// CuTe headers
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/copy_traits_sm80.hpp>

// mctlass headers
#include <mctlass/mctlass.h>
#include <mctlass/array.h>
#include <mctlass/numeric_types.h>
#include <mctlass/numeric_conversion.h>
#include <mctlass/arch/maca_memory.h>

using namespace cute;

// ============================================================
// Constants
// ============================================================
#ifndef BLOCK_M
#define BLOCK_M 64
#endif
#ifndef NUM_WAVES
#define NUM_WAVES (BLOCK_M / 16)
#endif
#ifndef BLOCK_N
#define BLOCK_N 64
#endif
#ifndef HEAD_DIM
#define HEAD_DIM 32
#endif

static constexpr int kBlockM = BLOCK_M;
static constexpr int kBlockN = BLOCK_N;
static constexpr int kHeadDim = HEAD_DIM;
static constexpr int kNWarps = NUM_WAVES;
static constexpr int kNThreads = kNWarps * 64; // 256

static_assert(kBlockM % 16 == 0, "BLOCK_M must be a multiple of 16");
static_assert(kBlockN == 64 || kBlockN == 128,
              "the first long-sequence candidate supports BLOCK_N 64 or 128");
static_assert(kNWarps == kBlockM / 16, "NUM_WAVES must match BLOCK_M / 16 for the M-partition");
static_assert(kHeadDim == 32, "the first long-sequence candidate is validated for D=32");

static constexpr int kBlockKSmem = 32;
static constexpr int kSwizzle = 2; // Swizzle<2,3,3>

// ============================================================
// Type aliases
// ============================================================
using elem_type = mctlass::bfloat16_t;
using Element = elem_type;
using ElementAccum = float;
using index_t = int64_t;

// ============================================================
// Kernel Traits (CuTe/mctlass, following FA2 exactly)
// ============================================================

// MMA Atom: native MACA 16x16x16 BF16
using MMA_Atom_Arch = MMA_Atom<MACA_16x16x16_F32BF16BF16F32>;

// TiledMMA: 4 warps along M direction
using TiledMma =
    TiledMMA<MMA_Atom_Arch, Layout<Shape<Int<kNWarps>, _1, _1>>, Layout<Shape<_1, _1, _1>>>;

// Smem layout with Swizzle<2,3,3> for 16x32 atom
using SmemLayoutAtom =
    decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                         Layout<Shape<_16, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
using SmemLayoutKV =
    decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));

// V transposed layout
using SmemLayoutAtomVtNoSwizzle =
    Layout<Shape<Int<kBlockKSmem>, Int<kBlockN>>, Stride<_1, Int<kBlockKSmem>>>;
using SmemLayoutAtomVt =
    decltype(composition(Swizzle<kSwizzle, 3, 3>{}, SmemLayoutAtomVtNoSwizzle{}));
using SmemLayoutVt =
    decltype(tile_to_shape(SmemLayoutAtomVt{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));
using SmemLayoutVtNoSwizzle =
    decltype(tile_to_shape(SmemLayoutAtomVtNoSwizzle{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));

// Output smem layout
using SmemLayoutO = decltype(tile_to_shape(
    decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                         Layout<Shape<_16, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{})){},
    Shape<Int<kBlockM>, Int<kHeadDim>>{}));

// Gmem copy: 128-bit per thread, kGmemElemsPerLoad = 8 bf16
static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element); // 8
static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;          // 32/8 = 4

using GmemLayoutAtom = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                              Stride<Int<kGmemThreadsPerRow>, _1>>;

// Synchronous global-to-shared copy used by the production baseline.
using GmemCopyAtomQKV = Copy_Atom<DefaultCopy, Element>;

using GmemTiledCopyQKV =
    decltype(make_tiled_copy(GmemCopyAtomQKV{}, GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));

using GmemTiledCopyO = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, Element>{}, GmemLayoutAtom{},
                                                Layout<Shape<_1, _8>>{}));

// Smem copy atoms
using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
using SmemCopyAtomO = Copy_Atom<UniversalCopy<uint64_t>, Element>;

// ============================================================
// Utility macros
// ============================================================
#ifndef CEIL_DIV
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#endif
