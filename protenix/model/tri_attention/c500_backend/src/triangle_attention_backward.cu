/* C500 BF16 反向：输入适配、设备计算、归约、工作区调度和兼容 C ABI。
 * Based on the colleague CuTe FA2
 * forward in triattention_c500_source.tar.gz. Q is already scaled.
 * dQ/dK/dV use native BF16 MMA with FP32 accumulators. Bias gradients retain
 * FP32 dS and use fixed-order reductions, without floating-point atomics.
 */
#include "triangle_attention_config.h"
#include "pair_bias_cache.h"
#include <cstdint>
#include <algorithm>

namespace triangle_attention_backward {
constexpr int BwdTile = 64, Dim = 32, Threads = 256;
constexpr float Log2E = 1.4426950408889634f;
static_assert(kBlockM == BwdTile && kBlockN == BwdTile && kHeadDim == Dim && kNThreads == Threads);

__device__ __forceinline__ void wait_shared_memory_writes() {
    __builtin_mxc_arrive(64);
    __syncthreads();
}

// A private backward workspace keeps both orientations aligned to float4.
// Like the colleague forward cache, it is refreshed on every invocation and
// requires one device and sequential streams per loaded library.
static float *pair_bias_workspace = nullptr;
static size_t pair_bias_capacity = 0;

#ifndef BWD_DBIAS_GROUP
#define BWD_DBIAS_GROUP 3
#endif
#ifndef BWD_FUSE_DBIAS2
#define BWD_FUSE_DBIAS2 1
#endif
#ifndef BWD_DS_WORKSPACE_BYTES
#define BWD_DS_WORKSPACE_BYTES (256ULL * 1024 * 1024)
#endif
// Bounded scratch for the FP32 dS already computed by dQ. Chunks are reduced
// in stream order, retaining deterministic bias gradients without atomics.
static float *score_gradient_workspace = nullptr;
static size_t score_gradient_capacity = 0;
static int score_gradient_reuse_override = -1;

// The colleague MMA conversion intentionally uses FAST_HALF_CVT (half-up).
// Raw adapters must instead match Triton's round-to-nearest-even conversion.
__device__ __forceinline__ Element round_bfloat16_nearest_even(float value) {
    uint32_t bits = __float_as_uint(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return Element::bitcast(uint16_t(bits >> 16));
}

// Raw-input adapters preserve the original BF16 rounding chain. Disjoint
// vector ranges avoid executing bias masks over the much larger Q tensor.
// 输入适配：缩放 raw Q 并按 nearest-even 舍入，两个 FP32 偏置转换为 BF16。
__global__ void prepare_raw_inputs_kernel(const Element *q, const float *b1, const float *b2,
                                          Element *qs, Element *b1s, Element *b2s, int64_t qc,
                                          int64_t c1, int64_t c2) {
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t qv = (qc + 7) / 8, v1 = (c1 + 7) / 8, v2 = (c2 + 7) / 8;
    if (t < qv) {
        const int64_t i = t * 8;
        if (i + 7 < qc) {
            mctlass::AlignedArray<Element, 8, 16> value;
            *reinterpret_cast<uint4 *>(value.data()) = *reinterpret_cast<const uint4 *>(q + i);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                value[j] = round_bfloat16_nearest_even(float(value[j]) * 0.17677669529663689f);
            *reinterpret_cast<uint4 *>(qs + i) = *reinterpret_cast<const uint4 *>(value.data());
        } else
            for (int j = 0; j < 8 && i + j < qc; ++j)
                qs[i + j] = round_bfloat16_nearest_even(float(q[i + j]) * 0.17677669529663689f);
    } else if (t < qv + v1 + v2) {
        const bool first = t < qv + v1;
        const int64_t i = (t - qv - (first ? 0 : v1)) * 8, count = first ? c1 : c2;
        const float *input = first ? b1 : b2;
        Element *output = first ? b1s : b2s;
        if (i + 7 < count) {
            mctlass::AlignedArray<float, 8, 16> value;
            *reinterpret_cast<float4 *>(value.data()) =
                *reinterpret_cast<const float4 *>(input + i);
            *reinterpret_cast<float4 *>(value.data() + 4) =
                *reinterpret_cast<const float4 *>(input + i + 4);
            mctlass::AlignedArray<Element, 8, 16> converted;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                converted[j] = round_bfloat16_nearest_even(value[j]);
            *reinterpret_cast<uint4 *>(output + i) =
                *reinterpret_cast<const uint4 *>(converted.data());
        } else
            for (int j = 0; j < 8 && i + j < count; ++j)
                output[i + j] = round_bfloat16_nearest_even(input[i + j]);
    }
}

// 输出适配：保留 raw dQ 的两次 BF16 舍入链；dBias2 经 BF16 舍入后存回 FP32。
__global__ void restore_raw_gradients_kernel(const Element *input, const float *bias_input,
                                             Element *output, float *bias_output, int64_t qc,
                                             int64_t bc) {
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x, qv = (qc + 7) / 8,
                  bv = (bc + 7) / 8;
    if (t < qv) {
        const int64_t i = t * 8;
        if (i + 7 < qc) {
            mctlass::AlignedArray<Element, 8, 16> value;
            *reinterpret_cast<uint4 *>(value.data()) = *reinterpret_cast<const uint4 *>(input + i);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                value[j] = round_bfloat16_nearest_even(float(value[j]) * 0.17677669529663689f);
            *reinterpret_cast<uint4 *>(output + i) = *reinterpret_cast<const uint4 *>(value.data());
        } else
            for (int j = 0; j < 8 && i + j < qc; ++j)
                output[i + j] =
                    round_bfloat16_nearest_even(float(input[i + j]) * 0.17677669529663689f);
    } else if (t < qv + bv) {
        const int64_t i = (t - qv) * 8;
        if (i + 7 < bc) {
            mctlass::AlignedArray<float, 8, 16> value;
            *reinterpret_cast<float4 *>(value.data()) =
                *reinterpret_cast<const float4 *>(bias_input + i);
            *reinterpret_cast<float4 *>(value.data() + 4) =
                *reinterpret_cast<const float4 *>(bias_input + i + 4);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                value[j] = float(round_bfloat16_nearest_even(value[j]));
            *reinterpret_cast<float4 *>(bias_output + i) =
                *reinterpret_cast<const float4 *>(value.data());
            *reinterpret_cast<float4 *>(bias_output + i + 4) =
                *reinterpret_cast<const float4 *>(value.data() + 4);
        } else
            for (int j = 0; j < 8 && i + j < bc; ++j)
                bias_output[i + j] = float(round_bfloat16_nearest_even(bias_input[i + j]));
    }
}

static int cached_multiprocessor_count = 0, cached_threads_per_multiprocessor = 0,
           cached_wave_size = 0;
// 归约：按 batch 和连续 N 范围，以固定顺序累加保存的 dS/分组局部和。
__global__ void reduce_score_gradients_by_batch_kernel(const float *ds, float *output,
                                                       int first_batch, int first_n, int samples,
                                                       int n, int s, int h) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x, count = int64_t(h) * s * s;
    if (i < count) {
        const int relative_batch = blockIdx.y, batch = first_batch + relative_batch;
        const int begin = relative_batch * n > first_n ? relative_batch * n - first_n : 0;
        const int end = (relative_batch + 1) * n - first_n < samples
                            ? (relative_batch + 1) * n - first_n
                            : samples;
        const int pitch = (s + 3) & ~3;
        const int64_t offset = (i / s) * pitch + i % s, stride = int64_t(h) * s * pitch;
        float value = 0.f;
        for (int sample = begin; sample < end; ++sample)
            value += ds[int64_t(sample) * stride + offset];
        float *destination = output + int64_t(batch) * count + i;
        *destination = (relative_batch == 0 && first_n > 0) ? *destination + value : value;
    }
}

// 偏置布局：一次转换得到行对齐的 Bias2 正向与转置布局，供 dQ 和 dK/dV 连续读取。
__global__ void prepare_pair_bias_layouts_kernel(const Element *input, float *normal,
                                                 float *transposed, int s, int pitch) {
    __shared__ float tile[32][33];
    const int tiles = (pitch + 31) / 32;
    const int matrix = blockIdx.y / tiles, tile_y = blockIdx.y % tiles;
    int x = blockIdx.x * 32 + threadIdx.x, y = tile_y * 32 + threadIdx.y;
    const int64_t base = int64_t(matrix) * s * pitch, src = int64_t(matrix) * s * s;
#pragma unroll
    for (int j = 0; j < 32; j += 8) {
        float value = (x < s && y + j < s) ? float(input[src + int64_t(y + j) * s + x]) : 0.f;
        tile[threadIdx.y + j][threadIdx.x] = value;
        if (x < pitch && y + j < s)
            normal[base + int64_t(y + j) * pitch + x] = value;
    }
    __syncthreads();
    x = tile_y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
#pragma unroll
    for (int j = 0; j < 32; j += 8)
        if (x < pitch && y + j < s)
            transposed[base + int64_t(y + j) * pitch + x] = tile[threadIdx.x][threadIdx.y + j];
}

// Backward tiles are independent of the forward's fixed 64x64 layout.
template <int M, int N, int W = M / 16> struct BackwardTileLayout {
    static constexpr int Warps = W, Rows = M / (16 * W), Threads = Warps * 64,
                         StorageElements = (M > N ? M : N) * Dim;
    static constexpr int Shared = 2 * StorageElements * sizeof(Element) + 2 * N * sizeof(float);
    using Mma =
        TiledMMA<MMA_Atom_Arch, Layout<Shape<Int<Warps>, _1, _1>>, Layout<Shape<_1, _1, _1>>>;
    using Own = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<M>, Int<Dim>>{}));
    using Iter = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<N>, Int<Dim>>{}));
    using TransAtom = Layout<Shape<Int<Dim>, Int<N>>, Stride<_1, Int<Dim>>>;
    using Trans = decltype(tile_to_shape(composition(Swizzle<2, 3, 3>{}, TransAtom{}),
                                         Shape<Int<Dim>, Int<N>>{}));
    using TransPlain = TransAtom;
    using CopyLayout = Layout<Shape<Int<Threads / 4>, _4>, Stride<_4, _1>>;
    using Copy =
        decltype(make_tiled_copy(GmemCopyAtomQKV{}, CopyLayout{}, Layout<Shape<_1, _8>>{}));
};

template <int Rows, class Copy, bool Full, class SharedTensor>
__device__ __forceinline__ void load_rows_to_shared(const Element *base, int first, int s, int h,
                                                    SharedTensor dst) {
    Copy cp;
    auto th = cp.get_thread_slice(threadIdx.x);
    Tensor global = make_tensor(make_gmem_ptr(base + int64_t(first) * h * Dim),
                                Shape<Int<Rows>, Int<Dim>>{}, make_stride(h * Dim, _1{}));
    Tensor gs = th.partition_S(global);
    Tensor ss = th.partition_D(dst);
    Tensor identity = make_identity_tensor(make_shape(Int<Rows>{}, Int<Dim>{}));
    Tensor coord = th.partition_S(identity);
#pragma unroll
    for (int m = 0; m < size<1>(gs); ++m) {
        if (Full || first + get<0>(coord(0, m, 0)) < s) {
#pragma unroll
            for (int k = 0; k < size<2>(gs); ++k)
                cute::copy(cp, gs(_, m, k), ss(_, m, k));
        } else
            cute::clear(ss(_, m, _));
    }
}

#ifndef BWD_PIPELINE_DQ
#define BWD_PIPELINE_DQ 1
#endif
#ifndef BWD_PIPELINE_DKDV
#define BWD_PIPELINE_DKDV 1
#endif

template <int Rows, int BlockThreads> struct PrefetchedRows {
    static constexpr int Vectors = (Rows * Dim + BlockThreads * 8 - 1) / (BlockThreads * 8);
    mctlass::AlignedArray<Element, 8, 16> data[Vectors];
};

template <int Rows, int BlockThreads, bool Full>
__device__ __forceinline__ PrefetchedRows<Rows, BlockThreads>
prefetch_rows_to_registers(const Element *base, int first, int s, int h) {
    PrefetchedRows<Rows, BlockThreads> result;
    const int col = (threadIdx.x % 4) * 8;
#pragma unroll
    for (int i = 0; i < result.Vectors; ++i) {
        int row = threadIdx.x / 4 + i * (BlockThreads / 4);
        uint4 value = make_uint4(0, 0, 0, 0);
        if (row < Rows && (Full || first + row < s))
            value = *reinterpret_cast<const uint4 *>(base + int64_t(first + row) * h * Dim + col);
        *reinterpret_cast<uint4 *>(result.data[i].data()) = value;
    }
    return result;
}

template <int Rows, int BlockThreads, class SharedTensor>
__device__ __forceinline__ void
stage_prefetched_rows(const PrefetchedRows<Rows, BlockThreads> &input, SharedTensor dst) {
    const int col = (threadIdx.x % 4) * 8;
#pragma unroll
    for (int i = 0; i < input.Vectors; ++i) {
        const int row = threadIdx.x / 4 + i * (BlockThreads / 4);
        if (row < Rows)
            *reinterpret_cast<uint4 *>(&dst(row, col)) =
                *reinterpret_cast<const uint4 *>(input.data[i].data());
    }
}

template <class Acc, class A, class B, class S, class Copy, class Thr, class Mma>
__device__ __forceinline__ void mma_from_shared_memory(Acc &acc, A &a, B &b, S const &sb, Mma mma,
                                                       Copy cp, Thr th) {
    Tensor view = th.retile_D(b);
    cute::copy(cp, sb(_, _, _0{}), view(_, _, _0{}));
#pragma unroll
    for (int k = 0; k < size<2>(a); ++k) {
        if (k < size<2>(a) - 1)
            cute::copy(cp, sb(_, _, k + 1), view(_, _, k + 1));
        cute::gemm(mma, a(_, _, k), b(_, _, k), acc);
    }
}

template <int Warps, bool RawQ = false, class Acc>
__device__ __forceinline__ void store_bfloat16_gradient(Acc &acc, Element *output, int first, int s,
                                                        int h) {
    constexpr int Rows = decltype(size(acc))::value / 8;
    const int lane = threadIdx.x % 64, first_row = threadIdx.x / 64 * 16 + lane % 16;
    float *data = reinterpret_cast<float *>(acc.data());
#pragma unroll
    for (int mi = 0; mi < Rows; ++mi) {
        int row = first_row + mi * Warps * 16;
        if (first + row < s) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                mctlass::AlignedArray<Element, 4, 8> values;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    Element rounded = Element(data[(nt * Rows + mi) * 4 + e]);
                    if constexpr (RawQ)
                        rounded =
                            round_bfloat16_nearest_even(float(rounded) * 0.17677669529663689f);
                    values[e] = rounded;
                }
                const int col = nt * 16 + (lane / 16) * 4;
                *reinterpret_cast<uint2 *>(output + int64_t(first + row) * h * Dim + col) =
                    *reinterpret_cast<const uint2 *>(values.data());
            }
        }
    }
}

// 保存状态：Delta = sum_d(O * dO)，按 [B,N,S,H] 存储。
__global__ void compute_output_gradient_dot_kernel(const Element *o, const Element *dout,
                                                   float *delta, int64_t count) {
    const int group = threadIdx.x % 4;
    const int64_t row = int64_t(blockIdx.x) * 64 + threadIdx.x / 4;
    float value = 0.f;
    if (row < count) {
        mctlass::AlignedArray<Element, 8, 16> ov, dv;
        const int64_t base = row * Dim + group * 8;
        *reinterpret_cast<uint4 *>(ov.data()) = *reinterpret_cast<const uint4 *>(o + base);
        *reinterpret_cast<uint4 *>(dv.data()) = *reinterpret_cast<const uint4 *>(dout + base);
#pragma unroll
        for (int d = 0; d < 8; ++d)
            value += float(ov[d]) * float(dv[d]);
    }
    // All lanes participate, including the partial final wave.
    value += __shfl_xor_sync(uint64_t(-1), value, 1, 4);
    value += __shfl_xor_sync(uint64_t(-1), value, 2, 4);
    if (group == 0 && row < count)
        delta[row] = value;
}

template <int M, int N, int W, bool Full, bool Transposed>
__device__ __forceinline__ void initialize_attention_scores(float *score, const float *b1,
                                                            const float *b2, const float *own_bias,
                                                            int first, int step, int s) {
    constexpr int Rows = M / (16 * W);
    const int pitch = (s + 3) & ~3;
    const int lane = threadIdx.x % 64, base_row = threadIdx.x / 64 * 16 + lane % 16,
              columns = (lane / 16) * 4;
#pragma unroll
    for (int nt = 0; nt < N / 16; ++nt) {
        int col = step + nt * 16 + columns;
#pragma unroll
        for (int mi = 0; mi < Rows; ++mi) {
            int row = first + base_row + mi * W * 16, offset = (nt * Rows + mi) * 4;
            bool valid = Full || row < s;
            if (Full || (valid && col + 3 < s)) {
                float4 v2 = *reinterpret_cast<const float4 *>(b2 + int64_t(row) * pitch + col);
                if constexpr (Transposed) {
                    float x = own_bias[mi];
                    score[offset] = x + v2.x;
                    score[offset + 1] = x + v2.y;
                    score[offset + 2] = x + v2.z;
                    score[offset + 3] = x + v2.w;
                } else {
                    float4 v1 = *reinterpret_cast<const float4 *>(b1 + col);
                    score[offset] = v1.x + v2.x;
                    score[offset + 1] = v1.y + v2.y;
                    score[offset + 2] = v1.z + v2.z;
                    score[offset + 3] = v1.w + v2.w;
                }
            } else {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float x = 0.f;
                    if (valid && col + e < s) {
                        if constexpr (Transposed)
                            x = own_bias[mi];
                        else
                            x = b1[col + e];
                        x += b2[int64_t(row) * pitch + col + e];
                    }
                    score[offset + e] = x;
                }
            }
        }
    }
}

// Query-owned CTA: dQ = dS K. Q/dO are cached across all key tiles.
// dQ：重建 P 和 dS；可保存 FP32 dS 供偏置归约复用，RawQ 控制写回舍入链。
template <int M, int N, bool Full = false, int W = M / 16, bool SaveDS = false, bool RawQ = false>
__global__ __launch_bounds__(W * 64) void compute_query_gradient_kernel(
    const Element *q, const Element *k, const Element *v, const float *bias1, const float *bias2,
    const float *lse, const float *delta, const Element *dout, Element *dq, int n, int s, int h,
    int total_bn, int group_log2, int start_bn, float *ds_output) {
    using L = BackwardTileLayout<M, N, W>;
    extern __shared__ char raw[];
    Element *sm = reinterpret_cast<Element *>(raw);
    Tensor s0 = make_tensor(make_smem_ptr(sm), typename L::Iter{});
    Tensor s1 = make_tensor(make_smem_ptr(sm + L::StorageElements), typename L::Iter{});
    Tensor s0t = make_tensor(s0.data(), typename L::Trans{});
    Tensor s0tn = make_tensor(s0.data(), typename L::TransPlain{});
    Tensor own0 = make_tensor(make_smem_ptr(sm), typename L::Own{});
    Tensor own1 = make_tensor(make_smem_ptr(sm + L::StorageElements), typename L::Own{});
    const int local_bn = (blockIdx.z << group_log2) + (blockIdx.x & ((1 << group_log2) - 1));
    if (local_bn >= total_bn)
        return;
    const int bn = start_bn + local_bn;
    const int head = blockIdx.y, b = bn / n, first = (blockIdx.x >> group_log2) * M;
    const int lane = threadIdx.x % 64, row = threadIdx.x / 64 * 16 + lane % 16,
              columns = (lane / 16) * 4;
    const int64_t base = (int64_t(bn) * s * h + head) * Dim;
    q += base;
    k += base;
    v += base;
    dout += base;
    dq += base;
    bias1 += int64_t(bn) * ((s + 3) & ~3);
    bias2 += (int64_t(b) * h + head) * s * ((s + 3) & ~3);
    lse += int64_t(bn) * s * h + head;
    delta += int64_t(bn) * s * h + head;
    typename L::Mma mma;
    auto mt = mma.get_thread_slice(threadIdx.x);
    auto ca = make_tiled_copy_A(SmemCopyAtom{}, mma);
    auto ta = ca.get_thread_slice(threadIdx.x);
    auto cb = make_tiled_copy_B(SmemCopyAtom{}, mma);
    auto tb = cb.get_thread_slice(threadIdx.x);
    auto ct = make_tiled_copy_B(SmemCopyAtomTransposed{}, mma);
    auto tt = ct.get_thread_slice(threadIdx.x);
    Tensor rq = mt.partition_fragment_A(own0), rd = mt.partition_fragment_A(own1);
    Tensor rk = mt.partition_fragment_B(s0), rv = mt.partition_fragment_B(s1);
    Tensor rkt = mt.partition_fragment_B(s0tn);
    Tensor ak = tb.partition_S(s0), av = tb.partition_S(s1);
    Tensor akt = tt.partition_S(s0t);
    load_rows_to_shared<M, typename L::Copy, Full>(q, first, s, h, own0);
    load_rows_to_shared<M, typename L::Copy, Full>(dout, first, s, h, own1);
    wait_shared_memory_writes();
    Tensor qview = ta.retile_D(rq);
    Tensor dview = ta.retile_D(rd);
    cute::copy(ca, ta.partition_S(own0), qview);
    cute::copy(ca, ta.partition_S(own1), dview);
    __syncthreads();
    Tensor result = partition_fragment_C(mma, Shape<Int<M>, Int<Dim>>{});
    clear(result);
    float lm[L::Rows], dl[L::Rows];
#pragma unroll
    for (int mi = 0; mi < L::Rows; ++mi) {
        int i = first + row + mi * W * 16;
        lm[mi] = (Full || i < s) ? lse[int64_t(i) * h] : 0.f;
        dl[mi] = (Full || i < s) ? delta[int64_t(i) * h] : 0.f;
    }
    // Only the measured long/odd-sequence tiles benefit from prefetch.
    // Short-sequence and saved-dS tiles keep direct shared-memory copies.
    constexpr bool PipelineDQ =
        BWD_PIPELINE_DQ && !SaveDS && W == 2 && ((M == 64 && N == 32) || (M == 32 && N == 64));
    PrefetchedRows<N, L::Threads> pending_k, pending_v;
    if constexpr (PipelineDQ) {
        pending_k = prefetch_rows_to_registers<N, L::Threads, Full>(k, 0, s, h);
        pending_v = prefetch_rows_to_registers<N, L::Threads, Full>(v, 0, s, h);
    }
    for (int key = 0; key < s; key += N) {
        if constexpr (PipelineDQ) {
            stage_prefetched_rows(pending_k, s0);
            stage_prefetched_rows(pending_v, s1);
            if (key + N < s) {
                pending_k = prefetch_rows_to_registers<N, L::Threads, Full>(k, key + N, s, h);
                pending_v = prefetch_rows_to_registers<N, L::Threads, Full>(v, key + N, s, h);
            }
        } else {
            load_rows_to_shared<N, typename L::Copy, Full>(k, key, s, h, s0);
            load_rows_to_shared<N, typename L::Copy, Full>(v, key, s, h, s1);
        }
        Tensor score = partition_fragment_C(mma, Shape<Int<M>, Int<N>>{});
        Tensor dp = partition_fragment_C(mma, Shape<Int<M>, Int<N>>{});
        clear(dp);
        float *sp = reinterpret_cast<float *>(score.data());
        float *pp = reinterpret_cast<float *>(dp.data());
        initialize_attention_scores<M, N, W, Full, false>(sp, bias1, bias2, nullptr, first, key, s);
        wait_shared_memory_writes();
        mma_from_shared_memory(score, rq, rk, ak, mma, cb, tb);
        mma_from_shared_memory(dp, rd, rv, av, mma, cb, tb);
#pragma unroll
        for (int nt = 0; nt < N / 16; ++nt)
#pragma unroll
            for (int mi = 0; mi < L::Rows; ++mi)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    int i = (nt * L::Rows + mi) * 4 + e, j = key + nt * 16 + columns + e;
                    bool valid = Full || (first + row + mi * W * 16 < s && j < s);
                    float p = valid ? __builtin_exp2f(sp[i] * Log2E - lm[mi]) : 0.f;
                    sp[i] = p * (pp[i] - dl[mi]);
                }
        if constexpr (SaveDS) {
            const int pitch = (s + 3) & ~3;
#pragma unroll
            for (int nt = 0; nt < N / 16; ++nt)
#pragma unroll
                for (int mi = 0; mi < L::Rows; ++mi) {
                    int r = first + row + mi * W * 16, c = key + nt * 16 + columns;
                    if (Full || (r < s && c < s)) {
                        float *dst =
                            ds_output + ((int64_t(local_bn) * h + head) * s + r) * pitch + c;
                        int offset = (nt * L::Rows + mi) * 4;
                        if (Full || c + 3 < s)
                            *reinterpret_cast<float4 *>(dst) = make_float4(
                                sp[offset], sp[offset + 1], sp[offset + 2], sp[offset + 3]);
                        else {
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                if (c + e < s)
                                    dst[e] = sp[offset + e];
                        }
                    }
                }
        }
        constexpr int Count = decltype(size(score))::value;
        mctlass::NumericArrayConverter<Element, float, Count> convert;
        auto ds = convert(*reinterpret_cast<const mctlass::Array<float, Count> *>(score.data()));
        Tensor rds = make_tensor(make_rmem_ptr<Element>(&ds), score.layout());
        mma_from_shared_memory(result, rds, rkt, akt, mma, ct, tt);
        __syncthreads();
    }
    store_bfloat16_gradient<W, RawQ>(result, dq, first, s, h);
}

// Each CTA owns a query tile and a consecutive N group. dS is consumed by
// dQ immediately, and only the FP32 sum for that N group reaches scratch.
// No atomics or BF16 dBias accumulation are used.
// 分组 dQ：连续三路 N 复用 Bias2，在计算 dQ 时累加 dBias2 局部和。
template <int M, int N, bool Full, int W, int G, bool RawQ>
__global__ __launch_bounds__(W * 64) void compute_grouped_query_bias_gradients_kernel(
    const Element *q, const Element *k, const Element *v, const float *bias1, const float *bias2,
    const float *lse, const float *delta, const Element *dout, Element *dq, int n, int s, int h,
    int group_begin, int group_count, float *partial) {
    using L = BackwardTileLayout<M, N, W>;
    extern __shared__ char raw[];
    Element *sm = reinterpret_cast<Element *>(raw);
    Tensor s0 = make_tensor(make_smem_ptr(sm), typename L::Iter{});
    Tensor s1 = make_tensor(make_smem_ptr(sm + L::StorageElements), typename L::Iter{});
    Tensor s0t = make_tensor(s0.data(), typename L::Trans{});
    Tensor s0tn = make_tensor(s0.data(), typename L::TransPlain{});
    Tensor own0 = make_tensor(make_smem_ptr(sm), typename L::Own{});
    Tensor own1 = make_tensor(make_smem_ptr(sm + L::StorageElements), typename L::Own{});
    const int local_group = blockIdx.z;
    if (local_group >= group_count)
        return;
    const int groups = (n + G - 1) / G, global_group = group_begin + local_group;
    const int batch = global_group / groups, first_n = (global_group % groups) * G;
    const int head = blockIdx.y, first = blockIdx.x * M, pitch = (s + 3) & ~3;
    const int lane = threadIdx.x % 64, row = threadIdx.x / 64 * 16 + lane % 16,
              columns = (lane / 16) * 4;
    typename L::Mma mma;
    auto mt = mma.get_thread_slice(threadIdx.x);
    auto ca = make_tiled_copy_A(SmemCopyAtom{}, mma);
    auto ta = ca.get_thread_slice(threadIdx.x);
    auto cb = make_tiled_copy_B(SmemCopyAtom{}, mma);
    auto tb = cb.get_thread_slice(threadIdx.x);
    auto ct = make_tiled_copy_B(SmemCopyAtomTransposed{}, mma);
    auto tt = ct.get_thread_slice(threadIdx.x);
    Tensor aq = mt.partition_fragment_A(own0), ad = mt.partition_fragment_A(own1);
    Tensor out = partition_fragment_C(mma, Shape<Int<M>, Int<Dim>>{});
    constexpr int AC = decltype(size(aq))::value, RC = decltype(size(out))::value,
                  SC = M * N / (W * 64);
    mctlass::Array<Element, AC> cached_q[G], cached_d[G];
    mctlass::Array<float, RC> result[G];
    float lm[G][L::Rows], dl[G][L::Rows];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        Tensor rq = make_tensor(cached_q[g].data(), aq.layout());
        Tensor rd = make_tensor(cached_d[g].data(), ad.layout());
        Tensor accum = make_tensor(result[g].data(), out.layout());
        clear(accum);
        if (first_n + g < n) {
            const int bn = batch * n + first_n + g;
            const int64_t base = (int64_t(bn) * s * h + head) * Dim;
            load_rows_to_shared<M, typename L::Copy, Full>(q + base, first, s, h, own0);
            load_rows_to_shared<M, typename L::Copy, Full>(dout + base, first, s, h, own1);
            wait_shared_memory_writes();
            Tensor qview = ta.retile_D(rq), dview = ta.retile_D(rd);
            cute::copy(ca, ta.partition_S(own0), qview);
            cute::copy(ca, ta.partition_S(own1), dview);
            __syncthreads();
#pragma unroll
            for (int mi = 0; mi < L::Rows; ++mi) {
                const int i = first + row + mi * W * 16;
                lm[g][mi] = (Full || i < s) ? lse[(int64_t(bn) * s + i) * h + head] : 0.f;
                dl[g][mi] = (Full || i < s) ? delta[(int64_t(bn) * s + i) * h + head] : 0.f;
            }
        }
    }
    Tensor rk = mt.partition_fragment_B(s0), rv = mt.partition_fragment_B(s1);
    Tensor rkt = mt.partition_fragment_B(s0tn);
    Tensor ak = tb.partition_S(s0), av = tb.partition_S(s1);
    Tensor akt = tt.partition_S(s0t);
    const float *b2 = bias2 + (int64_t(batch) * h + head) * s * pitch;
    for (int key = 0; key < s; key += N) {
        mctlass::Array<float, SC> summed;
#pragma unroll
        for (int i = 0; i < SC; ++i)
            summed[i] = 0.f;
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (first_n + g < n) {
                const int bn = batch * n + first_n + g;
                const int64_t base = (int64_t(bn) * s * h + head) * Dim;
                Tensor rq = make_tensor(cached_q[g].data(), aq.layout());
                Tensor rd = make_tensor(cached_d[g].data(), ad.layout());
                Tensor accum = make_tensor(result[g].data(), out.layout());
                load_rows_to_shared<N, typename L::Copy, Full>(k + base, key, s, h, s0);
                load_rows_to_shared<N, typename L::Copy, Full>(v + base, key, s, h, s1);
                Tensor score = partition_fragment_C(mma, Shape<Int<M>, Int<N>>{});
                Tensor dp = partition_fragment_C(mma, Shape<Int<M>, Int<N>>{});
                clear(dp);
                float *sp = reinterpret_cast<float *>(score.data());
                float *pp = reinterpret_cast<float *>(dp.data());
                initialize_attention_scores<M, N, W, Full, false>(sp, bias1 + int64_t(bn) * pitch,
                                                                  b2, nullptr, first, key, s);
                wait_shared_memory_writes();
                mma_from_shared_memory(score, rq, rk, ak, mma, cb, tb);
                mma_from_shared_memory(dp, rd, rv, av, mma, cb, tb);
#pragma unroll
                for (int nt = 0; nt < N / 16; ++nt)
#pragma unroll
                    for (int mi = 0; mi < L::Rows; ++mi)
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const int i = (nt * L::Rows + mi) * 4 + e,
                                      j = key + nt * 16 + columns + e;
                            const bool valid = Full || (first + row + mi * W * 16 < s && j < s);
                            float p = valid ? __builtin_exp2f(sp[i] * Log2E - lm[g][mi]) : 0.f;
                            sp[i] = p * (pp[i] - dl[g][mi]);
                            summed[i] += sp[i];
                        }
                mctlass::NumericArrayConverter<Element, float, SC> convert;
                auto ds =
                    convert(*reinterpret_cast<const mctlass::Array<float, SC> *>(score.data()));
                Tensor rds = make_tensor(make_rmem_ptr<Element>(&ds), score.layout());
                mma_from_shared_memory(accum, rds, rkt, akt, mma, ct, tt);
                __syncthreads();
            }
        }
#pragma unroll
        for (int nt = 0; nt < N / 16; ++nt)
#pragma unroll
            for (int mi = 0; mi < L::Rows; ++mi) {
                const int r = first + row + mi * W * 16, c = key + nt * 16 + columns,
                          off = (nt * L::Rows + mi) * 4;
                if (Full || (r < s && c < s)) {
                    float *dst = partial + ((int64_t(local_group) * h + head) * s + r) * pitch + c;
                    if (Full || c + 3 < s)
                        *reinterpret_cast<float4 *>(dst) = make_float4(
                            summed[off], summed[off + 1], summed[off + 2], summed[off + 3]);
                    else {
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            if (c + e < s)
                                dst[e] = summed[off + e];
                    }
                }
            }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        if (first_n + g < n) {
            const int bn = batch * n + first_n + g;
            const int64_t base = (int64_t(bn) * s * h + head) * Dim;
            Tensor accum = make_tensor(result[g].data(), out.layout());
            store_bfloat16_gradient<W, RawQ>(accum, dq + base, first, s, h);
        }
    }
}

// Key-owned CTA: dK = dS^T Q, dV = P^T dO. The score is transposed;
// the padded backward bias workspace supplies contiguous transposed rows.
// dK/dV：沿 key 方向计算两个梯度，并在请求五梯度时产生 dBias1 的逐 head 局部和。
template <int M, int N, bool Full = false, int W = M / 16>
__global__ __launch_bounds__(W * 64) void compute_key_value_gradients_kernel(
    const Element *q, const Element *k, const Element *v, const float *bias1, const float *bias2t,
    const float *lse, const float *delta, const Element *dout, Element *dk, Element *dv,
    float *db1_head, int n, int s, int h, int total_bn, int group_log2) {
    using L = BackwardTileLayout<M, N, W>;
    extern __shared__ char raw[];
    Element *sm = reinterpret_cast<Element *>(raw);
    Tensor s0 = make_tensor(make_smem_ptr(sm), typename L::Iter{});
    Tensor s1 = make_tensor(make_smem_ptr(sm + L::StorageElements), typename L::Iter{});
    Tensor s0t = make_tensor(s0.data(), typename L::Trans{}),
           s1t = make_tensor(s1.data(), typename L::Trans{});
    Tensor s0tn = make_tensor(s0.data(), typename L::TransPlain{}),
           s1tn = make_tensor(s1.data(), typename L::TransPlain{});
    float *lm = reinterpret_cast<float *>(sm + 2 * L::StorageElements);
    float *dl = lm + N;
    Tensor own0 = make_tensor(make_smem_ptr(sm), typename L::Own{});
    Tensor own1 = make_tensor(make_smem_ptr(sm + L::StorageElements), typename L::Own{});
    const int bn = (blockIdx.z << group_log2) + (blockIdx.x & ((1 << group_log2) - 1));
    if (bn >= total_bn)
        return;
    const int head = blockIdx.y, b = bn / n, first = (blockIdx.x >> group_log2) * M;
    const int lane = threadIdx.x % 64, row = threadIdx.x / 64 * 16 + lane % 16,
              columns = (lane / 16) * 4;
    const int64_t base = (int64_t(bn) * s * h + head) * Dim;
    q += base;
    k += base;
    v += base;
    dout += base;
    dk += base;
    dv += base;
    bias1 += int64_t(bn) * ((s + 3) & ~3);
    bias2t += (int64_t(b) * h + head) * s * ((s + 3) & ~3);
    lse += int64_t(bn) * s * h + head;
    delta += int64_t(bn) * s * h + head;
    typename L::Mma mma;
    auto mt = mma.get_thread_slice(threadIdx.x);
    auto ca = make_tiled_copy_A(SmemCopyAtom{}, mma);
    auto ta = ca.get_thread_slice(threadIdx.x);
    auto cb = make_tiled_copy_B(SmemCopyAtom{}, mma);
    auto tb = cb.get_thread_slice(threadIdx.x);
    auto ct = make_tiled_copy_B(SmemCopyAtomTransposed{}, mma);
    auto tt = ct.get_thread_slice(threadIdx.x);
    Tensor rk = mt.partition_fragment_A(own0), rv = mt.partition_fragment_A(own1);
    Tensor rq = mt.partition_fragment_B(s0), rd = mt.partition_fragment_B(s1);
    Tensor rqt = mt.partition_fragment_B(s0tn), rdt = mt.partition_fragment_B(s1tn);
    Tensor aq = tb.partition_S(s0), ad = tb.partition_S(s1);
    Tensor aqt = tt.partition_S(s0t), adt = tt.partition_S(s1t);
    load_rows_to_shared<M, typename L::Copy, Full>(k, first, s, h, own0);
    load_rows_to_shared<M, typename L::Copy, Full>(v, first, s, h, own1);
    wait_shared_memory_writes();
    Tensor kview = ta.retile_D(rk), vview = ta.retile_D(rv);
    cute::copy(ca, ta.partition_S(own0), kview);
    cute::copy(ca, ta.partition_S(own1), vview);
    __syncthreads();
    Tensor resultk = partition_fragment_C(mma, Shape<Int<M>, Int<Dim>>{});
    Tensor resultv = partition_fragment_C(mma, Shape<Int<M>, Int<Dim>>{});
    clear(resultk);
    clear(resultv);
    float db1_local[L::Rows], b1[L::Rows];
#pragma unroll
    for (int mi = 0; mi < L::Rows; ++mi) {
        int i = first + row + mi * W * 16;
        db1_local[mi] = 0.f;
        b1[mi] = (Full || i < s) ? bias1[i] : 0.f;
    }
    constexpr bool PipelineDKDV =
        BWD_PIPELINE_DKDV && M == 32 && ((N == 32 && W == 1) || (N == 64 && W == 2));
    PrefetchedRows<N, L::Threads> pending_q, pending_do;
    if constexpr (PipelineDKDV) {
        pending_q = prefetch_rows_to_registers<N, L::Threads, Full>(q, 0, s, h);
        pending_do = prefetch_rows_to_registers<N, L::Threads, Full>(dout, 0, s, h);
    }
    for (int query = 0; query < s; query += N) {
        if constexpr (PipelineDKDV) {
            stage_prefetched_rows(pending_q, s0);
            stage_prefetched_rows(pending_do, s1);
            if (query + N < s) {
                pending_q = prefetch_rows_to_registers<N, L::Threads, Full>(q, query + N, s, h);
                pending_do = prefetch_rows_to_registers<N, L::Threads, Full>(dout, query + N, s, h);
            }
        } else {
            load_rows_to_shared<N, typename L::Copy, Full>(q, query, s, h, s0);
            load_rows_to_shared<N, typename L::Copy, Full>(dout, query, s, h, s1);
        }
        if (threadIdx.x < N) {
            int i = query + threadIdx.x;
            lm[threadIdx.x] = (Full || i < s) ? lse[int64_t(i) * h] : 0.f;
            dl[threadIdx.x] = (Full || i < s) ? delta[int64_t(i) * h] : 0.f;
        }
        Tensor score = partition_fragment_C(mma, Shape<Int<M>, Int<N>>{});
        Tensor dp = partition_fragment_C(mma, Shape<Int<M>, Int<N>>{});
        clear(dp);
        float *sp = reinterpret_cast<float *>(score.data());
        float *pp = reinterpret_cast<float *>(dp.data());
        initialize_attention_scores<M, N, W, Full, true>(sp, bias1, bias2t, b1, first, query, s);
        wait_shared_memory_writes();
        mma_from_shared_memory(score, rk, rq, aq, mma, cb, tb);
        mma_from_shared_memory(dp, rv, rd, ad, mma, cb, tb);
#pragma unroll
        for (int nt = 0; nt < N / 16; ++nt)
#pragma unroll
            for (int mi = 0; mi < L::Rows; ++mi)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    int c = nt * 16 + columns + e, i = (nt * L::Rows + mi) * 4 + e;
                    bool valid = Full || (first + row + mi * W * 16 < s && query + c < s);
                    float p = valid ? __builtin_exp2f(sp[i] * Log2E - lm[c]) : 0.f;
                    float ds = p * (pp[i] - dl[c]);
                    sp[i] = p;
                    pp[i] = ds;
                    db1_local[mi] += ds;
                }
        constexpr int Count = decltype(size(score))::value;
        mctlass::NumericArrayConverter<Element, float, Count> convert;
        auto probabilities =
            convert(*reinterpret_cast<const mctlass::Array<float, Count> *>(score.data()));
        Tensor rp = make_tensor(make_rmem_ptr<Element>(&probabilities), score.layout());
        mma_from_shared_memory(resultv, rp, rdt, adt, mma, ct, tt);
        // P is dead before dS is converted; their BF16 temporaries can reuse registers.
        auto derivatives =
            convert(*reinterpret_cast<const mctlass::Array<float, Count> *>(dp.data()));
        Tensor rds = make_tensor(make_rmem_ptr<Element>(&derivatives), dp.layout());
        mma_from_shared_memory(resultk, rds, rqt, aqt, mma, ct, tt);
        __syncthreads();
    }
    store_bfloat16_gradient<W>(resultk, dk, first, s, h);
    store_bfloat16_gradient<W>(resultv, dv, first, s, h);
    if (db1_head) {
#pragma unroll
        for (int mi = 0; mi < L::Rows; ++mi) {
            float x = db1_local[mi];
            x += __shfl_xor_sync(uint64_t(-1), x, 16, 64);
            x += __shfl_xor_sync(uint64_t(-1), x, 32, 64);
            int i = first + row + mi * W * 16;
            if (lane < 16 && i < s)
                db1_head[(int64_t(bn) * h + head) * s + i] = x;
        }
    }
}

// Split N into independent groups. Every CTA owns one (query,key) tile and
// serially accumulates its N subset; the final group reduction is deterministic.
// 通用 dBias2 回退：未复用 dS 时重算分组偏置梯度；覆盖工作区或驻留条件不满足的输入。
template <int M, int N, bool Full = false, int W = M / 16>
__global__ __launch_bounds__(W * 64) void compute_pair_bias_gradient_kernel(
    const Element *q, const Element *k, const Element *v, const float *bias1, const float *bias2,
    const float *lse, const float *delta, const Element *dout, float *partial, int n, int s, int h,
    int groups) {
    using L = BackwardTileLayout<M, N, W>;
    extern __shared__ char raw[];
    Element *sm = reinterpret_cast<Element *>(raw);
    Tensor s0 = make_tensor(make_smem_ptr(sm), typename L::Iter{});
    Tensor s1 = make_tensor(make_smem_ptr(sm + L::StorageElements), typename L::Iter{});
    Tensor own0 = make_tensor(make_smem_ptr(sm), typename L::Own{});
    Tensor own1 = make_tensor(make_smem_ptr(sm + L::StorageElements), typename L::Own{});
    const int tiles = (s + N - 1) / N, query = (blockIdx.x / tiles) * M,
              key = (blockIdx.x % tiles) * N;
    const int head = blockIdx.y, b = blockIdx.z / groups, group = blockIdx.z % groups;
    const int lane = threadIdx.x % 64, row = threadIdx.x / 64 * 16 + lane % 16,
              columns = (lane / 16) * 4;
    typename L::Mma mma;
    auto mt = mma.get_thread_slice(threadIdx.x);
    auto ca = make_tiled_copy_A(SmemCopyAtom{}, mma);
    auto ta = ca.get_thread_slice(threadIdx.x);
    auto cb = make_tiled_copy_B(SmemCopyAtom{}, mma);
    auto tb = cb.get_thread_slice(threadIdx.x);
    Tensor rq = mt.partition_fragment_A(own0), rd = mt.partition_fragment_A(own1);
    Tensor rk = mt.partition_fragment_B(s0), rv = mt.partition_fragment_B(s1);
    Tensor ak = tb.partition_S(s0), av = tb.partition_S(s1);
    constexpr int Count = M * N / (W * 64);
    const int pitch = (s + 3) & ~3;
    float accumulated[Count], b2[Count];
#pragma unroll
    for (int nt = 0; nt < N / 16; ++nt)
#pragma unroll
        for (int mi = 0; mi < L::Rows; ++mi) {
            int i = (nt * L::Rows + mi) * 4, j = key + nt * 16 + columns,
                r = query + row + mi * W * 16;
            bool valid = Full || r < s;
            if (Full || (valid && j + 3 < s)) {
                float4 x = *reinterpret_cast<const float4 *>(
                    bias2 + ((int64_t(b) * h + head) * s + r) * pitch + j);
                b2[i] = x.x;
                b2[i + 1] = x.y;
                b2[i + 2] = x.z;
                b2[i + 3] = x.w;
            } else {
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    b2[i + e] = (valid && j + e < s)
                                    ? bias2[((int64_t(b) * h + head) * s + r) * pitch + j + e]
                                    : 0.f;
            }
#pragma unroll
            for (int e = 0; e < 4; ++e)
                accumulated[i + e] = 0.f;
        }
    for (int ni = group; ni < n; ni += groups) {
        const int bn = b * n + ni;
        const int64_t base = (int64_t(bn) * s * h + head) * Dim;
        load_rows_to_shared<M, typename L::Copy, Full>(q + base, query, s, h, own0);
        load_rows_to_shared<M, typename L::Copy, Full>(dout + base, query, s, h, own1);
        wait_shared_memory_writes();
        Tensor qview = ta.retile_D(rq);
        Tensor dview = ta.retile_D(rd);
        cute::copy(ca, ta.partition_S(own0), qview);
        cute::copy(ca, ta.partition_S(own1), dview);
        __syncthreads();
        load_rows_to_shared<N, typename L::Copy, Full>(k + base, key, s, h, s0);
        load_rows_to_shared<N, typename L::Copy, Full>(v + base, key, s, h, s1);
        Tensor score = partition_fragment_C(mma, Shape<Int<M>, Int<N>>{});
        Tensor dp = partition_fragment_C(mma, Shape<Int<M>, Int<N>>{});
        clear(dp);
        float *sp = reinterpret_cast<float *>(score.data());
        float *pp = reinterpret_cast<float *>(dp.data());
#pragma unroll
        for (int nt = 0; nt < N / 16; ++nt) {
            int j = key + nt * 16 + columns;
            float4 x =
                j < s ? *reinterpret_cast<const float4 *>(bias1 + int64_t(bn) * ((s + 3) & ~3) + j)
                      : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int mi = 0; mi < L::Rows; ++mi) {
                int i = (nt * L::Rows + mi) * 4;
                sp[i] = b2[i] + x.x;
                sp[i + 1] = b2[i + 1] + x.y;
                sp[i + 2] = b2[i + 2] + x.z;
                sp[i + 3] = b2[i + 3] + x.w;
            }
        }
        wait_shared_memory_writes();
        mma_from_shared_memory(score, rq, rk, ak, mma, cb, tb);
        mma_from_shared_memory(dp, rd, rv, av, mma, cb, tb);
#pragma unroll
        for (int mi = 0; mi < L::Rows; ++mi) {
            int r = query + row + mi * W * 16;
            const int64_t stat = (int64_t(bn) * s + r) * h + head;
            const float lm = (Full || r < s) ? lse[stat] : 0.f,
                        dl = (Full || r < s) ? delta[stat] : 0.f;
#pragma unroll
            for (int nt = 0; nt < N / 16; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    int i = (nt * L::Rows + mi) * 4 + e, j = key + nt * 16 + columns + e;
                    float p =
                        (Full || (r < s && j < s)) ? __builtin_exp2f(sp[i] * Log2E - lm) : 0.f;
                    accumulated[i] += p * (pp[i] - dl);
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < L::Rows; ++mi) {
        int r = query + row + mi * W * 16;
        if (Full || r < s) {
#pragma unroll
            for (int nt = 0; nt < N / 16; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    int j = key + nt * 16 + columns + e;
                    if (Full || j < s)
                        partial[(((int64_t(b) * groups + group) * h + head) * s + r) * s + j] =
                            accumulated[(nt * L::Rows + mi) * 4 + e];
                }
        }
    }
}

// dBias1：固定顺序归约各 head 的局部和；五梯度 prepared 接口仍需要此路径。
__global__ void reduce_mask_bias_gradient_kernel(const float *partial, float *output, int bn, int s,
                                                 int h) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < int64_t(bn) * s) {
        int ns = i / s, j = i % s;
        float value = 0.f;
        for (int head = 0; head < h; ++head)
            value += partial[(int64_t(ns) * h + head) * s + j];
        output[i] = value;
    }
}
// dBias2：固定顺序归约通用重算路径的 N 分组局部和。
__global__ void reduce_pair_bias_gradient_kernel(const float *partial, float *output, int b, int s,
                                                 int h, int groups) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x, stride = int64_t(h) * s * s;
    if (i < int64_t(b) * stride) {
        int batch = i / stride;
        int64_t off = i % stride;
        float value = 0.f;
        for (int group = 0; group < groups; ++group)
            value += partial[(int64_t(batch) * groups + group) * stride + off];
        output[i] = value;
    }
}

// Small-N training path: one CTA computes dQ/dK/dV and FP32 dS for one head.
// dS is reused across all three matrix products; only deterministic bias
// reductions remain separate. The matrix scratch aliases dead Q/K/V storage.
template <int Tile> struct FusedSequenceTileLayout {
    static constexpr int Waves = Tile / 16, BlockThreads = Tile * 4;
    static constexpr int Elements =
        (Tile * Tile + Tile * Dim > 4 * Tile * Dim ? Tile * Tile + Tile * Dim : 4 * Tile * Dim);
    static constexpr int Shared =
        Elements * sizeof(Element) + (2 * Tile + Tile * Waves) * sizeof(float);
    using Matrix = decltype(composition(
        Swizzle<2, 3, 3>{}, Layout<Shape<Int<Tile>, Int<Tile>>, Stride<Int<Tile>, _1>>{}));
    using MatrixT = decltype(composition(
        Swizzle<2, 3, 3>{}, Layout<Shape<Int<Tile>, Int<Tile>>, Stride<_1, Int<Tile>>>{}));
};

// 整序列融合：序列可装入单个 tile 时，复用 P/dS 完成多项梯度；同时支持 raw/prepared 输入。
template <int Tile, bool Raw>
__global__ __launch_bounds__(Tile * 4) void compute_fused_gradients_kernel(
    const Element *q, const Element *k, const Element *v, const void *bias1, const void *bias2,
    const Element *output, const float *lse, const Element *dout, float *global_delta, Element *dq,
    Element *dk, Element *dv, float *db1_head, float *db2_partial, int n, int s, int h) {
    using L = BackwardTileLayout<Tile, Tile, Tile / 16>;
    using Small = FusedSequenceTileLayout<Tile>;
    extern __shared__ char raw[];
    Element *sm = reinterpret_cast<Element *>(raw);
    Tensor sq = make_tensor(make_smem_ptr(sm), typename L::Own{});
    Tensor sk = make_tensor(make_smem_ptr(sm + Tile * Dim), typename L::Own{});
    Tensor sv = make_tensor(make_smem_ptr(sm + 2 * Tile * Dim), typename L::Own{});
    Tensor sd = make_tensor(make_smem_ptr(sm + 3 * Tile * Dim), typename L::Own{});
    float *shared_delta = reinterpret_cast<float *>(sm + Small::Elements);
    float *shared_lse = shared_delta + Tile;
    float *bias_rows = shared_lse + Tile;
    const int bn = blockIdx.z, head = blockIdx.y, b = bn / n, ni = bn % n;
    const int lane = threadIdx.x % 64, wave = threadIdx.x / 64, row = wave * 16 + lane % 16,
              columns = (lane / 16) * 4;
    const int load_row = threadIdx.x / 4, load_col = (threadIdx.x % 4) * 8;
    const int64_t base = (int64_t(bn) * s * h + head) * Dim;
    q += base;
    k += base;
    v += base;
    dout += base;
    output += base;
    dq += base;
    dk += base;
    dv += base;
    uint4 qvalue = make_uint4(0, 0, 0, 0), kvalue = qvalue, vvalue = qvalue, dvalue = qvalue,
          ovalue = qvalue;
    if (load_row < s) {
        const int64_t address = int64_t(load_row) * h * Dim + load_col;
        qvalue = *reinterpret_cast<const uint4 *>(q + address);
        kvalue = *reinterpret_cast<const uint4 *>(k + address);
        vvalue = *reinterpret_cast<const uint4 *>(v + address);
        dvalue = *reinterpret_cast<const uint4 *>(dout + address);
        ovalue = *reinterpret_cast<const uint4 *>(output + address);
    }
    mctlass::AlignedArray<Element, 8, 16> qvec, dvec, ovec;
    *reinterpret_cast<uint4 *>(qvec.data()) = qvalue;
    *reinterpret_cast<uint4 *>(dvec.data()) = dvalue;
    *reinterpret_cast<uint4 *>(ovec.data()) = ovalue;
    if constexpr (Raw) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
            qvec[i] = round_bfloat16_nearest_even(float(qvec[i]) * 0.17677669529663689f);
        qvalue = *reinterpret_cast<const uint4 *>(qvec.data());
    }
    *reinterpret_cast<uint4 *>(&sq(load_row, load_col)) = qvalue;
    *reinterpret_cast<uint4 *>(&sk(load_row, load_col)) = kvalue;
    *reinterpret_cast<uint4 *>(&sv(load_row, load_col)) = vvalue;
    *reinterpret_cast<uint4 *>(&sd(load_row, load_col)) = dvalue;
    float delta = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        delta += float(ovec[i]) * float(dvec[i]);
    delta += __shfl_xor_sync(uint64_t(-1), delta, 1, 4);
    delta += __shfl_xor_sync(uint64_t(-1), delta, 2, 4);
    if (threadIdx.x % 4 == 0) {
        shared_delta[load_row] = delta;
        if (load_row < s)
            global_delta[(int64_t(bn) * s + load_row) * h + head] = delta;
    }
    if (threadIdx.x < Tile)
        shared_lse[threadIdx.x] =
            threadIdx.x < s ? lse[(int64_t(bn) * s + threadIdx.x) * h + head] : 0.f;
    wait_shared_memory_writes();
    typename L::Mma mma;
    auto mt = mma.get_thread_slice(threadIdx.x);
    auto ca = make_tiled_copy_A(SmemCopyAtom{}, mma);
    auto ta = ca.get_thread_slice(threadIdx.x);
    auto cb = make_tiled_copy_B(SmemCopyAtom{}, mma);
    auto tb = cb.get_thread_slice(threadIdx.x);
    auto ct = make_tiled_copy_B(SmemCopyAtomTransposed{}, mma);
    auto tt = ct.get_thread_slice(threadIdx.x);
    Tensor rq = mt.partition_fragment_A(sq);
    Tensor rd = mt.partition_fragment_A(sd);
    Tensor rk = mt.partition_fragment_B(sk);
    Tensor rv = mt.partition_fragment_B(sv);
    Tensor aq = ta.partition_S(sq);
    Tensor ad = ta.partition_S(sd);
    Tensor ak = tb.partition_S(sk);
    Tensor av = tb.partition_S(sv);
    Tensor qview = ta.retile_D(rq);
    Tensor dview = ta.retile_D(rd);
    cute::copy(ca, aq, qview);
    cute::copy(ca, ad, dview);
    Tensor score = partition_fragment_C(mma, Shape<Int<Tile>, Int<Tile>>{});
    Tensor dp = partition_fragment_C(mma, Shape<Int<Tile>, Int<Tile>>{});
    clear(dp);
    float *sp = reinterpret_cast<float *>(score.data());
    float *pp = reinterpret_cast<float *>(dp.data());
#pragma unroll
    for (int nt = 0; nt < Tile / 16; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            int col = nt * 16 + columns + e, index = nt * 4 + e;
            float a = 0.f, c = 0.f;
            if (row < s && col < s) {
                if constexpr (Raw) {
                    a = float(round_bfloat16_nearest_even(
                        static_cast<const float *>(bias1)[int64_t(bn) * s + col]));
                    c = float(round_bfloat16_nearest_even(static_cast<const float *>(
                        bias2)[((int64_t(b) * h + head) * s + row) * s + col]));
                } else {
                    a = float(static_cast<const Element *>(bias1)[int64_t(bn) * s + col]);
                    c = float(static_cast<const Element *>(
                        bias2)[((int64_t(b) * h + head) * s + row) * s + col]);
                }
            }
            sp[index] = a + c;
        }
    mma_from_shared_memory(score, rq, rk, ak, mma, cb, tb);
    mma_from_shared_memory(dp, rd, rv, av, mma, cb, tb);
#pragma unroll
    for (int nt = 0; nt < Tile / 16; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            int col = nt * 16 + columns + e, index = nt * 4 + e;
            float probability =
                (row < s && col < s) ? __builtin_exp2f(sp[index] * Log2E - shared_lse[row]) : 0.f;
            float derivative = probability * (pp[index] - shared_delta[row]);
            sp[index] = probability;
            pp[index] = derivative;
            // Store the complete four-column fragment below, with a tail guard.
        }
    if (db2_partial && row < s) {
#pragma unroll
        for (int nt = 0; nt < Tile / 16; ++nt) {
            int col = nt * 16 + columns;
            float *dst = db2_partial + (((int64_t(b) * n + ni) * h + head) * s + row) * s + col;
            if (col + 3 < s && (s & 3) == 0)
                *reinterpret_cast<float4 *>(dst) =
                    make_float4(pp[nt * 4], pp[nt * 4 + 1], pp[nt * 4 + 2], pp[nt * 4 + 3]);
            else {
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    if (col + e < s)
                        dst[e] = pp[nt * 4 + e];
            }
        }
    }
    if (db1_head) {
#pragma unroll
        for (int nt = 0; nt < Tile / 16; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float value = pp[nt * 4 + e];
#pragma unroll
                for (int step = 1; step < 16; step *= 2)
                    value += __shfl_xor_sync(uint64_t(-1), value, step, 64);
                if (lane % 16 == 0)
                    bias_rows[wave * Tile + nt * 16 + columns + e] = value;
            }
    }
    constexpr int Count = Tile * Tile / (Tile * 4);
    mctlass::NumericArrayConverter<Element, float, Count> convert;
    auto probabilities =
        convert(*reinterpret_cast<const mctlass::Array<float, Count> *>(score.data()));
    auto derivatives = convert(*reinterpret_cast<const mctlass::Array<float, Count> *>(dp.data()));
    Tensor rds = make_tensor(make_rmem_ptr<Element>(&derivatives), score.layout());
    Tensor skt = make_tensor(sk.data(), typename L::Trans{});
    Tensor sktn = make_tensor(sk.data(), typename L::TransPlain{});
    Tensor rkt = mt.partition_fragment_B(sktn);
    Tensor akt = tt.partition_S(skt);
    Tensor resultq = partition_fragment_C(mma, Shape<Int<Tile>, Int<Dim>>{});
    clear(resultq);
    mma_from_shared_memory(resultq, rds, rkt, akt, mma, ct, tt);
    if constexpr (Raw) {
        float *values = reinterpret_cast<float *>(resultq.data());
        if (row < s) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                mctlass::AlignedArray<Element, 4, 8> packed;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    Element rounded = Element(values[nt * 4 + e]);
                    packed[e] = round_bfloat16_nearest_even(float(rounded) * 0.17677669529663689f);
                }
                *reinterpret_cast<uint2 *>(dq + int64_t(row) * h * Dim + nt * 16 + columns) =
                    *reinterpret_cast<const uint2 *>(packed.data());
            }
        }
    } else
        store_bfloat16_gradient<Tile / 16>(resultq, dq, 0, s, h);
    __syncthreads();
    if (db1_head) {
        if (threadIdx.x < s) {
            float value = 0.f;
#pragma unroll
            for (int w = 0; w < Tile / 16; ++w)
                value += bias_rows[w * Tile + threadIdx.x];
            db1_head[(int64_t(bn) * h + head) * s + threadIdx.x] = value;
        }
    }
    Tensor matrix = make_tensor(make_smem_ptr(sm), typename Small::Matrix{});
    Tensor transposed = make_tensor(make_smem_ptr(sm), typename Small::MatrixT{});
    Tensor temp = make_tensor(make_smem_ptr(sm + Tile * Tile), typename L::Own{});
#pragma unroll
    for (int nt = 0; nt < Tile / 16; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            matrix(row, nt * 16 + columns + e) = derivatives[nt * 4 + e];
    *reinterpret_cast<uint4 *>(&temp(load_row, load_col)) = qvalue;
    wait_shared_memory_writes();
    Tensor rt = mt.partition_fragment_A(transposed);
    Tensor tview = ta.retile_D(rt);
    cute::copy(ca, ta.partition_S(transposed), tview);
    Tensor temp_t = make_tensor(temp.data(), typename L::Trans{});
    Tensor temp_tn = make_tensor(temp.data(), typename L::TransPlain{});
    Tensor rb = mt.partition_fragment_B(temp_tn);
    Tensor ab = tt.partition_S(temp_t);
    Tensor resultk = partition_fragment_C(mma, Shape<Int<Tile>, Int<Dim>>{});
    clear(resultk);
    mma_from_shared_memory(resultk, rt, rb, ab, mma, ct, tt);
    store_bfloat16_gradient<Tile / 16>(resultk, dk, 0, s, h);
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < Tile / 16; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            matrix(row, nt * 16 + columns + e) = probabilities[nt * 4 + e];
    *reinterpret_cast<uint4 *>(&temp(load_row, load_col)) = dvalue;
    wait_shared_memory_writes();
    cute::copy(ca, ta.partition_S(transposed), tview);
    Tensor resultv = partition_fragment_C(mma, Shape<Int<Tile>, Int<Dim>>{});
    clear(resultv);
    mma_from_shared_memory(resultv, rt, rb, ab, mma, ct, tt);
    store_bfloat16_gradient<Tile / 16>(resultv, dv, 0, s, h);
}

// 融合 raw 输出：沿 N 归约 dBias2，完成约定的 BF16 舍入并以 FP32 保存。
__global__ void reduce_fused_raw_pair_bias_kernel(const float *partial, float *output, int b, int n,
                                                  int s, int h) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x, stride = int64_t(h) * s * s;
    if (i < int64_t(b) * stride) {
        const int batch = i / stride;
        const int64_t offset = i % stride;
        float value = 0.f;
        for (int ni = 0; ni < n; ++ni)
            value += partial[(int64_t(batch) * n + ni) * stride + offset];
        output[i] = float(round_bfloat16_nearest_even(value));
    }
}

template <bool Raw>
static cudaError_t
launch_fused_backward(const void *q, const void *k, const void *v, const void *b1, const void *b2,
                      const void *o, const void *lse, const void *dout, void *dq, void *dk,
                      void *dv, void *db1, void *db2, void *delta, void *db1_head,
                      void *db2_partial, int b, int n, int s, int h, cudaStream_t stream) {
#define SMALL(T)                                                                                   \
    do {                                                                                           \
        if constexpr (FusedSequenceTileLayout<T>::Shared > 32768) {                                \
            auto error = cudaFuncSetAttribute(compute_fused_gradients_kernel<T, Raw>,              \
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,         \
                                              FusedSequenceTileLayout<T>::Shared);                 \
            if (error != cudaSuccess)                                                              \
                return error;                                                                      \
        }                                                                                          \
        compute_fused_gradients_kernel<T, Raw>                                                     \
            <<<dim3(1, h, b * n), FusedSequenceTileLayout<T>::BlockThreads,                        \
               FusedSequenceTileLayout<T>::Shared, stream>>>(                                      \
                static_cast<const Element *>(q), static_cast<const Element *>(k),                  \
                static_cast<const Element *>(v), b1, b2, static_cast<const Element *>(o),          \
                static_cast<const float *>(lse), static_cast<const Element *>(dout),               \
                static_cast<float *>(delta), static_cast<Element *>(dq),                           \
                static_cast<Element *>(dk), static_cast<Element *>(dv),                            \
                static_cast<float *>(db1_head), static_cast<float *>(db2_partial), n, s, h);       \
    } while (0)
    if (s <= 32) {
        SMALL(32);
    } else {
        SMALL(64);
    }
#undef SMALL
    if (db1)
        reduce_mask_bias_gradient_kernel<<<(int64_t(b) * n * s + 255) / 256, 256, 0, stream>>>(
            static_cast<float *>(db1_head), static_cast<float *>(db1), b * n, s, h);
    if (db2) {
        const int64_t count = int64_t(b) * h * s * s;
        if constexpr (Raw)
            reduce_fused_raw_pair_bias_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
                static_cast<float *>(db2_partial), static_cast<float *>(db2), b, n, s, h);
        else
            reduce_pair_bias_gradient_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
                static_cast<float *>(db2_partial), static_cast<float *>(db2), b, s, h, n);
    }
    return cudaGetLastError();
}

// Occupancy is kernel metadata, cached per loaded library/device. No input
// shape or timing result is cached by the production policy.
// 工作区与通用调度：容量复用、驻留能力、tile 和融合路径选择。
static int device_multiprocessor_count() {
    if (!cached_multiprocessor_count) {
        int device = 0;
        cudaDeviceProp properties{};
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
            cached_multiprocessor_count = properties.multiProcessorCount;
            cached_threads_per_multiprocessor = properties.maxThreadsPerMultiProcessor;
            cached_wave_size = properties.warpSize;
        } else
            cudaGetLastError();
    }
    return cached_multiprocessor_count;
}

static cudaError_t reserve_score_gradient_workspace(size_t count) {
    if (count > SIZE_MAX / sizeof(float))
        return cudaErrorInvalidValue;
    if (count > score_gradient_capacity) {
        float *next = nullptr;
        auto error = cudaMalloc(&next, count * sizeof(float));
        if (error != cudaSuccess)
            return error;
        if (score_gradient_workspace)
            cudaFree(score_gradient_workspace);
        score_gradient_workspace = next;
        score_gradient_capacity = count;
    }
    return cudaSuccess;
}

static cudaError_t prepare_pair_bias_layouts(const Element *input, int b, int h, int s,
                                             cudaStream_t stream, float *&normal,
                                             float *&transposed) {
    const int pitch = (s + 3) & ~3;
    const size_t count = size_t(b) * h * s * pitch;
    if (count > SIZE_MAX / (2 * sizeof(float)))
        return cudaErrorInvalidValue;
    if (count > pair_bias_capacity) {
        float *next = nullptr;
        auto error = cudaMalloc(&next, 2 * count * sizeof(float));
        if (error != cudaSuccess)
            return error;
        if (pair_bias_workspace)
            cudaFree(pair_bias_workspace);
        pair_bias_workspace = next;
        pair_bias_capacity = count;
    }
    normal = pair_bias_workspace;
    transposed = pair_bias_workspace + pair_bias_capacity;
    const int tiles = (pitch + 31) / 32;
    prepare_pair_bias_layouts_kernel<<<dim3(tiles, tiles * b * h), dim3(32, 8), 0, stream>>>(
        input, normal, transposed, s, pitch);
    return cudaGetLastError();
}

template <int M, int N, bool Full, int W> static int pair_bias_active_blocks() {
    static int active = 0;
    if (!active) {
        const auto status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, compute_pair_bias_gradient_kernel<M, N, Full, W>,
            BackwardTileLayout<M, N, W>::Threads, BackwardTileLayout<M, N, W>::Shared);
        if (status != cudaSuccess) {
            cudaGetLastError();
            active = 1;
        }
    }
    return std::max(active, 1);
}
static int select_pair_bias_tile(int b, int s, int h) {
    if (s <= BwdTile)
        return 2; // Single on-chip sequence tile.
    const int sms = device_multiprocessor_count();
    const int active = s % BwdTile == 0 ? pair_bias_active_blocks<64, 64, true, 4>()
                                        : pair_bias_active_blocks<64, 64, false, 4>();
    const int64_t independent =
        int64_t(b) * h * ((s + BwdTile - 1) / BwdTile) * ((s + BwdTile - 1) / BwdTile);
    return independent < int64_t(sms) * active ? 3 : 0;
}
static int select_pair_bias_group_count(int b, int n, int s, int h) {
    const int mode = select_pair_bias_tile(b, s, h), sms = device_multiprocessor_count();
    const int m = mode == 2 ? Dim : BwdTile, k = mode == 0 ? BwdTile : Dim;
    const int active =
        mode == 0 ? (s % BwdTile == 0 ? pair_bias_active_blocks<64, 64, true, 4>()
                                      : pair_bias_active_blocks<64, 64, false, 4>())
                  : (mode == 2 ? (s % Dim == 0 ? pair_bias_active_blocks<32, 32, true, 2>()
                                               : pair_bias_active_blocks<32, 32, false, 2>())
                               : (s % BwdTile == 0 ? pair_bias_active_blocks<64, 32, true, 2>()
                                                   : pair_bias_active_blocks<64, 32, false, 2>()));
    const int64_t independent = int64_t(b) * h * ((s + m - 1) / m) * ((s + k - 1) / k);
    const int needed =
        int(std::max<int64_t>(1, (int64_t(sms) * active + independent - 1) / independent));
    // Bound partial storage. A complete on-chip sequence can share dS across
    // all three products whenever its N partials fit that budget.
    const uint64_t one_batch = uint64_t(h) * s * s * sizeof(float);
    const int capacity = int(std::max<uint64_t>(1, ((64ULL * 1024 * 1024) / one_batch) / b));
    if (s <= BwdTile && n <= capacity)
        return n;
    return std::max(1, std::min({n, capacity, needed}));
}
static bool can_fuse_sequence(int n, int s, int groups) {
    return s <= BwdTile && groups == n;
}
static bool can_use_native_raw_adapter(int64_t elements) {
    const int sms = device_multiprocessor_count();
    const int resident = std::max(1, cached_threads_per_multiprocessor / Threads);
    return (elements + Threads * 8 - 1) / (Threads * 8) >= int64_t(sms) * resident * 2;
}

// Select reuse from the actual resident CTA capacity and bounded scratch.
// No measured B/N/S entries or shape-keyed tuning cache participate.
template <int M, int N, bool Full, int G, bool RawQ> static int grouped_query_active_blocks() {
    static int active = -1;
    if (active < 0) {
        const auto error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, compute_grouped_query_bias_gradients_kernel<M, N, Full, 2, G, RawQ>, 128,
            BackwardTileLayout<M, N, 2>::Shared);
        if (error != cudaSuccess) {
            cudaGetLastError();
            active = 0;
        }
    }
    return active;
}
template <int M, int N, bool Full, bool RawQ> static int query_active_blocks() {
    static int active = -1;
    if (active < 0) {
        const auto error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, compute_query_gradient_kernel<M, N, Full, 2, false, RawQ>, 128,
            BackwardTileLayout<M, N, 2>::Shared);
        if (error != cudaSuccess) {
            cudaGetLastError();
            active = 0;
        }
    }
    return active;
}
template <bool RawQ>
static bool can_reuse_grouped_score_gradients(int b, int n, int s, int h, size_t samples) {
    if (n < 2 || samples == 0)
        return false;
    constexpr int G = BWD_DBIAS_GROUP;
    const int sms = device_multiprocessor_count();
    if (sms <= 0)
        return false;
    const int active = (s & 3)
                           ? grouped_query_active_blocks<32, 64, false, G, RawQ>()
                           : (s % 64 == 0 ? grouped_query_active_blocks<64, 32, true, G, RawQ>()
                                          : grouped_query_active_blocks<64, 32, false, G, RawQ>());
    if (active <= 0)
        return false;
    const int tile_m = (s & 3) ? 32 : 64;
    const size_t groups = size_t(b) * ((n + G - 1) / G), chunk = std::min(groups, samples);
    const uint64_t blocks = uint64_t((s + tile_m - 1) / tile_m) * h * chunk;
    // Avoid a nearly empty final scheduling wave. The original full backward
    // has nine matrix-product passes; reusing dS removes two of those passes.
    // Require the new wave utilization to preserve that 7/9 work advantage.
    // This is computed from CTA counts and kernel metadata, never timings.
    const uint64_t resident = uint64_t(sms) * active;
    if (blocks < resident)
        return false;
    const int old_active = (s & 3) ? query_active_blocks<32, 64, false, RawQ>()
                                   : (s % 64 == 0 ? query_active_blocks<64, 32, true, RawQ>()
                                                  : query_active_blocks<64, 32, false, RawQ>());
    if (old_active <= 0)
        return false;
    const uint64_t old_blocks = uint64_t((s + tile_m - 1) / tile_m) * h * b * n;
    const uint64_t old_resident = uint64_t(sms) * old_active;
    const double utilization = double(blocks) / (((blocks + resident - 1) / resident) * resident);
    const double old_utilization =
        double(old_blocks) / (((old_blocks + old_resident - 1) / old_resident) * old_resident);
    return 9.0 * utilization >= 7.0 * old_utilization;
}

// Profiling overrides are process-wide; -1 uses the resource-based policy.
static int query_tile_override = -1, key_value_tile_override = -1, pair_bias_tile_override = -1;
static int query_grid_override = -1, key_value_grid_override = -1;
} // namespace triangle_attention_backward
extern "C" int colleague_backward_group_count(int b, int n, int s, int h) {
    if (b <= 0 || n <= 0 || s <= 0 || s > 65535 || h <= 0 || h > 65535 || int64_t(b) * n > 65535)
        return 0;
    return triangle_attention_backward::select_pair_bias_group_count(b, n, s, h);
}
extern "C" int colleague_backward_raw_policy(int n, int s, int groups, int64_t elements) {
    return triangle_attention_backward::can_fuse_sequence(n, s, groups)
               ? 2
               : int(triangle_attention_backward::can_use_native_raw_adapter(elements));
}
// profiling 调参接口：由现有 profile_backward.py 调用；默认 -1 使用通用资源分派。
extern "C" int colleague_backward_set_config(int dq, int dkdv) {
    if (dq < -1 || dq > 8 || dkdv < -1 || dkdv > 8)
        return int(cudaErrorInvalidValue);
    triangle_attention_backward::query_tile_override = dq;
    triangle_attention_backward::key_value_tile_override = dkdv;
    return 0;
}
extern "C" int colleague_backward_set_bias_config(int config) {
    if (config < -1 || config > 4)
        return int(cudaErrorInvalidValue);
    triangle_attention_backward::pair_bias_tile_override = config;
    return 0;
}
extern "C" int colleague_backward_set_grid_config(int dq, int dkdv) {
    if (dq < -1 || dq > 4 || dkdv < -1 || dkdv > 4)
        return int(cudaErrorInvalidValue);
    triangle_attention_backward::query_grid_override = dq;
    triangle_attention_backward::key_value_grid_override = dkdv;
    return 0;
}
extern "C" int colleague_backward_set_fusion_mode(int mode) {
    if (mode < -1 || mode > 1)
        return int(cudaErrorInvalidValue);
    triangle_attention_backward::score_gradient_reuse_override = mode;
    return 0;
}

static int launch_backward_impl(const void *q, const void *k, const void *v, const void *b1,
                                const void *b2, const void *o, const void *lse, const void *dout,
                                void *dq, void *dk, void *dv, void *db1, void *db2, void *delta,
                                void *db1_head, void *db2_partial, int b, int n, int s, int h,
                                int groups, void *stream_ptr, bool raw_q) {
    using namespace triangle_attention_backward;
    const int64_t bn = int64_t(b) * n;
    if (!q || !k || !v || !b1 || !b2 || !o || !lse || !dout || !dq || !dk || !dv || !delta ||
        b <= 0 || n <= 0 || s <= 0 || s > 65535 || h <= 0 || h > 65535 || bn > 65535 ||
        groups <= 0 || groups > n || (db1 && !db1_head) || (db2 && !db2_partial))
        return int(cudaErrorInvalidValue);
    auto stream = static_cast<cudaStream_t>(stream_ptr);
    auto qb = static_cast<const Element *>(q), kb = static_cast<const Element *>(k),
         vb = static_cast<const Element *>(v);
    auto go = static_cast<const Element *>(dout);
#ifndef BWD_SMALL_FUSED
#define BWD_SMALL_FUSED 1
#endif
    if (BWD_SMALL_FUSED && can_fuse_sequence(n, s, groups))
        return int(launch_fused_backward<false>(q, k, v, b1, b2, o, lse, dout, dq, dk, dv, db1, db2,
                                                delta, db1_head, db2_partial, b, n, s, h, stream));
    auto b1f = bias1_get_f32_padded(static_cast<const __nv_bfloat16 *>(b1), b, n, s, stream);
    float *b2f = nullptr, *b2t = nullptr;
    auto bias_status =
        prepare_pair_bias_layouts(static_cast<const Element *>(b2), b, h, s, stream, b2f, b2t);
    if (bias_status != cudaSuccess)
        return int(bias_status);
    const size_t ds_stride = size_t(h) * s * ((s + 3) & ~3);
    const size_t ds_samples = size_t(BWD_DS_WORKSPACE_BYTES) / (sizeof(float) * ds_stride);
    // Compare FP32 dS write/read traffic with replaying Q/K/V/dO. This works
    // for aligned and ragged tiles, independently of concrete B/N/S values.
    const uint64_t replay_bytes = uint64_t(h) * s * Dim * sizeof(Element) * 4;
    const bool auto_fuse =
        BWD_FUSE_DBIAS2 != 0 && ds_stride * sizeof(float) * 2 <= replay_bytes * 2;
    const bool grouped = db2 && !auto_fuse && score_gradient_reuse_override < 0 &&
                         query_tile_override < 0 && query_grid_override < 0 &&
                         (raw_q ? can_reuse_grouped_score_gradients<true>(b, n, s, h, ds_samples)
                                : can_reuse_grouped_score_gradients<false>(b, n, s, h, ds_samples));
    const bool fused =
        grouped ||
        (db2 && ds_samples > 0 &&
         (score_gradient_reuse_override >= 0 ? score_gradient_reuse_override != 0 : auto_fuse));
    const size_t work_items =
        grouped ? size_t(b) * ((n + BWD_DBIAS_GROUP - 1) / BWD_DBIAS_GROUP) : size_t(bn);
    const int chunk_size = fused ? int(std::min(work_items, ds_samples)) : int(bn);
    if (fused) {
        auto error = reserve_score_gradient_workspace(size_t(chunk_size) * ds_stride);
        if (error != cudaSuccess)
            return int(error);
    }
    int64_t stats = bn * s * h;
    compute_output_gradient_dot_kernel<<<(stats + 63) / 64, Threads, 0, stream>>>(
        static_cast<const Element *>(o), go, static_cast<float *>(delta), stats);
    const int sms = device_multiprocessor_count();
    const bool enough_matrices =
        sms > 0 &&
        bn * h >= int64_t(sms) * std::max(1, cached_threads_per_multiprocessor / Threads) * 2;
    const int dq_mode =
        query_tile_override >= 0
            ? query_tile_override
            : ((s & 3) ? (s <= BwdTile ? 0 : 4)
                       : (s <= Dim ? 2 : (s <= BwdTile && !enough_matrices ? 5 : 6)));
    const int dkdv_mode =
        key_value_tile_override >= 0
            ? key_value_tile_override
            : ((s & 3) ? (s <= BwdTile ? 0 : 4) : (s <= BwdTile && !enough_matrices ? 2 : 5));
    const int dq_log2 = query_grid_override >= 0 ? query_grid_override : 0;
    const int dkdv_log2 = key_value_grid_override >= 0 ? key_value_grid_override : 0;
    const int dq_group = 1 << dq_log2, dkdv_group = 1 << dkdv_log2;
#define RUN_DQ_KERNEL(M, N, F, W, SAVE, RAW)                                                       \
    compute_query_gradient_kernel<M, N, F, W, SAVE, RAW>                                           \
        <<<dim3((s + M - 1) / M * dq_group, h, (chunk_count + dq_group - 1) / dq_group),           \
           BackwardTileLayout<M, N, W>::Threads, BackwardTileLayout<M, N, W>::Shared, stream>>>(   \
            qb, kb, vb, b1f, b2f, static_cast<const float *>(lse), static_cast<float *>(delta),    \
            go, static_cast<Element *>(dq), n, s, h, chunk_count, dq_log2, chunk_begin,            \
            score_gradient_workspace)
#define RUN_DQ_IMPL(M, N, F, W, SAVE)                                                              \
    do {                                                                                           \
        if (raw_q) {                                                                               \
            RUN_DQ_KERNEL(M, N, F, W, SAVE, true);                                                 \
        } else {                                                                                   \
            RUN_DQ_KERNEL(M, N, F, W, SAVE, false);                                                \
        }                                                                                          \
    } while (0)
#define RUN_DQ(M, N, F, W)                                                                         \
    do {                                                                                           \
        if (save_ds) {                                                                             \
            RUN_DQ_IMPL(M, N, F, W, true);                                                         \
        } else {                                                                                   \
            RUN_DQ_IMPL(M, N, F, W, false);                                                        \
        }                                                                                          \
    } while (0)
#define RUN_DKDV(M, N, F, W)                                                                       \
    compute_key_value_gradients_kernel<M, N, F, W>                                                 \
        <<<dim3((s + M - 1) / M * dkdv_group, h, (bn + dkdv_group - 1) / dkdv_group),              \
           BackwardTileLayout<M, N, W>::Threads, BackwardTileLayout<M, N, W>::Shared, stream>>>(   \
            qb, kb, vb, b1f, b2t, static_cast<const float *>(lse), static_cast<float *>(delta),    \
            go, static_cast<Element *>(dk), static_cast<Element *>(dv),                            \
            static_cast<float *>(db1_head), n, s, h, bn, dkdv_log2)
    auto launch_dq = [&](int chunk_begin, int chunk_count, bool save_ds) {
        switch (dq_mode) {
        case 5:
            if (s % 32 == 0) {
                RUN_DQ(32, 32, true, 1);
            } else {
                RUN_DQ(32, 32, false, 1);
            }
            break;
        case 6:
            if (s % 64 == 0) {
                RUN_DQ(64, 32, true, 2);
            } else {
                RUN_DQ(64, 32, false, 2);
            }
            break;
        case 7:
            RUN_DQ(64, 64, false, 4);
            break;
        case 8:
            RUN_DQ(32, 32, false, 2);
            break;
        case 1:
            RUN_DQ(16, 32, false, 1);
            break;
        case 2:
            if (s % 32 == 0) {
                RUN_DQ(32, 32, true, 2);
            } else {
                RUN_DQ(32, 32, false, 2);
            }
            break;
        case 3:
            RUN_DQ(64, 32, false, 4);
            break;
        case 4:
            RUN_DQ(32, 64, false, 2);
            break;
        default:
            if (s % 64 == 0) {
                RUN_DQ(64, 64, true, 4);
            } else {
                RUN_DQ(64, 64, false, 4);
            }
            break;
        }
    };
    if (grouped) {
        constexpr int G = BWD_DBIAS_GROUP;
        const int groups_per_batch = (n + G - 1) / G, total_groups = b * groups_per_batch;
        const int chunk_groups = int(std::min(size_t(total_groups), ds_samples));
        auto launch_group = [&](int begin, int count) {
#define GROUP_CALL(M, N, F, R)                                                                     \
    compute_grouped_query_bias_gradients_kernel<M, N, F, 2, G, R>                                  \
        <<<dim3((s + M - 1) / M, h, count), 128, BackwardTileLayout<M, N, 2>::Shared, stream>>>(   \
            qb, kb, vb, b1f, b2f, static_cast<const float *>(lse), static_cast<float *>(delta),    \
            go, static_cast<Element *>(dq), n, s, h, begin, count, score_gradient_workspace)
#define GROUP_RUN(M, N, F)                                                                         \
    do {                                                                                           \
        if (raw_q) {                                                                               \
            GROUP_CALL(M, N, F, true);                                                             \
        } else {                                                                                   \
            GROUP_CALL(M, N, F, false);                                                            \
        }                                                                                          \
    } while (0)
            if ((s & 3) == 0) {
                if (s % 64 == 0) {
                    GROUP_RUN(64, 32, true);
                } else {
                    GROUP_RUN(64, 32, false);
                }
            } else {
                GROUP_RUN(32, 64, false);
            }
#undef GROUP_RUN
#undef GROUP_CALL
        };
        for (int begin = 0; begin < total_groups; begin += chunk_groups) {
            const int count = std::min(chunk_groups, total_groups - begin);
            launch_group(begin, count);
            const int batches =
                (begin + count - 1) / groups_per_batch - begin / groups_per_batch + 1;
            const int64_t elements = int64_t(h) * s * s;
            reduce_score_gradients_by_batch_kernel<<<dim3((elements + 255) / 256, batches), 256, 0,
                                                     stream>>>(
                score_gradient_workspace, static_cast<float *>(db2), begin / groups_per_batch,
                begin % groups_per_batch, count, groups_per_batch, s, h);
        }
    } else if (fused) {
        // A bounded contiguous B*N chunk can span several batches. Each
        // reduction CTA owns one batch and sums its N slice in fixed order.
        for (int first = 0; first < bn; first += chunk_size) {
            const int count = std::min(chunk_size, int(bn) - first);
            launch_dq(first, count, true);
            const int batches = (first + count - 1) / n - first / n + 1;
            const int64_t elements = int64_t(h) * s * s;
            reduce_score_gradients_by_batch_kernel<<<dim3((elements + 255) / 256, batches), 256, 0,
                                                     stream>>>(score_gradient_workspace,
                                                               static_cast<float *>(db2), first / n,
                                                               first % n, count, n, s, h);
        }
    } else
        launch_dq(0, int(bn), false);
    switch (dkdv_mode) {
    case 5:
        if (s % 32 == 0) {
            RUN_DKDV(32, 32, true, 1);
        } else {
            RUN_DKDV(32, 32, false, 1);
        }
        break;
    case 6:
        if (s % 64 == 0) {
            RUN_DKDV(64, 32, true, 2);
        } else {
            RUN_DKDV(64, 32, false, 2);
        }
        break;
    case 7:
        RUN_DKDV(64, 64, false, 4);
        break;
    case 8:
        RUN_DKDV(32, 32, false, 2);
        break;
    case 1:
        RUN_DKDV(16, 32, false, 1);
        break;
    case 2:
        if (s % 32 == 0) {
            RUN_DKDV(32, 32, true, 2);
        } else {
            RUN_DKDV(32, 32, false, 2);
        }
        break;
    case 3:
        RUN_DKDV(64, 32, false, 4);
        break;
    case 4:
        RUN_DKDV(32, 64, false, 2);
        break;
    default:
        RUN_DKDV(64, 64, false, 4);
        break;
    }
#undef RUN_DQ
#undef RUN_DQ_IMPL
#undef RUN_DQ_KERNEL
#undef RUN_DKDV
    if (db1)
        reduce_mask_bias_gradient_kernel<<<(bn * s + 255) / 256, 256, 0, stream>>>(
            static_cast<float *>(db1_head), static_cast<float *>(db1), bn, s, h);
    if (db2 && !fused) {
        const int bias_groups = groups;
#define RUN_DB2(M, N, F, W)                                                                        \
    compute_pair_bias_gradient_kernel<M, N, F, W>                                                  \
        <<<dim3(((s + M - 1) / M) * ((s + N - 1) / N), h, b * bias_groups),                        \
           BackwardTileLayout<M, N, W>::Threads, BackwardTileLayout<M, N, W>::Shared, stream>>>(   \
            qb, kb, vb, b1f, b2f, static_cast<const float *>(lse), static_cast<float *>(delta),    \
            go, static_cast<float *>(db2_partial), n, s, h, bias_groups)
        const int bias_mode =
            pair_bias_tile_override >= 0 ? pair_bias_tile_override : select_pair_bias_tile(b, s, h);
        switch (bias_mode) {
        case 1:
            if (s % 32 == 0) {
                RUN_DB2(32, 32, true, 1);
            } else {
                RUN_DB2(32, 32, false, 1);
            }
            break;
        case 2:
            if (s % 32 == 0) {
                RUN_DB2(32, 32, true, 2);
            } else {
                RUN_DB2(32, 32, false, 2);
            }
            break;
        case 3:
            if (s % 64 == 0) {
                RUN_DB2(64, 32, true, 2);
            } else {
                RUN_DB2(64, 32, false, 2);
            }
            break;
        case 4:
            if (s % 64 == 0) {
                RUN_DB2(32, 64, true, 2);
            } else {
                RUN_DB2(32, 64, false, 2);
            }
            break;
        default:
            if (s % 64 == 0) {
                RUN_DB2(64, 64, true, 4);
            } else {
                RUN_DB2(64, 64, false, 4);
            }
            break;
        }
#undef RUN_DB2
        int64_t count = int64_t(b) * h * s * s;
        reduce_pair_bias_gradient_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
            static_cast<float *>(db2_partial), static_cast<float *>(db2), b, s, h, bias_groups);
    }
    return int(cudaGetLastError());
}

// prepared-Q 五梯度接口：Q 已缩放，偏置为 BF16；所有 C ABI 名称和参数保持 v1 兼容。
extern "C" int colleague_backward(const void *q, const void *k, const void *v, const void *b1,
                                  const void *b2, const void *o, const void *lse, const void *dout,
                                  void *dq, void *dk, void *dv, void *db1, void *db2, void *delta,
                                  void *db1_head, void *db2_partial, int b, int n, int s, int h,
                                  int groups, void *stream) {
    return launch_backward_impl(q, k, v, b1, b2, o, lse, dout, dq, dk, dv, db1, db2, delta,
                                db1_head, db2_partial, b, n, s, h, groups, stream, false);
}

extern "C" int colleague_prepare_raw(const void *q, const void *b1, const void *b2, void *qs,
                                     void *b1s, void *b2s, int64_t qc, int64_t c1, int64_t c2,
                                     void *stream) {
    if (!q || !b1 || !b2 || !qs || !b1s || !b2s || qc <= 0 || c1 <= 0 || c2 <= 0)
        return int(cudaErrorInvalidValue);
    const int64_t vectors = (qc + 7) / 8 + (c1 + 7) / 8 + (c2 + 7) / 8;
    triangle_attention_backward::prepare_raw_inputs_kernel<<<(vectors + 255) / 256, 256, 0,
                                                             static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Element *>(q), static_cast<const float *>(b1),
        static_cast<const float *>(b2), static_cast<Element *>(qs), static_cast<Element *>(b1s),
        static_cast<Element *>(b2s), qc, c1, c2);
    return int(cudaGetLastError());
}

extern "C" int colleague_restore_raw(const void *dq, const void *db, void *out, void *bias_out,
                                     int64_t qc, int64_t bc, void *stream) {
    if (qc < 0 || bc < 0 || (!qc && !bc) || (qc && (!dq || !out)) || (bc && (!db || !bias_out)))
        return int(cudaErrorInvalidValue);
    const int64_t vectors = (qc + 7) / 8 + (bc + 7) / 8;
    triangle_attention_backward::restore_raw_gradients_kernel<<<
        (vectors + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Element *>(dq), static_cast<const float *>(db),
        static_cast<Element *>(out), static_cast<float *>(bias_out), qc, bc);
    return int(cudaGetLastError());
}

// Complete raw-input backward. Caller-owned temporaries have the same shapes
// as the existing prepared ABI; cached-state callers can skip input preparation.
extern "C" int colleague_backward_raw_workspace(
    const void *q, const void *k, const void *v, const void *b1, const void *b2, const void *o,
    const void *lse, const void *dout, void *dq, void *dk, void *dv, void *db1, void *db2,
    void *delta, void *db1_head, void *db2_partial, void *qs, void *b1s, void *b2s, void *dq_scaled,
    void *db2_unrounded, int b, int n, int s, int h, int groups, int prepare_inputs, void *stream) {
    if (!q || !k || !v || !b1 || !b2 || !o || !lse || !dout || !dq || !dk || !dv || !delta || !qs ||
        !b1s || !b2s || !dq_scaled || db1 || db1_head ||
        (db2 && (!db2_partial || !db2_unrounded)) || b <= 0 || n <= 0 || s <= 0 || s > 65535 ||
        h <= 0 || h > 65535 || int64_t(b) * n > 65535 || groups <= 0 || groups > n ||
        prepare_inputs < 0 || prepare_inputs > 1)
        return int(cudaErrorInvalidValue);
    const int64_t qc = int64_t(b) * n * s * h * triangle_attention_backward::Dim,
                  c1 = int64_t(b) * n * s, c2 = int64_t(b) * h * s * s;
    if (triangle_attention_backward::can_fuse_sequence(n, s, groups))
        return int(triangle_attention_backward::launch_fused_backward<true>(
            q, k, v, b1, b2, o, lse, dout, dq, dk, dv, nullptr, db2, delta, nullptr, db2_partial, b,
            n, s, h, static_cast<cudaStream_t>(stream)));
    if (prepare_inputs) {
        const int error = colleague_prepare_raw(q, b1, b2, qs, b1s, b2s, qc, c1, c2, stream);
        if (error)
            return error;
    }
    const int error = launch_backward_impl(
        qs, k, v, b1s, b2s, o, lse, dout, dq, dk, dv, nullptr, db2 ? db2_unrounded : nullptr, delta,
        nullptr, db2 ? db2_partial : nullptr, b, n, s, h, groups, stream, true);
    if (error)
        return error;
    // dQ was rounded/scaled/rounded in its MMA epilogue. Only the much smaller
    // FP32 bias output still needs the original BF16 round trip.
    return db2 ? colleague_restore_raw(nullptr, db2_unrounded, nullptr, db2, 0, c2, stream) : 0;
}

// Raw-Q/FP32-bias entry for the same small-shape training contract. It fuses
// raw-input conversion and the explicitly rounded raw-Q/bias gradient stores.
// 兼容早期 compare_triton.py 的融合 raw 入口；正式 Python 接口由 raw_workspace 统一选路。
extern "C" int colleague_backward_small_raw(const void *q, const void *k, const void *v,
                                            const void *b1, const void *b2, const void *o,
                                            const void *lse, const void *dout, void *dq, void *dk,
                                            void *dv, void *db1, void *db2, void *delta,
                                            void *db1_head, void *db2_partial, int b, int n, int s,
                                            int h, int groups, void *stream_ptr) {
    if (!q || !k || !v || !b1 || !b2 || !o || !lse || !dout || !dq || !dk || !dv || !delta ||
        b <= 0 || n <= 0 || s <= 0 || s > triangle_attention_backward::BwdTile || h <= 0 ||
        h > 65535 || int64_t(b) * n > 65535 || groups != n || (db1 && !db1_head) ||
        (db2 && !db2_partial))
        return int(cudaErrorInvalidValue);
    return int(triangle_attention_backward::launch_fused_backward<true>(
        q, k, v, b1, b2, o, lse, dout, dq, dk, dv, db1, db2, delta, db1_head, db2_partial, b, n, s,
        h, static_cast<cudaStream_t>(stream_ptr)));
}
