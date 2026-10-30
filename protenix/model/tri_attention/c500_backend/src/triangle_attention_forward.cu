/**
 * C500 配套训练前向，沿用同事版 CuTe/mctlass FlashAttention2 结构。
 * Q/K/V/O 为连续 BF16 [B,N,S,H,32]；Q 已由调用者缩放并舍入为 BF16。
 * Bias1 [B,N,S]、Bias2 [B,H,S,S] 经 FP32 工作区读入分数累加器。
 * O 之外可保存 FP32 log2-LSE [B,N,S,H]，供独立反向读取。
 * 保留在线 softmax、寄存器复用及预取；使用 64x64 tile 和 256 个线程。
 */

#include "triangle_attention_config.h"

// ============================================================
// Utility: Allreduce (wave-level shuffle reduction)
// ============================================================
struct WaveReduce64 {
    template <typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        auto x1 = __shfl_xor_sync(uint64_t(-1), x, 48);
        auto x2 = __shfl_xor_sync(uint64_t(-1), x, 32);
        auto x3 = __shfl_xor_sync(uint64_t(-1), x, 16);
        return op(op(op(x, x1), x2), x3);
    }
};

// ============================================================
// Utility: MaxOp, SumOp
// ============================================================
struct MaxOp {
    __device__ __forceinline__ float operator()(float x, float y) { return fmaxf(x, y); }
};

struct SumOp {
    __device__ __forceinline__ float operator()(float x, float y) { return x + y; }
};

// 等待同步拷贝的共享内存写入可见，再由整个 CTA 读取。
__device__ __forceinline__ void wait_shared_memory_writes() {
    __builtin_mxc_arrive(64);
    __syncthreads();
}

// ============================================================
// Utility: convert_layout_acc_rowcol (MACA version)
//   acc layout: (MMA=4, MMA_M, MMA_N)
//   → (nrow=(1, MMA_M), ncol=(4, MMA_N))
// ============================================================
template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    return make_layout(make_layout(cute::Layout<_1>{}, get<1>(acc_layout)),
                       make_layout(get<0>(acc_layout), get<2>(acc_layout)));
}

// ============================================================
// Utility: gemm_rs (A already in registers, only B from smem)
// ============================================================
template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
          typename TiledMmaT, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void
mma_with_register_left_operand(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsB,
                               TiledMmaT tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

// ============================================================
// Utility: predicated copy (global → smem)
// ============================================================
template <bool Is_even_MN = true, bool Clear_OOB = true, typename TiledCopy, typename TensorS,
          typename TensorD, typename TensorC>
__forceinline__ __device__ void copy_rows_with_bounds(TiledCopy tiled_copy, TensorS const &S,
                                                      TensorD &D, TensorC const &identity_MN,
                                                      int max_MN) {
#pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
            }
        } else if (Clear_OOB) {
            cute::clear(D(_, m, _));
        }
    }
}

using Bias1Element = float;
using Bias2Element = float;

// ============================================================
// 设备计算：在线 softmax 与两次 MMA，Q 已由调用者缩放并舍入为 BF16。
// ============================================================
template <bool FullTiles>
__global__ __launch_bounds__(kNThreads, 1) void compute_attention_forward_kernel(
    const Element *__restrict__ Q_ptr, const Element *__restrict__ K_ptr,
    const Element *__restrict__ V_ptr, const Bias1Element *__restrict__ Bias1_ptr,
    const Bias2Element *__restrict__ Bias2_ptr, Element *__restrict__ O_ptr,
    float *__restrict__ Lse_ptr, int B, int N, int S, int H) {
    int m_block;
    int bidh;
    int bidb;
    int bidn;

    m_block = blockIdx.x;
    bidh = blockIdx.y;
    const int bidb_n = blockIdx.z; // b * N + n
    bidb = bidb_n / N;
    bidn = bidb_n % N;

    const int tidx = threadIdx.x;

    // Early exit if this block is fully out of bounds
    if (m_block * kBlockM >= S)
        return;

    // ---- Strides for [B, N, S, H, D] layout ----
    const int row_stride = H * kHeadDim; // stride along S dimension
    const int head_stride = kHeadDim;    // stride along H dimension

    // ---- Compute base pointers ----
    // QKV base: [bidb, bidn, 0, bidh, 0]
    const index_t qkv_offset = (index_t)bidb * N * S * H * kHeadDim +
                               (index_t)bidn * S * H * kHeadDim + bidh * head_stride;

    // Bias1 base: [bidb, bidn, 0]
    const int bias1_stride = (S + 3) & ~3;
    const index_t bias1_offset = ((index_t)bidb * N + bidn) * bias1_stride;

    // Bias2 base: [bidb, bidh, 0, 0]  (float, shape B,H,S,S)
    const index_t bias2_offset = (index_t)bidb * H * S * S + (index_t)bidh * S * S;

    // ---- Shared memory ----
    extern __shared__ char smem_[];

    // sQ and sK share the same memory (Q loaded first, then overwritten by K)
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), SmemLayoutVt{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), SmemLayoutVtNoSwizzle{});

    // ---- Global memory tensors ----
    Tensor gQ = make_tensor(make_gmem_ptr(Q_ptr + qkv_offset + m_block * kBlockM * row_stride),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(row_stride, _1{}));

    // K and V start from the first N-block
    Tensor gK = make_tensor(make_gmem_ptr(K_ptr + qkv_offset), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(V_ptr + qkv_offset), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(row_stride, _1{}));

    // ---- Tiled copies ----
    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // ---- MMA partitioning ----
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    // Output accumulator
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_o);

    // ---- Smem copy retiling ----
    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // ---- Predicates ----
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    // ---- Pre-compute per-element (row, col) coordinates for bias loading ----
    // CuTe partition_C layout for MACA 16x16x16 with 4 warps along M:
    //   Each thread has 16 elements: 4 N-tiles × 4 consecutive columns per tile
    //   row = wave_id * 16 + (lane_id % 16)  [FIXED for all 16 elements]
    //   col = tn * 16 + (lane_id / 16) * 4 + e  [e=0..3 within each N-tile]
    const int wave_id = tidx / 64;
    const int lane_id = tidx % 64;
    const int my_row = wave_id * 16 + (lane_id % 16); // fixed row for this thread
    const int col_base_in_tile = (lane_id / 16) * 4;  // starting col within a 16-col N-tile

    // Identity tensor (needed for epilogue output store predicate)
    Tensor cS = make_identity_tensor(make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
    Tensor tScS = thr_mma.partition_C(cS);

    // ================================================================
    // PROLOGUE: Load Q → smem → registers
    // ================================================================

    if constexpr (FullTiles) {
        copy_rows_with_bounds<true, false>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, 0);
    } else {
        copy_rows_with_bounds<false, true>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ,
                                           S - m_block * kBlockM);
    }
    cute::cp_async_fence();
    wait_shared_memory_writes();

    // Load Q from smem to registers
    {
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }
    __syncthreads();

    // ================================================================
    // N-LOOP: iterate over K/V blocks
    // ================================================================
    const int n_blocks = (S + kBlockN - 1) / kBlockN;

    // Softmax state
    constexpr int kNRows = size<1>(acc_o);
    float row_max[kNRows];
    float row_sum[kNRows];

#pragma unroll
    for (int i = 0; i < kNRows; ++i) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.0f;
    }

    // ---- Pre-compute bias pointers (constant across N-loop) ----
    const int global_row = m_block * kBlockM + my_row;
    const bool row_valid = (global_row < S);
    const Bias2Element *bias2_row = Bias2_ptr + bias2_offset + (index_t)global_row * S;
    const Bias1Element *bias1_base = Bias1_ptr + bias1_offset;

    for (int n_block = 0; n_block < n_blocks; ++n_block) {
        // ---- Load K to smem ----
        const int k_offset = n_block * kBlockN;
        if (n_block > 0) {
            tKgK.data() = tKgK.data() + int(kBlockN * row_stride);
        }

        const bool is_last_block = !FullTiles && (k_offset + kBlockN > S);
        if constexpr (FullTiles) {
            copy_rows_with_bounds<true, false>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, 0);
        } else {
            if (is_last_block) {
                copy_rows_with_bounds<false, true>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV,
                                                   S - k_offset);
            } else {
                copy_rows_with_bounds<true, false>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, 0);
            }
        }
        cute::cp_async_fence();

        // ---- Initialize acc_s with Bias (Scheme B, load parallel with K async) ----
        // K is in-flight via async copy. Bias loads issued here overlap with K latency.
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        {
            float *acc_s_ptr = reinterpret_cast<float *>(acc_s.data());
#pragma unroll
            for (int tn = 0; tn < kBlockN / 16; ++tn) {
                int col_start = k_offset + tn * 16 + col_base_in_tile;
                if (row_valid && col_start + 3 < S) {
                    float4 b2_vec = *reinterpret_cast<const float4 *>(&bias2_row[col_start]);
                    // Padded rows guarantee 16B alignment for every float4 load.
                    float4 b1_vec = *reinterpret_cast<const float4 *>(&bias1_base[col_start]);
                    acc_s_ptr[tn * 4 + 0] = b1_vec.x + b2_vec.x;
                    acc_s_ptr[tn * 4 + 1] = b1_vec.y + b2_vec.y;
                    acc_s_ptr[tn * 4 + 2] = b1_vec.z + b2_vec.z;
                    acc_s_ptr[tn * 4 + 3] = b1_vec.w + b2_vec.w;
                } else if (row_valid) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        int gc = col_start + e;
                        if (gc < S) {
                            acc_s_ptr[tn * 4 + e] = bias1_base[gc] + bias2_row[gc];
                        } else {
                            acc_s_ptr[tn * 4 + e] = -INFINITY;
                        }
                    }
                } else {
                    acc_s_ptr[tn * 4 + 0] = -INFINITY;
                    acc_s_ptr[tn * 4 + 1] = -INFINITY;
                    acc_s_ptr[tn * 4 + 2] = -INFINITY;
                    acc_s_ptr[tn * 4 + 3] = -INFINITY;
                }
            }
        }

        // Wait for K load to complete
        wait_shared_memory_writes();

        // ---- QK GEMM: S = Q * K^T + bias ----
        mma_with_register_left_operand(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K,
                                       smem_thr_copy_K);

        // ---- Load V to smem ----
        if (n_block > 0) {
            tVgV.data() = tVgV.data() + int(kBlockN * row_stride);
        }
        __syncthreads();

        if constexpr (FullTiles) {
            copy_rows_with_bounds<true, false>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, 0);
        } else {
            if (is_last_block) {
                copy_rows_with_bounds<false, true>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV,
                                                   S - k_offset);
            } else {
                copy_rows_with_bounds<true, false>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, 0);
            }
        }
        cute::cp_async_fence();
        wait_shared_memory_writes();

        // ---- Online Softmax ----
        {
            Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
            const float scale = M_LOG2E; // Q is pre-scaled by 1/sqrt(D)

            if (n_block == 0) {
                MaxOp max_op;
#pragma unroll
                for (int mi = 0; mi < size<0>(scores); ++mi) {
                    row_max[mi] = scores(mi, 0);
#pragma unroll
                    for (int ni = 1; ni < size<1>(scores); ++ni) {
                        row_max[mi] = max_op(row_max[mi], scores(mi, ni));
                    }
                    row_max[mi] = WaveReduce64::run(row_max[mi], max_op);
                }

#pragma unroll
                for (int mi = 0; mi < size<0>(scores); ++mi) {
                    float max_scaled = row_max[mi] == -INFINITY ? 0.f : row_max[mi] * scale;
                    row_sum[mi] = 0.f;
#pragma unroll
                    for (int ni = 0; ni < size<1>(scores); ++ni) {
                        scores(mi, ni) = __builtin_exp2f(scores(mi, ni) * scale - max_scaled);
                        row_sum[mi] += scores(mi, ni);
                    }
                }
            } else {
                MaxOp max_op;
                float scores_max_prev[kNRows];
#pragma unroll
                for (int mi = 0; mi < size<0>(scores); ++mi) {
                    scores_max_prev[mi] = row_max[mi];
#pragma unroll
                    for (int ni = 0; ni < size<1>(scores); ++ni) {
                        row_max[mi] = max_op(row_max[mi], scores(mi, ni));
                    }
                    row_max[mi] = WaveReduce64::run(row_max[mi], max_op);
                }

                // Rescale acc_o and row_sum
                Tensor acc_o_rowcol =
                    make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
                for (int mi = 0; mi < size<0>(scores); ++mi) {
                    float scores_max_cur = row_max[mi] == -INFINITY ? 0.f : row_max[mi];
                    float scores_scale =
                        __builtin_exp2f((scores_max_prev[mi] - scores_max_cur) * scale);
                    row_sum[mi] *= scores_scale;
#pragma unroll
                    for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                        acc_o_rowcol(mi, ni) *= scores_scale;
                    }
                }

// Apply exp2 to current scores
#pragma unroll
                for (int mi = 0; mi < size<0>(scores); ++mi) {
                    float max_scaled = row_max[mi] == -INFINITY ? 0.f : row_max[mi] * scale;
#pragma unroll
                    for (int ni = 0; ni < size<1>(scores); ++ni) {
                        scores(mi, ni) = __builtin_exp2f(scores(mi, ni) * scale - max_scaled);
                        row_sum[mi] += scores(mi, ni);
                    }
                }
            }
        }

        // ---- Convert acc_s (fp32) → P (bf16) in registers ----
        constexpr int acc_s_numel = decltype(size(acc_s))::value;
        mctlass::NumericArrayConverter<Element, ElementAccum, acc_s_numel> convert_op;
        auto rP_frag = convert_op(
            *reinterpret_cast<const mctlass::Array<ElementAccum, acc_s_numel> *>(acc_s.data()));
        Tensor rP = make_tensor(make_rmem_ptr<Element>(&rP_frag), acc_s.layout());

        // ---- PV GEMM: O += P * V^T ----
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        mma_with_register_left_operand(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V,
                                       smem_thr_copy_V);
    } // end N-loop

    // ================================================================
    // EPILOGUE: Normalize + store
    // ================================================================
    {
        SumOp sum_op;
#pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            row_sum[mi] = WaveReduce64::run(row_sum[mi], sum_op);
        }

        // Training-only auxiliary output; the original O path is unchanged.
        static_assert(kNRows == 1, "LSE mapping requires the default 64x64 tile");
        if (Lse_ptr && lane_id < 16 && global_row < S) {
            const float sum = row_sum[0];
            const float value = sum > 0.f ? row_max[0] * float(M_LOG2E) + log2f(sum) : -INFINITY;
            Lse_ptr[((int64_t(bidb) * N + bidn) * S + global_row) * H + bidh] = value;
        }

        // Normalize acc_o
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float inv_sum =
                (row_sum[mi] == 0.f || row_sum[mi] != row_sum[mi]) ? 1.f : 1.f / row_sum[mi];
#pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                acc_o_rowcol(mi, ni) *= inv_sum;
            }
        }

        // Convert fp32 → bf16
        constexpr int acc_o_numel = decltype(size(acc_o))::value;
        mctlass::NumericArrayConverter<Element, ElementAccum, acc_o_numel> convert_o;
        auto rO_frag = convert_o(
            *reinterpret_cast<const mctlass::Array<ElementAccum, acc_o_numel> *>(acc_o.data()));
        Tensor rO = make_tensor(make_rmem_ptr<Element>(&rO_frag), acc_o.layout());

        // Write to smem (using O layout)
        Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), SmemLayoutO{});
        auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

        __syncthreads();
        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
        __syncthreads();

        // Store from smem to global
        const index_t row_offset_out = qkv_offset + m_block * kBlockM * row_stride;
        Tensor gO =
            make_tensor(make_gmem_ptr(O_ptr + row_offset_out), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_stride(row_stride, _1{}));

        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

        Tensor tOrO_out = make_tensor<Element>(shape(tOgO));
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO_out);

        // Predicated store (handle OOB rows)
        Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);

#pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            if constexpr (FullTiles) {
#pragma unroll
                for (int k = 0; k < size<2>(tOgO); ++k) {
                    cute::copy(tOrO_out(_, m, k), tOgO(_, m, k));
                }
            } else if (get<0>(tOcO(0, m, 0)) < S - m_block * kBlockM) {
#pragma unroll
                for (int k = 0; k < size<2>(tOgO); ++k) {
                    cute::copy(tOrO_out(_, m, k), tOgO(_, m, k));
                }
            }
        }
    }
}

// ============================================================
// 主机调度：每次刷新 FP32 偏置，按 tile 对齐情况选择完整或带边界检查的路径。
// ============================================================

#include "pair_bias_cache.h"

namespace triangle_attention_forward {

void launch(const __nv_bfloat16 *Q, const __nv_bfloat16 *K, const __nv_bfloat16 *V,
            const __nv_bfloat16 *Bias1, const __nv_bfloat16 *Bias2, __nv_bfloat16 *O, float *Lse,
            int B, int N, int S, int H, cudaStream_t stream) {
    // Padded f32 rows make every four-element Bias1 group float4-aligned.
    const Bias1Element *bias1_kernel_ptr = bias1_get_f32_padded(Bias1, B, N, S, stream);

    // Get cached f32 Bias2 buffer (alloc once per shape, convert every call)
    Bias2Element *bias2_kernel_ptr = bias2_get_f32(Bias2, B, H, S, stream);

    // Launch main kernel
    dim3 grid(CEIL_DIV(S, kBlockM), H, B * N);
    dim3 block(kNThreads);
    // Shared memory: Q/K share + V = 2 * kBlockN * kHeadDim * sizeof(Element)
    constexpr size_t smem_size = (size(SmemLayoutKV{}) + size(SmemLayoutKV{})) * sizeof(Element);

    if (smem_size >= 32 * 1024) {
        cudaFuncSetAttribute(compute_attention_forward_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        cudaFuncSetAttribute(compute_attention_forward_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    if ((S % kBlockM) == 0 && (S % kBlockN) == 0) {
        compute_attention_forward_kernel<true><<<grid, block, smem_size, stream>>>(
            reinterpret_cast<const Element *>(Q), reinterpret_cast<const Element *>(K),
            reinterpret_cast<const Element *>(V), bias1_kernel_ptr, bias2_kernel_ptr,
            reinterpret_cast<Element *>(O), Lse, B, N, S, H);
        return;
    }
    compute_attention_forward_kernel<false><<<grid, block, smem_size, stream>>>(
        reinterpret_cast<const Element *>(Q), reinterpret_cast<const Element *>(K),
        reinterpret_cast<const Element *>(V), bias1_kernel_ptr, bias2_kernel_ptr,
        reinterpret_cast<Element *>(O), Lse, B, N, S, H);
}

} // namespace triangle_attention_forward

// 兼容 v1 的 C ABI：传入空 LSE 指针时仅返回 O；非空时保存反向所需状态。
extern "C" int colleague_forward_train(const void *q, const void *k, const void *v, const void *b1,
                                       const void *b2, void *o, void *lse, int b, int n, int s,
                                       int h, void *stream) {
    if (!q || !k || !v || !b1 || !b2 || !o || b <= 0 || n <= 0 || s <= 0 || s > 65535 || h <= 0 ||
        h > 65535 || int64_t(b) * n > 65535)
        return int(cudaErrorInvalidValue);
    triangle_attention_forward::launch(
        static_cast<const __nv_bfloat16 *>(q), static_cast<const __nv_bfloat16 *>(k),
        static_cast<const __nv_bfloat16 *>(v), static_cast<const __nv_bfloat16 *>(b1),
        static_cast<const __nv_bfloat16 *>(b2), static_cast<__nv_bfloat16 *>(o),
        static_cast<float *>(lse), b, n, s, h, static_cast<cudaStream_t>(stream));
    return int(cudaGetLastError());
}
