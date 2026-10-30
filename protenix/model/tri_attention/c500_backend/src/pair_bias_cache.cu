/**
 * pair_bias_cache.cu — Bias2 bf16→f32 转换缓存实现
 *
 * 按 element count (B*H*S*S) 做 workspace cache key，相同 count 复用 buffer。
 * 默认每次刷新内容；assume_unchanged=true 时才复用上次的转换结果。
 * 进程退出时由 CUDA runtime 自动回收所有 device memory。
 */

#include "pair_bias_cache.h"
#include <unordered_map>

// ============================================================
// bf16 → f32 转换 kernel
// ============================================================
static __global__ void bf16_to_f32_kernel(const __nv_bfloat16 *__restrict__ src,
                                          float *__restrict__ dst, size_t count) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}

static __global__ void bf16_to_f32_padded_rows_kernel(const __nv_bfloat16 *__restrict__ src,
                                                      float *__restrict__ dst, int rows, int S,
                                                      int S_padded) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)rows * S;
    if (idx < count) {
        size_t row = idx / S;
        int col = (int)(idx - row * S);
        dst[row * S_padded + col] = __bfloat162float(src[idx]);
    }
}

// ============================================================
// Cache 实现
// ============================================================

using Cache = std::unordered_map<size_t, float *>;

static Cache s_cache;
static Cache s_bias1_padded_cache;

template <typename Convert>
static float *get_or_convert(Cache &cache, size_t count, bool assume_unchanged, Convert &&convert) {
    auto it = cache.find(count);
    if (it != cache.end()) {
        if (!assume_unchanged) {
            convert(it->second);
        }
        return it->second;
    }

    float *buf = nullptr;
    cudaMalloc(&buf, count * sizeof(float));
    convert(buf);
    cache.emplace(count, buf);
    return buf;
}

float *bias1_get_f32_padded(const __nv_bfloat16 *bias1_bf16, int B, int N, int S,
                            cudaStream_t stream, bool assume_unchanged) {
    const int rows = B * N;
    const int S_padded = (S + 3) & ~3;
    const size_t workspace_count = (size_t)rows * S_padded;
    return get_or_convert(s_bias1_padded_cache, workspace_count, assume_unchanged, [&](float *buf) {
        constexpr int THREADS = 256;
        const size_t input_count = (size_t)rows * S;
        int blocks = (int)((input_count + THREADS - 1) / THREADS);
        bf16_to_f32_padded_rows_kernel<<<blocks, THREADS, 0, stream>>>(bias1_bf16, buf, rows, S,
                                                                       S_padded);
    });
}

float *bias2_get_f32(const __nv_bfloat16 *bias2_bf16, int B, int H, int S, cudaStream_t stream,
                     bool assume_unchanged) {
    size_t count = (size_t)B * H * S * S;
    return get_or_convert(s_cache, count, assume_unchanged, [&](float *buf) {
        constexpr int THREADS = 256;
        int blocks = (int)((count + THREADS - 1) / THREADS);
        bf16_to_f32_kernel<<<blocks, THREADS, 0, stream>>>(bias2_bf16, buf, count);
    });
}
