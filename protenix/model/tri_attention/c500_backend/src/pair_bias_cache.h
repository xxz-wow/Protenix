#pragma once
/**
 * pair_bias_cache.h — Bias2 bf16→f32 转换缓存
 *
 * 按 element count (B*H*S*S) 做 workspace cache key:
 *   - 首次遇到该 shape: cudaMalloc 分配 buffer + bf16→f32 转换
 *   - 再次遇到相同 count:
 *       - assume_unchanged=false (默认): 复用 buffer, 重新执行 bf16→f32 转换
 *       - assume_unchanged=true:         复用 buffer, 跳过转换, 直接返回上次的结果
 *
 * 生命周期: buffer 在进程退出时由 CUDA runtime 自动回收, 不主动 free。
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

/**
 * bias1_get_f32_padded — 获取带行 padding 的 Bias1 f32 workspace
 *
 * 输入布局为紧凑 (B,N,S)，输出布局为 (B,N,S_pad)，其中
 * S_pad=round_up(S,4)，保证每行起始地址满足 float4 的 16B 对齐。
 * 默认复用 workspace 但每次重新转换内容。
 */
float *bias1_get_f32_padded(const __nv_bfloat16 *bias1_bf16, int B, int N, int S,
                            cudaStream_t stream, bool assume_unchanged = false);

/**
 * bias2_get_f32 — 获取 Bias2 的 f32 缓存 buffer
 *
 * @param bias2_bf16       输入 bf16 Bias2 device 指针
 * @param B, H, S          shape 参数, buffer size = B*H*S*S*sizeof(float)
 * @param assume_unchanged 若为 true, 假设同 shape 的 bias 值不变, 跳过重复转换
 * @return                 f32 buffer device 指针 (生命周期 = 进程生命周期)
 */
float *bias2_get_f32(const __nv_bfloat16 *bias2_bf16, int B, int H, int S, cudaStream_t stream,
                     bool assume_unchanged = false);
