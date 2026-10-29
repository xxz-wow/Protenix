# 原生依赖与构建边界

当前只依赖 FastLayerNorm 与 TF32 Confidence 两个有效原生扩展。Git 不提供二进制、内部镜像地址、运行库摘要或内部运行证据；需从匹配交付环境取得扩展及 `benchmarks/checks/` 中说明的配套校验文件。

TF32 Confidence 源码直接位于 `protenix/model/musa_kernels/tri_attn_musa_tf32_bm128_w8.mu`，Python ABI 位于同目录 `tf32_confidence.py`，加载 `libtri_attn_musa_tf32_bm128_w8.so`。它在显式 TF32 开启时使用 TF32 WMMA，保持 FP32 输入输出。

在匹配 MUSA 5.2、mp_31 工具链中可使用以下编译形式；本次同步没有重新编译，重新构建后必须验证，不能视为已测二进制：

```bash
cd /workspace/Protenix/protenix/model/musa_kernels
/usr/local/musa/bin/mcc -O3 -std=c++17 --cuda-gpu-arch=mp_31 -fPIC -shared -x musa \
  tri_attn_musa_tf32_bm128_w8.mu -o libtri_attn_musa_tf32_bm128_w8.so
```

FastLayerNorm 使用匹配 MUSA 扩展；仓库中的开源 CUDA 参考内核不是该 MUSA 扩展的完整构建源码。FlashAttention、Triton、torch_musa 和 MUSA SDK 由运行环境提供。

先将匹配环境的配套校验文件放入 `optimizations/s5000/benchmarks/checks/`，再核验并复制两个原生扩展：

```bash
python optimizations/s5000/prepare_native.py \
  --source-root /path/to/matching-delivery/Protenix \
  --target-root /workspace/Protenix
```

工具先核验全部依赖再复制；缺失或摘要不符时停止，不修改模型 forward。退役的 Mutlass/v21 专用源码、构建文件及依赖不再保留。
