# 当前计算优化与源码职责

以下只列当前保留的13类计算功能。模块、dtype、shape分支及配套缓存归入对应功能。

| 编号 | 当前功能 | 实现文件 |
|---|---|---|
| 01 | Attention计算内核融合 | `protenix/model/modules/primitives.py`、`protenix/model/musa_kernels/tf32_confidence.py`、`protenix/model/musa_kernels/tri_attn_musa_tf32_bm128_w8.mu`、`protenix/model/triangular/layers.py` |
| 02 | TriangleMultiplication融合 | `protenix/model/musa_kernels/triangle_fusion.py`、`protenix/model/triangular/triangular.py` |
| 03 | Attention QKVG合并投影与权重缓存 | `protenix/model/modules/primitives.py`、`protenix/model/triangular/layers.py` |
| 04 | QKV布局与Q缩放融合 | `protenix/model/modules/primitives.py`、`protenix/model/musa_kernels/tf32_diffusion_qkv_layout.py`、`protenix/model/musa_kernels/tf32_pairformer_qkv_layout.py`、`protenix/model/triangular/layers.py` |
| 05 | Diffusion静态pair归一化与bias缓存 | `protenix/model/modules/diffusion.py`、`protenix/model/modules/transformer.py` |
| 06 | 多采样Diffusion条件共享 | `protenix/model/modules/diffusion.py` |
| 07 | FastLayerNorm | `protenix/model/triangular/layers.py` |
| 08 | Transition合并投影与SwiGLU融合 | `protenix/model/modules/primitives.py` |
| 09 | 局部Attention批次展平 | `protenix/model/modules/primitives.py` |
| 10 | AdaLN scale与shift合并投影 | `protenix/model/modules/primitives.py` |
| 11 | 源mask检查缓存 | `protenix/model/musa_kernels/runtime.py`、`protenix/model/triangular/triangular.py` |
| 12 | FP32 Bias与Softmax融合 | `protenix/model/modules/primitives.py`、`protenix/model/musa_kernels/fp32_bias_softmax.py`、`protenix/model/musa_kernels/tf32_batch5_softmax.py`、`protenix/model/musa_kernels/tf32_selective_attention.py`、`protenix/model/triangular/layers.py` |
| 13 | 推理分块调度 | `protenix/model/modules/confidence.py`、`protenix/model/protenix.py`、`protenix/model/triangular/layers.py` |

每项说明见 `reports/01.md` 至 `reports/13.md`，提交索引见 `COMMIT_GUIDE.md`。性能验证按数据集和计时边界分别记录，组合收益不重复拆分。
