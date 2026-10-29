# 实现分段到实际优化项的归并

以下旧编号是上次21个实现分段的追溯编号，不是当前优化项数。当前只采用左列13项编号；不会保留旧分段的独立commit或重复说明。

| 当前实际优化 | 上轮实现分段 | 归并依据 |
|---|---|---|
| 01 Attention计算内核融合 | 01, 13 | 主干原生BF16 AMP的MUSA Flash/SDPA与Confidence的原生TF32 Attention是同一功能的不同精度/模块实现，只计一项；已删除的BF16 Confidence/v21不保留也不计数。 |
| 02 TriangleMultiplication融合 | 02, 09, 15, 17 | 主干与FP32 Confidence三角乘法的投影、门控、layout和输出融合归为一项；打包权重缓存与门控tile调优是该项实现细节，不单独计数。 |
| 03 Attention QKVG合并投影与权重缓存 | 03, 07, 14, 17 | 主干Triangle、Diffusion和FP32 Confidence的Q/K/V/gate合并投影归为一项；参数版本检查、打包缓存及原生AMP权重缓存归入该项。 |
| 04 QKV布局与Q缩放融合 | 18, 19 | Pairformer与Diffusion的QKV布局/缩放融合使用不同shape实现，同一布局优化只计一项；不与QKVG线性投影合并重复描述为同一收益。 |
| 05 Diffusion静态pair归一化与bias缓存 | 04 | 按pair张量及参数版本缓存跨diffusion step不变的归一化和bias；与按采样维度共享条件的下一项处理不同重复计算。 |
| 06 多采样Diffusion条件共享 | 05 | 对同一噪声级别、stride为0的采样条件只计算一次再广播，动态坐标仍独立处理。 |
| 07 FastLayerNorm | 06 | 使用匹配MUSA环境的FastLayerNorm，保留敏感Confidence输出归一化排除规则；不把每个调用模块另算一项。 |
| 08 Transition合并投影与SwiGLU融合 | 08 | 两路投影打包和SwiGLU融合是同一Transition优化的组成部分，权重缓存和分块回退不另计。 |
| 09 局部Attention批次展平 | 10 | 展平局部窗口的批次/头维度以减少小批次调用；当前路径保留受保护FP32，替代旧BF16展平方案，不重复计旧方案。 |
| 10 AdaLN scale与shift合并投影 | 11 | 同一AdaLN条件的scale/shift投影合并，打包缓存和失效检查属于该项，不另算优化。 |
| 11 源mask检查缓存 | 12 | 缓存源mask归约结果，消除重复GPU同步；与参数投影缓存处理不同对象和成本。 |
| 12 FP32 Bias与Softmax融合 | 16, 20 | 选择性FP32 token Attention中的bias/Softmax融合与五采样长序列共享bias是同一项的通用和专用实现；通用FP32回退也归此项，不另计无命中分支。 |
| 13 推理分块调度 | 21, 13 | 统一归档主干长序列chunk640、原生TF32 Confidence不分块和严格FP32回退shape表；它们是同一分块调度策略的条件分支，不各计一项。 |

旧分段17中的乘法权重缓存归入02，Attention权重缓存归入03；它们不再作为独立优化。旧分段13中的TF32 Attention内核归入01，外层不分块选择归入13。两处按代码职责划分，不复制实现或重复分配收益。原基础实现中的通用FP32 Bias/Softmax回退归入12，严格FP32 shape分块表归入13，不额外计数。

旧BF16 Confidence Mutlass/v21、FP32转BF16 SDPA、CT/长Attention局部BF16与pair BF16 bias代码仍保持删除。本轮不恢复旧方案、不改变精度策略、不新增优化。只保留当前实际实现，不为替代方案分配第二项编号。
