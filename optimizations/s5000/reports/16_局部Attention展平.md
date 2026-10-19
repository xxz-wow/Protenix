# 优化16：局部Attention展平

直接修改 protenix/model/modules/primitives.py::_local_attention。把局部窗口的前导维展平后一次调用支持的BF16 SDPA，减少原chunk_layer逐块调度；输出恢复原shape。不支持的设备、dtype或shape继续走原chunk路径。

依赖优化08的BF16 SDPA。PROTENIX_DISABLE_MUSA_LOCAL_SDPA_FLATTEN=1可关闭，当前组合默认启用。单独开启却关闭SDPA时，本项不会触发，不能用这一结果推断全开组合无用。

## 21PDB五采样证据

10 cycles、200 steps、5 samples、seed101；MSA/cache/efficient fusion/TF32开启，每PDB先warm再正式一次。完整流程统一为21条正式Featurizing→succeeded累计。

| 配置 | 正式完整流程 |
|---|---:|
| 原15项＋mask检查优化 | 420.160秒 |
| 同时增加本项和优化17 | 409.691秒 |

两项合用吞吐提升2.56%，不能分别记到本项或优化17。两次均使用同一物理GPU，属于不同时段的单轮测量。当前组合local_flatten计数50400次，含warm；FP32→BF16 SDPA调用计数由519240降至147240次，调用数不等同于kernel计时。

历史全关基线下SDPA＋展平依赖组吞吐提升5.25%，是依赖组合结果，不是本项单独收益。原始实验编号09＋10仅用于追溯，与当前编号不同。

见[组合复测](../../../docs/S5000推理优化组合复测_20261019.md)。210份warm/正式CIF输出完整不等于结构精度等价。
