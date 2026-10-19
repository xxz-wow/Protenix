# 优化15：BF16 pair bias缓存

修改 protenix/model/modules/transformer.py::AttentionPairBias.standard_multihead_attention，在已有04静态pair缓存的首次bias构建位置转换为BF16并缓存，后续diffusion step复用。不增加第二套缓存。

启用：PROTENIX_ENABLE_MUSA_PAIR_BIAS_BF16_CACHE=1。要求04缓存路径生效、MUSA无梯度推理及efficient fusion开启；默认关闭，其他路径不变。

## 已有21PDB性能证据

原14项组合对照496.596秒，本项490.422秒；完整Job扣warm吞吐提升1.26%。21PDB、5采样，每PDB完整warm再正式一次，正式仅一轮，其余新三项关闭。首次bias构建1008次（含warm）。

原始证据：/workspace/eval21_four_single_ns5_20261016/runs/pair_bias_bf16_cache/summary.json。约1%单轮收益尚不能认定稳定，不应与04收益相加。本记录早于旧版不分块、局部展平及AdaLN合并投影移除，不代表删减组合已复测。

## 数值边界

这是缓存dtype调整，会改变数值舍入。输出完整性通过不等于结构精度验收；7uxr五采样平均pLDDT相对本轮对照下降1.0376，运行非确定性及模型复用等影响尚未完全排除。保留显式启用开关。

## 当前组合口径

当前发布18项组合的21PDB五采样正式完整流程为409.691秒；逐条Featurizing→succeeded累计、排除warm。该组合成绩不作为本项独立收益。详见[最新组合复测](../../../docs/S5000推理优化组合复测_20261019.md)。
