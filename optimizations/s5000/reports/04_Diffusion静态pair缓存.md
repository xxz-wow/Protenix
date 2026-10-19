# 优化04：Diffusion静态pair归一化与bias缓存

## 最新21PDB五采样单项证据

2026-10-16原14项全关基线，单独开启本项，新四项关闭；每PDB完整warm一次后正式一次。完整Job扣warm同卡基线1235.82秒，本项1185.99秒，吞吐提升4.20%。每项仅一轮，不能与其他项收益相加，小幅变化尚未证明稳定。该记录早于旧版不分块、局部展平及AdaLN合并投影删除，不是当前18项组合成绩。完整边界见[筛选报告](../../../docs/S5000推理优化筛选与新增四项_20261019.md)。

## 历史实现与筛选记录

2026-10-16：实现已直接写入原模型函数，不通过适配器替换函数。

## 源码位置

- `protenix/model/modules/diffusion.py::DiffusionModule.f_forward`
- `protenix/model/modules/transformer.py::AttentionPairBias.standard_multihead_attention`

## 既有性能证据
复用diffusion多步中不变的pair归一化与bias，按上下文及签名控制缓存。

组合版21 PDB运行通过，没有重新测本项独立五采样收益。显式BF16 bias候选未启用，原pair缓存仍启用。

既有耗时来自2026-10-15的适配器版；源码内集成验证见[迁移验证](../SOURCE_INTEGRATION.md)。本项不新增独立性能收益声明，SDPA与局部展平的收益不能相加。

## 当前组合口径

当前发布18项组合的21PDB五采样正式完整流程为409.691秒；逐条Featurizing→succeeded累计、排除warm。该组合成绩不作为本项独立收益。详见[最新组合复测](../../../docs/S5000推理优化组合复测_20261019.md)。
