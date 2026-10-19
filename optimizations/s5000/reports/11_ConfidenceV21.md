# 优化11：Confidence Attention使用BF16与v21

## 最新21PDB五采样单项证据

2026-10-16原14项全关基线，单独开启本项，新四项关闭；每PDB完整warm一次后正式一次。完整Job扣warm同卡基线1225.53秒，本项1203.56秒，吞吐提升1.83%。每项仅一轮，不能与其他项收益相加，小幅变化尚未证明稳定。该记录早于旧版不分块、局部展平及AdaLN合并投影删除，不是当前18项组合成绩。完整边界见[筛选报告](../../../docs/S5000推理优化筛选与新增四项_20261019.md)。

## 历史实现与筛选记录

2026-10-16：实现已直接写入原模型函数，不通过适配器替换函数。

## 源码位置

- `protenix/model/triangular/triangular.py::TriangleAttention.forward`
- `protenix/model/triangular/layers.py::_tri_attention`
- `protenix/model/tri_attention/v21/__init__.py`
- `protenix/model/tri_attention/v21/tri_attn_musa_v21_training.mu`

## 既有性能证据
confidence attention进入BF16并使用v21 native，主干仍走MUSA Flash；同一attention不同时执行两套。

7rev单采样predict41.300秒，相对基准吞吐+2.60%。42个组合输入native调用1,680次且无v21 fallback。native源码重建SHA256与原实测库一致，源码及准确构建开关随本提交提供。

既有耗时来自2026-10-15的适配器版；源码内集成验证见[迁移验证](../SOURCE_INTEGRATION.md)。本项不新增独立性能收益声明，SDPA与局部展平的收益不能相加。

## 当前组合口径

当前发布18项组合的21PDB五采样正式完整流程为409.691秒；逐条Featurizing→succeeded累计、排除warm。该组合成绩不作为本项独立收益。详见[最新组合复测](../../../docs/S5000推理优化组合复测_20261019.md)。
