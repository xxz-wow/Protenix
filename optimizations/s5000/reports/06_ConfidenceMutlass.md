# 优化06：Confidence三角乘法使用BF16 Mutlass

## 最新21PDB五采样单项证据

2026-10-16原14项全关基线，单独开启本项，新四项关闭；每PDB完整warm一次后正式一次。完整Job扣warm同卡基线1240.35秒，本项1227.06秒，吞吐提升1.08%。每项仅一轮，不能与其他项收益相加，小幅变化尚未证明稳定。该记录早于旧版不分块、局部展平及AdaLN合并投影删除，不是当前18项组合成绩。完整边界见[筛选报告](../../../docs/S5000推理优化筛选与新增四项_20261019.md)。

## 历史实现与筛选记录

2026-10-16：实现已直接写入原模型函数，不通过适配器替换函数。

## 源码位置

- `protenix/model/triangular/triangular.py::TriangleMultiplicativeUpdate.forward`
- `protenix/model/modules/confidence.py::ConfidenceHead.__init__`
- `protenix/model/musa_kernels/mutlass_triangle_bmm/triangle_bmm.py`

## 既有性能证据
仅confidence阶段走BF16 Mutlass pack，主干仍使用输出融合实现。

融合版42输入调用1,680次。二进制不提交，最终脚本按SHA256从匹配镜像复制。无本项单独五采样收益。

既有耗时来自2026-10-15的适配器版；源码内集成验证见[迁移验证](../SOURCE_INTEGRATION.md)。本项不新增独立性能收益声明，SDPA与局部展平的收益不能相加。

## 当前组合口径

当前发布18项组合的21PDB五采样正式完整流程为409.691秒；逐条Featurizing→succeeded累计、排除warm。该组合成绩不作为本项独立收益。详见[最新组合复测](../../../docs/S5000推理优化组合复测_20261019.md)。
