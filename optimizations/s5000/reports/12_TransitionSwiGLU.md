# 优化12：Transition合并投影与SwiGLU融合

2026-10-16：实现已直接写入原模型函数，不通过适配器替换函数。

## 源码位置

- `protenix/model/modules/primitives.py::Transition`

## 既有性能证据
本提交只引入对应primitive和路径选择。

7rev单采样predict基准42.374秒，候选41.279秒，吞吐+2.65%。单次微小正变化尚未证明稳定。

既有耗时来自2026-10-15的适配器版；源码内集成验证见[迁移验证](../SOURCE_INTEGRATION.md)。本项不新增独立性能收益声明，SDPA与局部展平的收益不能相加。
