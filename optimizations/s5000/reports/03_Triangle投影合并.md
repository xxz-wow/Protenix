# 优化03：Triangle QKV与gate合并投影

2026-10-16：实现已直接写入原模型函数，不通过适配器替换函数。

## 源码位置

- `protenix/model/triangular/layers.py::Attention.forward`

## 既有性能证据
支持的triangle attention拼接并缓存Q/K/V/gate投影权重，用合并投影替代分散计算。缓存随权重版本、device、dtype失效。

与后续Diffusion投影合并不是同一模块。融合版42输入调用45,360次，包含confidence BF16启用后的调用；无本项同口径独立收益。

既有耗时来自2026-10-15的适配器版；源码内集成验证见[迁移验证](../SOURCE_INTEGRATION.md)。本项不新增独立性能收益声明，SDPA与局部展平的收益不能相加。
