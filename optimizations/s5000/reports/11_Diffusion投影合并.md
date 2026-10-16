# 优化11：Diffusion Attention QKV与gate合并投影

2026-10-16：实现已直接写入原模型函数，不通过适配器替换函数。

## 源码位置

- `protenix/model/modules/primitives.py::Attention`

## 既有性能证据
本提交只引入对应primitive和路径选择。

7rev单采样predict基准42.374秒，候选42.148秒，吞吐+0.54%。单次微小正变化尚未证明稳定。此处作用于Diffusion而非Triangle。首轮接口失败后修复补测，补测前后基准42.325/42.361秒。

既有耗时来自2026-10-15的适配器版；源码内集成验证见[迁移验证](../SOURCE_INTEGRATION.md)。本项不新增独立性能收益声明，SDPA与局部展平的收益不能相加。
