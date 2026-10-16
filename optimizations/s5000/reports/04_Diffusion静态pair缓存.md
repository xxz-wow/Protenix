# 优化04：Diffusion静态pair归一化与bias缓存

2026-10-16：实现已直接写入原模型函数，不通过适配器替换函数。

## 源码位置

- `protenix/model/modules/diffusion.py::DiffusionModule.f_forward`
- `protenix/model/modules/transformer.py::AttentionPairBias.standard_multihead_attention`

## 既有性能证据
复用diffusion多步中不变的pair归一化与bias，按上下文及签名控制缓存。

组合版21 PDB运行通过，没有重新测本项独立五采样收益。显式BF16 bias候选未启用，原pair缓存仍启用。

既有耗时来自2026-10-15的适配器版；源码内集成验证见[迁移验证](../SOURCE_INTEGRATION.md)。本项不新增独立性能收益声明，SDPA与局部展平的收益不能相加。
