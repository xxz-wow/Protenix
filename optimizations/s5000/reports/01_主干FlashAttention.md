# 优化01：主干TriangleAttention使用MUSA Flash

2026-10-16：实现已直接写入原模型函数，不通过适配器替换函数。

## 源码位置

- `protenix/model/triangular/layers.py::_tri_attention`

## 既有性能证据
支持的BF16主干triangle attention使用MUSA Flash，保留dtype、shape和mask回退。

已有21 PDB五采样融合版42个输入共调用43,680次，包含warm；未取得本项单独的完整任务收益。confidence的v21路径在另一提交加入。

既有耗时来自2026-10-15的适配器版；源码内集成验证见[迁移验证](../SOURCE_INTEGRATION.md)。本项不新增独立性能收益声明，SDPA与局部展平的收益不能相加。
