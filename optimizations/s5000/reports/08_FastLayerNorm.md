# 优化08：普通LayerNorm使用fast LayerNorm

2026-10-16：实现已直接写入原模型函数，不通过适配器替换函数。

## 源码位置

- `protenix/model/triangular/layers.py::OpenFoldLayerNorm.forward`

## 既有性能证据
替换指定OpenFoldLayerNorm，保留5类confidence输出归一化回退，已有融合kernel中的归一化不重复执行。

7rev/851 tokens，N_sample=1，10 cycles、200 steps；predict基准42.374秒，候选39.229秒，吞吐+8.02%。首尾基准波动约0.42%。这是单次筛选，非21 PDB增量。

既有耗时来自2026-10-15的适配器版；源码内集成验证见[迁移验证](../SOURCE_INTEGRATION.md)。本项不新增独立性能收益声明，SDPA与局部展平的收益不能相加。
