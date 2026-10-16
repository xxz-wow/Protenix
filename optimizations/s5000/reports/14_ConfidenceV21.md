# 优化14：Confidence Attention使用BF16与v21

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
