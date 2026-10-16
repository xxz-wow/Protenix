# 优化07：Confidence三角乘法使用BF16 Mutlass

2026-10-16：实现已直接写入原模型函数，不通过适配器替换函数。

## 源码位置

- `protenix/model/triangular/triangular.py::TriangleMultiplicativeUpdate.forward`
- `protenix/model/modules/confidence.py::ConfidenceHead.__init__`
- `protenix/model/musa_kernels/mutlass_triangle_bmm/triangle_bmm.py`

## 既有性能证据
仅confidence阶段走BF16 Mutlass pack，主干仍使用输出融合实现。

融合版42输入调用1,680次。二进制不提交，最终脚本按SHA256从匹配镜像复制。无本项单独五采样收益。

既有耗时来自2026-10-15的适配器版；源码内集成验证见[迁移验证](../SOURCE_INTEGRATION.md)。本项不新增独立性能收益声明，SDPA与局部展平的收益不能相加。
