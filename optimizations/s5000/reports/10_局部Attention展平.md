# 优化10：局部Attention布局展平

2026-10-16：实现已直接写入原模型函数，不通过适配器替换函数。

## 源码位置

- `protenix/model/modules/primitives.py::_local_attention`

## 既有性能证据
标准attention使用BF16 SDPA，含FP32转BF16分支；局部5维展平到4维后使用同一路径，两者配套列一项。

7rev单采样predict基准42.374秒：仅SDPA40.632秒，+4.29%；SDPA加局部展平40.679秒，+4.17%。局部展平没有在仅SDPA基础上继续提速，两行收益不能相加。

既有耗时来自2026-10-15的适配器版；源码内集成验证见[迁移验证](../SOURCE_INTEGRATION.md)。本项不新增独立性能收益声明，SDPA与局部展平的收益不能相加。
