# 原函数内集成验证（历史14项版本）

当前2026-10-19版本保留15项，连续编号01～15，新增四项为12～15；以下是迁移时的历史验证，不作为当前15项版本的完整验收。最新状态见[筛选报告](../../docs/S5000推理优化筛选与新增四项_20261019.md)。

2026-10-16。14项推理优化直接进入原模型方法，无forward替换、全局算子重绑定或原函数备份副本。共用Triton计算核和v21 native调用属于算子实现，不是启动安装适配器。

## 优化与原函数对应

| 优化 | 原函数 |
|---|---|
| 01 主干Flash | `triangular/layers.py::_tri_attention` |
| 02 三角乘法输出融合 | `triangular/triangular.py::TriangleMultiplicativeUpdate.forward` |
| 03 Triangle QKVG合并投影 | `triangular/layers.py::Attention.forward` |
| 04 静态pair归一化/bias缓存 | `modules/diffusion.py::DiffusionModule.f_forward`，显式传入`modules/transformer.py`的Attention调用链 |
| 05 多样本条件共享 | `modules/diffusion.py::DiffusionConditioning.forward` |
| 06 Confidence Mutlass | `triangular/triangular.py::TriangleMultiplicativeUpdate.forward` |
| 07 fast LayerNorm | `triangular/layers.py::OpenFoldLayerNorm.forward` |
| 08 标准SDPA | `modules/primitives.py::_attention` |
| 09 Diffusion QKVG合并投影 | `modules/primitives.py::Attention` |
| 10 Transition SwiGLU | `modules/primitives.py::Transition` |
| 11 Confidence BF16/v21 | `triangular/triangular.py::TriangleAttention.forward`，`triangular/layers.py::_tri_attention` |

以上路径均相对于`protenix/model/`。共用核位于`musa_kernels/triangle_fusion.py`和`tri_attention/v21/`，不含安装或替换模型函数的逻辑。

## 已完成验证

- 本地8项协议和调用链检查通过，修改的Python源码语法检查通过。
- S5000隔离目录：`/workspace/source_integrated_20261016/`，未修改既有实验目录。
- 21组算子输出与迁移前快照逐元素一致，最大绝对误差为0。覆盖mask回退、inplace、strided输入、Flash、confidence Mutlass/v21、条件共享、SDPA、Transition、AdaLN及权重变更后的缓存重建。
- 1025-token TriangleAttention不分块路径，两版均实际命中1次，输出有限且完整输出SHA256一致：`a5743dfbaefbe27890f8aefe1849de8498c3dcb9787ada543bc4b0d03ac37deb`。
- 7rev真实模型短推理：851 tokens、1 cycle、2 diffusion steps、5 samples、seed101；每版先warm一次、正式一次；cache、efficient fusion、TF32、预计算MSA开启。两版各10份CIF均通过样本数、原子数和坐标有限值检查。
- 两版命中次数一致：Flash208、Triangle QKVG288、三角乘法输出融合208、条件共享4、pair归一化命中2、pair bias命中48、confidence Mutlass80、v21 80；七个primitive计数也逐项一致。新packed权重构建计数合并了confidence，不能直接与旧计数比较。
- v21在新路径编译后的SHA256仍为`505d8c26d4ae84a113fa623b7ecb1e13d0f3b1412d944d483f31abd5570a4c6b`。

## 最终坐标的重复性边界

算子fixture逐元素一致不等于整个随机推理任务逐元素一致。真实模型短推理的最终CIF并不完全相同，因此额外复跑旧版。五个样本按坐标最小均方误差匹配排列，没有做空间对齐。正式输出结果如下：

| 比较 | 坐标RMS差 | 相对L2差 | 最大坐标差 |
|---|---:|---:|---:|
| 旧版第一次 vs 旧版重复 | 0.059895 | 6.06099e-5 | 0.557007 |
| 旧版第一次 vs 源码内集成 | 0.043491 | 4.40101e-5 | 0.428955 |

两个比较均发生sample1与sample2排序交换，直接按输出文件编号比较会出现很大的假差异。本次迁移差异小于这一次旧版自身的重复波动，但只有单PDB、短扩散测试，不能据此宣称结构准确性等价或逐元素等价。

## 性能数据边界

短推理正式日志为旧版6.60秒、源码版6.58秒，仅作为流程验证记录，不作为优化收益。未重跑10 cycles、200 steps的完整21PDB性能测试。468.35秒、89.94%仍是适配器版历史成绩。

详细摘要：[验证数据](benchmarks/source_integration_validation_20261016.json)。
