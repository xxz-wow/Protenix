# S5000推理优化：源码提交索引

固定开源基线：`4c355be4553512f72453ecbfb65e69f4c35d1413`。13类计算功能各对应一个源码及说明提交；模块、dtype、shape分支和配套缓存归入对应功能。

| 编号 | 优化项 | 源码commit | 说明文件 |
|---|---|---|---|
| 01 | Attention计算内核融合 | `9b4ab43` | `reports/01.md` |
| 02 | TriangleMultiplication融合 | `23184b7` | `reports/02.md` |
| 03 | Attention QKVG合并投影与权重缓存 | `3aa975d` | `reports/03.md` |
| 04 | QKV布局与Q缩放融合 | `e85d9b8` | `reports/04.md` |
| 05 | Diffusion静态pair归一化与bias缓存 | `9e12441` | `reports/05.md` |
| 06 | 多采样Diffusion条件共享 | `185ac76` | `reports/06.md` |
| 07 | FastLayerNorm | `1ee2caa` | `reports/07.md` |
| 08 | Transition合并投影与SwiGLU融合 | `77bc9b4` | `reports/08.md` |
| 09 | 局部Attention批次展平 | `5d0fad2` | `reports/09.md` |
| 10 | AdaLN scale与shift合并投影 | `ba701e5` | `reports/10.md` |
| 11 | 源mask检查缓存 | `53b2bd2` | `reports/11.md` |
| 12 | FP32 Bias与Softmax融合 | `f10eea9` | `reports/12.md` |
| 13 | 推理分块调度 | `ef2ab25` | `reports/13.md` |

支持提交：

- MUSA兼容与精度保护：`2f6dbb5`。
- Test2.3/21 PDB复现入口与匹配依赖说明：`493b551`。
- 当前源码清单和索引：本文件所在的交付索引提交。

源码文件及验证边界见`SOURCE_MANIFEST.json`和`VALIDATION.md`。组合验证收益不拆分为每项的独立收益。
