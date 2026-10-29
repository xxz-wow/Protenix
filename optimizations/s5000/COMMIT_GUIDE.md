# S5000推理优化：源码提交索引

固定开源基线：`4c355be4553512f72453ecbfb65e69f4c35d1413`。当前13项实际计算功能，一项源码及说明一个commit。替代实现、dtype分支、shape分支、打包缓存及tile调优不重复计数。

| 编号 | 优化项 | 源码commit | 说明文件 |
|---|---|---|---|
| 01 | Attention计算内核融合 | `4cbf324` | `reports/01.md` |
| 02 | TriangleMultiplication融合 | `a3ea54a` | `reports/02.md` |
| 03 | Attention QKVG合并投影与权重缓存 | `9f14e0a` | `reports/03.md` |
| 04 | QKV布局与Q缩放融合 | `332e434` | `reports/04.md` |
| 05 | Diffusion静态pair归一化与bias缓存 | `9652caf` | `reports/05.md` |
| 06 | 多采样Diffusion条件共享 | `dd1a1ce` | `reports/06.md` |
| 07 | FastLayerNorm | `99ddc07` | `reports/07.md` |
| 08 | Transition合并投影与SwiGLU融合 | `04bca0f` | `reports/08.md` |
| 09 | 局部Attention批次展平 | `805c1eb` | `reports/09.md` |
| 10 | AdaLN scale与shift合并投影 | `6cee0a7` | `reports/10.md` |
| 11 | 源mask检查缓存 | `99acc09` | `reports/11.md` |
| 12 | FP32 Bias与Softmax融合 | `4f6779c` | `reports/12.md` |
| 13 | 推理分块调度 | `2fe1ad0` | `reports/13.md` |

支持提交不计入优化项数：

- 基础兼容与精度保护：`4144ebd`。
- 复现入口与匹配依赖说明：`6844e48`。
- 当前索引、删增对照与源码清单：本文件所在提交。

共16个commit。旧分支历史被替换，不在旧低精度提交之后追加修补。GitHub保留完整开源项目并直接修改原函数，不发布补丁包或forward适配器。源文件定位见各报告及SOURCE_MANIFEST.json；没有统一的13项独立消融收益。
