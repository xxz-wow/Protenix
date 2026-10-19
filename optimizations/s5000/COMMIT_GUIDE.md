# 提交与报告索引

当前连续编号01～18，每项实现与对应报告位于同一个commit。历史从上游重新构建，不在旧历史末尾追加撤销或修补commit。

提交结构：基础兼容＋18项优化＋运行协议与验证汇总。所有优化直接修改原模型函数。

| 编号 | 主题 | Commit | 报告 |
|---|---|---|---|
| 基础 | 基础：MUSA设备兼容、计数器与native扩展加载 | [4932f61](https://github.com/xxz-wow/Protenix/commit/4932f6145ee9678e4ace5179009727087db0cf5b) | [报告](reports/00_基础兼容.md) |
| 01 | 优化01：在原TriangleAttention函数中使用MUSA Flash | [cf2f577](https://github.com/xxz-wow/Protenix/commit/cf2f577d1c7b73cf7b430f24d956d3baee33a5d2) | [报告](reports/01_主干FlashAttention.md) |
| 02 | 优化02：在原三角乘法forward中融合投影、布局与输出 | [d703d94](https://github.com/xxz-wow/Protenix/commit/d703d9497e18428432938f2ddf7d5a4daa1248fc) | [报告](reports/02_三角乘法输出融合.md) |
| 03 | 优化03：在原Triangle Attention中合并QKV与gate投影 | [0180421](https://github.com/xxz-wow/Protenix/commit/01804210e5911c563a1e7a98aba02c6f96962056) | [报告](reports/03_Triangle投影合并.md) |
| 04 | 优化04：在原Diffusion调用链中缓存静态pair和bias | [cbb02ad](https://github.com/xxz-wow/Protenix/commit/cbb02adecf880566b084b971167a08c3689acc3a) | [报告](reports/04_Diffusion静态pair缓存.md) |
| 05 | 优化05：在原DiffusionConditioning中共享多采样条件 | [d445ebd](https://github.com/xxz-wow/Protenix/commit/d445ebdd8aa0a81efaa99d484cc42e9c1c07fdee) | [报告](reports/05_多样本条件共享.md) |
| 06 | 优化06：在原Confidence三角乘法中调用BF16 Mutlass | [596f0aa](https://github.com/xxz-wow/Protenix/commit/596f0aa574b6f88c68ef13291388f60c24e2a297) | [报告](reports/06_ConfidenceMutlass.md) |
| 07 | 优化07：在原OpenFoldLayerNorm中调用fast LayerNorm | [583ed71](https://github.com/xxz-wow/Protenix/commit/583ed7119846ce04fb52674960d70a7427659e83) | [报告](reports/07_FastLayerNorm.md) |
| 08 | 优化08：在原标准Attention中使用BF16 SDPA | [d66f77e](https://github.com/xxz-wow/Protenix/commit/d66f77e0a449b3bdab65ab80c86d26d992f04fd2) | [报告](reports/08_标准SDPA.md) |
| 09 | 优化09：在原Diffusion Attention中合并QKV与gate投影 | [0c17533](https://github.com/xxz-wow/Protenix/commit/0c175335d0a58281199ca9ca7253a6a55979303a) | [报告](reports/09_Diffusion投影合并.md) |
| 10 | 优化10：在原Transition中合并投影并融合SwiGLU | [0f0ad58](https://github.com/xxz-wow/Protenix/commit/0f0ad583d82f0858961963ee99a3d173744ceb53) | [报告](reports/10_TransitionSwiGLU.md) |
| 11 | 优化11：在原Confidence Attention中使用BF16与v21 | [4b935f6](https://github.com/xxz-wow/Protenix/commit/4b935f6c7f4e3690ac26fee030f61e16f9a160b4) | [报告](reports/11_ConfidenceV21.md) |
| 12 | 优化12：在原ConditionedTransition中加入局部BF16 | [1938ec2](https://github.com/xxz-wow/Protenix/commit/1938ec272087c8b3c46582522acfcc5ddb105a5e) | [报告](reports/12_ConditionedTransition局部BF16.md) |
| 13 | 优化13：在原Attention中加入长样本局部BF16 | [46822e3](https://github.com/xxz-wow/Protenix/commit/46822e3aa2a0256c4aef18a80b3d4715ba758e97) | [报告](reports/13_Attention投影局部BF16.md) |
| 14 | 优化14：按真实shape调整三角乘法gate布局tile | [04cb846](https://github.com/xxz-wow/Protenix/commit/04cb846c1b3a231a58be17d44feb19ba26b801c4) | [报告](reports/14_三角乘法静态Tile.md) |
| 15 | 优化15：在原pair缓存构建位置保存BF16 bias | [3931d20](https://github.com/xxz-wow/Protenix/commit/3931d20bba7733f3dc9244cabf9af39b679498c6) | [报告](reports/15_BF16PairBias缓存.md) |
| 16 | 优化16：在原局部Attention中展平窗口并调用BF16 SDPA | [2bfab12](https://github.com/xxz-wow/Protenix/commit/2bfab12686b6da8a09a79c01b4a6e845db850ede) | [报告](reports/16_局部Attention展平.md) |
| 17 | 优化17：在原AdaLN函数中合并scale与shift投影 | [6e9b4c4](https://github.com/xxz-wow/Protenix/commit/6e9b4c4025db45198d22ca1480a933a145260b49) | [报告](reports/17_AdaLN合并投影.md) |
| 18 | 优化18：缓存源mask判断并消除重复GPU同步 | [8761ec8](https://github.com/xxz-wow/Protenix/commit/8761ec869c95eb8a503aed388e13bbe84bef8ec2) | [报告](reports/18_Mask检查缓存.md) |
| 集成 | 同口径完整流程统计、运行协议与验证证据 | 本提交 | [验证](VALIDATION.md) |

当前18项正式完整流程409.691秒，21PDB五采样，warm不计入。mask单独对照和恢复两项的联合对照见[组合复测](../../docs/S5000推理优化组合复测_20261019.md)。历史单项结果不等于组合内贡献。
