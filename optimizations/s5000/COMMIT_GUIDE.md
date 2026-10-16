# 提交与报告索引

14项优化直接修改原模型函数，每项代码和报告对应一个commit。基础兼容和集成验证单独提交，无运行时适配器安装。

| 序号 | 主题 | Commit | 报告 |
|---|---|---|---|
| 0 | 基础：MUSA设备兼容与模型计数器 | [6edbad2](https://github.com/xxz-wow/Protenix/commit/6edbad2e5e4ca5928f9cc920f80428a3cd04fd40) | [报告](reports/00_基础兼容.md) |
| 1 | 优化01：在原TriangleAttention函数中使用MUSA Flash | [dcd39b4](https://github.com/xxz-wow/Protenix/commit/dcd39b421be64c5c1b6240b3efa4503e6b67e621) | [报告](reports/01_主干FlashAttention.md) |
| 2 | 优化02：在原三角乘法forward中融合投影、布局与输出 | [d0a1b1e](https://github.com/xxz-wow/Protenix/commit/d0a1b1e0267caedf664eb5089dbdc9635cf72dee) | [报告](reports/02_三角乘法输出融合.md) |
| 3 | 优化03：在原Triangle Attention中合并QKV与gate投影 | [ec119b9](https://github.com/xxz-wow/Protenix/commit/ec119b93767a48ced183f711162be813970f52fb) | [报告](reports/03_Triangle投影合并.md) |
| 4 | 优化04：在原Diffusion调用链中缓存静态pair和bias | [3b50bfe](https://github.com/xxz-wow/Protenix/commit/3b50bfeac177d127ac9f2657e2777362d11ffd2a) | [报告](reports/04_Diffusion静态pair缓存.md) |
| 5 | 优化05：在原DiffusionConditioning中共享多采样条件 | [2b0b71b](https://github.com/xxz-wow/Protenix/commit/2b0b71b004280d5d0fccb75741aaf5160e742743) | [报告](reports/05_多样本条件共享.md) |
| 6 | 优化06：在原TriangleAttention中按shape取消BF16分块 | [e25b0a1](https://github.com/xxz-wow/Protenix/commit/e25b0a1ad473229395f38472b7adb02e97bb8511) | [报告](reports/06_BF16不分块.md) |
| 7 | 优化07：在原Confidence三角乘法中调用BF16 Mutlass | [f84c0ee](https://github.com/xxz-wow/Protenix/commit/f84c0ee9264d26eabc355474dae47d37e3cb0312) | [报告](reports/07_ConfidenceMutlass.md) |
| 8 | 优化08：在原OpenFoldLayerNorm中调用fast LayerNorm | [eba79bd](https://github.com/xxz-wow/Protenix/commit/eba79bd8545ff415c219ef200bc6fb839ae7ac58) | [报告](reports/08_FastLayerNorm.md) |
| 9 | 优化09：在原标准Attention中使用BF16 SDPA | [602f143](https://github.com/xxz-wow/Protenix/commit/602f143e9194449c8103d90d5287d516fe92a296) | [报告](reports/09_标准SDPA.md) |
| 10 | 优化10：在原局部Attention中展平SDPA布局 | [bae404f](https://github.com/xxz-wow/Protenix/commit/bae404f8d51111c8ab23522ef4034575322e0af3) | [报告](reports/10_局部Attention展平.md) |
| 11 | 优化11：在原Diffusion Attention中合并QKV与gate投影 | [34c8554](https://github.com/xxz-wow/Protenix/commit/34c855485b3758fc505f4a04d2abcf464ee7c7bf) | [报告](reports/11_Diffusion投影合并.md) |
| 12 | 优化12：在原Transition中合并投影并融合SwiGLU | [f4a185d](https://github.com/xxz-wow/Protenix/commit/f4a185d9db54ee42405e9e9a477f7c1ba8398c2e) | [报告](reports/12_TransitionSwiGLU.md) |
| 13 | 优化13：在原AdaLN中合并scale与shift投影 | [eaf60ca](https://github.com/xxz-wow/Protenix/commit/eaf60cab7eb4e629fa52b1f068db291ea7e7b6b3) | [报告](reports/13_AdaLN投影合并.md) |
| 14 | 优化14：在原Confidence Attention中使用BF16与v21 | [402fd6d](https://github.com/xxz-wow/Protenix/commit/402fd6d5e50f449f1d546f43252ca83f800ed6f9) | [报告](reports/14_ConfidenceV21.md) |
| 15 | 原生入口、计时与组合验证 | 本提交 | [报告](reports/15_组合复测与集成.md) |

SDPA与局部Attention展平已拆为第09和第10项。历史BF16不分块保留为第06项，本组21PDB最长964 tokens不会命中；另用1025-token形状验证该分支。

完整迁移验证和数值边界见[SOURCE_INTEGRATION.md](SOURCE_INTEGRATION.md)。2026-10-15完整任务性能是历史适配器版成绩，本次未重跑完整21PDB。
