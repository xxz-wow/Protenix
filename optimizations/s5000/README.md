# Protenix S5000 推理优化

本分支基于 bytedance/Protenix 开源 commit `4c355be4553512f72453ecbfb65e69f4c35d1413`，直接在原模型函数中实现13项实际优化。检出后即是完整源码，不需要应用补丁，也没有 forward 适配器。

一项优化与对应说明一个 commit；兼容、复现入口、交付索引分别为支持提交，共16个 commit。逐项源码与提交见 `COMMIT_GUIDE.md`，删除、替代和归并关系见 `S5000推理优化删增与归并对照报告_20261029.md`。

## 环境与精度

| 项目 | S5000运行环境 |
|---|---|
| GPU | MTT S5000 80GB，单卡 |
| Python / PyTorch | 3.10.12 / 2.9.1.post1 |
| torch_musa / MUSA | 2.9.1.post1 / 5.2 |
| Triton / FlashAttention | 3.2.0 / 2.6.3（MUSA） |
| 模型 | `protenix_base_default_v0.5.0` |
| 精度 | 保留模型原生BF16 AMP及显式FP32保护，TF32开启 |
| CPU线程 / LayerNorm | OMP_NUM_THREADS=4 / fast_layernorm |

已经移除额外FP32转BF16、Confidence BF16 Mutlass/v21、CT/长Attention局部BF16及pair BF16 bias捷径。FP32输入输出不代表TF32关闭后的严格IEEE FP32计算。

Git只提供源码、必要算子源码、说明和复测脚本，不包含权重、数据、MSA、二进制、内部镜像地址、运行库摘要或内部运行证据。FastLayerNorm及TF32 Confidence两个原生扩展需要匹配交付环境，详见 `native-dependencies.md`。

## 准备复测

复测入口沿用最佳实践的固定目录及校验，不下载或自动生成客户数据。先按 `benchmarks/checks/README.md` 取得匹配校验文件，并准备数据和原生依赖。不能以空校验文件绕过检查。

```text
/workspace/
  Protenix/    # 当前分支源码及匹配的两个原生扩展
  data/        # checkpoint、公共化学数据、test23/eval21输入与MSA
  benchmarks/  # 本分支的复测脚本及外部提供的checks校验文件
  output/      # 运行产物
```

在已把当前源码放入 `/workspace/Protenix` 的环境中，准备匹配文件后复制脚本：

```bash
cd /workspace/Protenix
cp -a optimizations/s5000/benchmarks /workspace/benchmarks
export PROTENIX_ROOT_DIR=/workspace/data
export PYTHONPATH=/workspace/Protenix
python -B /workspace/benchmarks/verify_installation.py
```

`verify_installation.py`还检查交付环境的目录隔离与符号链接，因此需要匹配的运行布局；它不是只检查Python依赖的通用安装器。若只是查看源码，可以直接运行下方离线源码检查。

## Test2.3：3 PDB

7r6r、7wux、7pzb；固定N_sample=5。默认GPU6，先确认空闲。输出目录必须为空。

```bash
PROTENIX_GPU=6 TEST23_OUTPUT_ROOT=/workspace/output/test23_new \
  bash /workspace/benchmarks/run_test23.sh
python -B /workspace/benchmarks/summary.py /workspace/output/test23_new/result.json
```

## recentPDB eval：21 PDB

使用配套21例输入与MSA；固定N_sample=5。

```bash
PROTENIX_GPU=6 EVAL21_OUTPUT_ROOT=/workspace/output/eval21_new \
  bash /workspace/benchmarks/run_eval21.sh
python -B /workspace/benchmarks/summary.py /workspace/output/eval21_new/result.json
```

两组固定v0.5.0、BF16配置、10 cycles、200 diffusion steps、seed101；MSA、diffusion cache、efficient fusion、TF32开启。每个PDB预热一次后正式一次，各跑一轮。当前入口不包含N_sample=1测试。

`TEST23_OUTPUT_ROOT`与`EVAL21_OUTPUT_ROOT`控制输出目录，benchmarks是脚本目录。完整流程看 `formal_full_process_seconds`，预测日志窗口看 `formal_log_seconds`；完整流程为正式Featurizing到succeeded耗时之和，排除warm及模型加载。

## 离线源码检查

```bash
python -B optimizations/s5000/tests/test_source_delivery.py
```

检查136个最终源码摘要、13项提交与说明、退役文件不存在、复测脚本摘要及Python/Shell语法。它不需要GPU，也不验证模型结构精度。

本次仅同步清理版源码和入口，没有重新跑GPU性能或结构精度实验，不把既有组合收益拆分为单项收益。8ech界面精度待确认项保留。验证边界见 `VALIDATION.md`。
