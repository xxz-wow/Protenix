# S5000 推理优化

本目录提供源码内集成版的运行入口、native依赖准备和既有正式结果摘要。运行环境为 **MTT S5000 + torch/torch_musa 2.9.1.post1**，不适用于直接在 CUDA 环境启用。

## 实测软硬件环境

本目录的21 PDB五采样结果使用以下环境，均为单卡推理：

| 项目 | S5000 优化版 |
| --- | --- |
| GPU | 摩尔线程 MTT S5000，设备显示80 GiB |
| 实测容器 | `xxz-protenix-official-seven-test23-gpu7-20261014` |
| S5000 基础镜像 | `registry.mthreads.com/public/protenix_v2:musa` |
| 模型 / 精度 | `protenix_base_default_v0.5.0` / BF16 |
| CPU 线程 | `OMP_NUM_THREADS=4` |
| Attention | 主干 MUSA Flash；confidence BF16＋v21 |
| TriangleMultiplication | 自有输出融合；confidence BF16 Mutlass |


镜像标签本身不能固定所有依赖版本。实测本地镜像 config ID、Torch版本及native扩展SHA256见 [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json)；该文件未记录的Python、驱动和MUSA SDK精确版本不在此推断。

## 当前方案

- 主干 TriangleAttention：MUSA Flash。
- TriangleMultiplication：当前输出融合实现；旧融合实现是回退链的一部分，不重复计算收益。
- Triangle Q/K/V/gate 合并投影、静态 pair 归一化/bias 缓存、多样本 diffusion 条件共享。
- confidence TriangleMultiplication：Mutlass；confidence TriangleAttention：BF16＋v21。
- 指定普通 LayerNorm：fast LayerNorm。
- 标准/局部 attention：BF16 SDPA，局部布局展平后使用同一 SDPA 路径。
- Diffusion Attention Q/K/V/gate 合并投影、Transition SwiGLU、AdaLN scale/shift 投影合并。

全部14项优化直接进入原模型函数，MUSA推理满足各自条件时选择快路径；CPU、训练及不支持的shape保留原函数回退。SDPA与局部布局展平分别列为第09和第10项。
`biasbf16`、`layout` 未启用。历史不分块适配器仍保留，但只对 1024＜tokens≤1536 生效，在本组最长964 tokens的21 PDB中调用为0。

## 准备环境和依赖

已验证环境来自 `registry.mthreads.com/public/protenix_v2:musa`。镜像标签可能变化，核对 [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) 中的软件版本、实测本地镜像 config ID 和扩展 SHA256；config ID 不是可直接用于 docker pull 的镜像 manifest digest。

在匹配的镜像中检出本分支，使用该镜像的 Python：

```bash
python optimizations/s5000/prepare_native.py --source-root /app/protenix
```

此命令：

1. 从已安装的官方镜像复制 fast LayerNorm 和 Mutlass 两个扩展到当前检出目录，先校验 SHA256。
2. 用 MUSA `mcc` 和 `protenix/model/tri_attention/v21/` 源码编译 `libv21_training_fast_p36q32_repro.so`，保留实测编译参数。
3. 记录本地 `native_build.json`。可通过 `MUSA_HOME` 或 `MCC` 指定编译器。

官方镜像扩展的源码/构建工具链不在本目录重新实现；本分支不分发它们的二进制，使用匹配镜像中已有的构建产物。CUDA 的 LayerNorm 自动编译回退不适用于这里的 MUSA 快路径，因此要先准备匹配扩展，不能只安装上游 requirements 后直接运行。

未提交模型权重、MSA、输入蛋白序列、数据集、预测结构、私钥或 native 二进制。自行挂载模型和数据。

## 运行一次正式测试，预热不计入结果

```bash
export PROTENIX_ROOT_DIR=/path/to/your/model_and_data_root
export MUSA_VISIBLE_DEVICES=0
python optimizations/s5000/launch.py \
  --input-json /path/to/your/formal_inputs.json \
  --output-dir /path/to/a/new/run_directory \
  --cycles 10 --steps 200 --samples 5 --seed 101
```

输入是标准 Protenix JSON，包含唯一的 `name` 和已有的预计算 MSA 路径。可传入21条正式输入，启动器自动插入对应 warm；也可传入已按 `case_warm, case` 排列的42条输入，不会再次添加预热。已配对的 warm/正式条目除名字外必须相同。

每个 PDB 是一次完整 warm（10 cycles、200 steps、5 samples）加一次正式执行，正式轮数为1。输出目录必须不存在或为空。不会搜索/复制任何其他用户的数据；输入中指定的 MSA 由调用者准备。

输出包含：

- `run.log`、`manifest.json`、`input_warm_formal.json`。
- `iteration_timings.json`、`wall_timing.json`、`optimization_stats.json`。
- `summary.json`、`report.md` 和 `output/` 中的结构及置信度。

## 计时与结果

以下是迁移前适配器版在2026-10-15完成的历史结果；不能视为源码内集成版新复测成绩。迁移验证见 [SOURCE_INTEGRATION.md](SOURCE_INTEGRATION.md)。

21 PDB，10 cycles、200 steps、5 samples，cache/efficient fusion/TF32开启：

| 指标 | A100 | S5000 |
|---|---:|---:|
| 正式样本 Model forward time 日志累计 | 421.24秒 | 468.35秒 |
| S5000/A100吞吐 | 100% | **89.94%** |
| Job循环扣除warm窗口 | 未取得同边界数据 | 494.49秒 |

日志累计包含 predict、后处理和写盘，排除迭代前特征准备；扣warm后的Job循环还包含正式数据等待等公共开销。两个指标均不包含模型启动/加载。`494.49` 不能直接与 A100 的 `421.24` 对比。

正式105份结构、含warm共210份均通过文件完整性、原子数及坐标有限值检查。尚未完成结构准确性验收。单样本/单次测试结果不等于统计显著性证据。

此前385.85秒为N_sample=1；23.29%来自7rev单采样、全部九个开关的predict阶段筛选，不是当前七个开关的21 PDB五采样提升。详见 [优化与性能报告](../../docs/S5000优化与性能报告_20261015.md) 和 [实测摘要](benchmarks/eval21_nsample5_20261015.json)。

## 源码布局

- `protenix/model/triangular/layers.py`：原Attention投影、Flash/v21分流和普通LayerNorm函数。
- `protenix/model/triangular/triangular.py`：原三角乘法和TriangleAttention forward。
- `protenix/model/modules/diffusion.py`、`transformer.py`：原Diffusion函数内的条件共享及显式pair缓存传递。
- `protenix/model/modules/primitives.py`：原标准/局部Attention、Diffusion投影、Transition和AdaLN函数。
- `protenix/model/modules/confidence.py`：标识confidence所属模块及保留原LayerNorm的输出层，不改写forward。
- `protenix/model/musa_kernels/triangle_fusion.py`：共用Triton计算核，无install或函数替换。
- `protenix/model/tri_attention/v21/`：native源码、构建脚本和直接调用接口。
- `runner/inference.py`：原推理入口及可选的warm/正式计时记录。

`launch.py`只负责测试协议；优化本身不依赖该启动器。直接使用该检出目录的 `runner/inference.py`，选择 `triangle_attention=triattention`、`triangle_multiplicative=torch`、BF16并开启cache和efficient fusion，也会执行对应快路径。`inference_demo.sh`各case参数不一定相同，需要满足这些条件。已删除 `adapters/`、`compat/`、`quick_controls.py`、复制的 `original_primitives.py` 及v21安装适配器。

## 检查

```bash
python -m unittest discover -s optimizations/s5000/tests -v
python optimizations/s5000/launch.py --help
```

源码快照保留原有版权说明，遵循仓库 LICENSE。native扩展依赖对应镜像；本分支提供 v21 源码及其编译脚本。

## 按优化点阅读提交

见[提交与报告索引](COMMIT_GUIDE.md)。每个优化提交同时包含对应实现和报告；基础兼容和组合复测单独提交。
