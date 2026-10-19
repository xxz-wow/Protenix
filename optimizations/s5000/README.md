# S5000 推理优化

面向摩尔线程MTT S5000 / MUSA。18项优化直接进入原模型函数，没有运行时forward适配器。

## 当前18项

| 编号 | 优化 | 报告 |
|---|---|---|
| 01 | 主干TriangleAttention MUSA Flash | [报告](reports/01_主干FlashAttention.md) |
| 02 | TriangleMultiplication融合链 | [报告](reports/02_三角乘法输出融合.md) |
| 03 | Triangle Q/K/V/gate合并投影 | [报告](reports/03_Triangle投影合并.md) |
| 04 | 静态pair归一化与bias缓存 | [报告](reports/04_Diffusion静态pair缓存.md) |
| 05 | 多采样Diffusion条件共享 | [报告](reports/05_多样本条件共享.md) |
| 06 | Confidence Mutlass | [报告](reports/06_ConfidenceMutlass.md) |
| 07 | Fast LayerNorm | [报告](reports/07_FastLayerNorm.md) |
| 08 | BF16 SDPA | [报告](reports/08_标准SDPA.md) |
| 09 | Diffusion Q/K/V/gate合并投影 | [报告](reports/09_Diffusion投影合并.md) |
| 10 | Transition SwiGLU融合 | [报告](reports/10_TransitionSwiGLU.md) |
| 11 | Confidence BF16 v21 | [报告](reports/11_ConfidenceV21.md) |
| 12 | ConditionedTransition局部BF16 | [报告](reports/12_ConditionedTransition局部BF16.md) |
| 13 | 长样本Attention局部BF16 | [报告](reports/13_Attention投影局部BF16.md) |
| 14 | TriangleMultiplication静态tile v2 | [报告](reports/14_三角乘法静态Tile.md) |
| 15 | BF16 pair bias缓存 | [报告](reports/15_BF16PairBias缓存.md) |
| 16 | 局部Attention展平 | [报告](reports/16_局部Attention展平.md) |
| 17 | AdaLN scale/shift合并投影 | [报告](reports/17_AdaLN合并投影.md) |
| 18 | 源mask判断缓存与传递 | [报告](reports/18_Mask检查缓存.md) |

16、17在BF16 SDPA和局部BF16等组合条件下恢复；18消除重复mask归约和同步。旧版仅针对1024<tokens<=1536的BF16不分块没有恢复。历史单项编号不等于当前编号，单项收益不能相加。

## 实测环境与依赖

| 项目 | S5000 |
|---|---|
| GPU | MTT S5000，设备显示80 GiB，单卡推理 |
| 镜像 | registry.mthreads.com/public/protenix_v2:musa |
| 模型 / 配置 | protenix_base_default_v0.5.0，BF16，10 cycles，200 diffusion steps，5 samples，seed101 |
| 数据与开关 | 21PDB，预计算MSA、diffusion cache、efficient fusion、TF32开启 |
| CPU线程 | OMP_NUM_THREADS=4 |

镜像config ID和扩展校验值见[SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json)。不提交native二进制、权重、MSA或数据集。

在匹配镜像中准备依赖：

~~~bash
python optimizations/s5000/prepare_native.py --source-root /app/protenix
~~~

准备器校验并复制镜像已有fast LayerNorm和Mutlass扩展，使用MUSA mcc编译仓库内v21源码。镜像config ID不是可直接docker pull的manifest digest。

## 运行当前组合

~~~bash
export PROTENIX_ROOT_DIR=/path/to/your/model_and_data_root
export MUSA_VISIBLE_DEVICES=0
export PROTENIX_ENABLE_MUSA_CT_LOCAL_BF16=1
export PROTENIX_ENABLE_MUSA_ATTN_LOCAL_BF16=1
export PROTENIX_ENABLE_MUSA_TRIMUL_TILE_V2=1
export PROTENIX_ENABLE_MUSA_PAIR_BIAS_BF16_CACHE=1
export PROTENIX_DISABLE_MUSA_LOCAL_SDPA_FLATTEN=0
export PROTENIX_DISABLE_MUSA_ADAPTIVE_LAYERNORM_FUSION=0
export PROTENIX_DISABLE_SHARED_SAMPLE_CONDITIONING=1

python optimizations/s5000/launch.py \
  --input-json /path/to/your/formal_inputs.json \
  --output-dir /path/to/a/new/run_directory \
  --cycles 10 --steps 200 --samples 5 --seed 101
~~~

12～15默认关闭，需上述四个ENABLE变量显式开启；其他项还取决于设备、推理模式及shape。历史DISABLE开关设1关闭对应路径；不要继承旧实验的禁用变量。16依赖08的BF16 SDPA。17和局部BF16可能改变舍入，保留回退。

输入为唯一name和预计算MSA路径的标准Protenix JSON。启动器插入逐PDB完整warm，已有内容相同的warm/正式配对不重复插入。输出目录须不存在或为空。直接运行同一源码runner/inference.py也可选择相应快路径。

## 当前完整流程性能

| 配置 | 正式完整流程 | 日志Model forward time累计 |
|---|---:|---:|
| 原15项 | 429.346秒 | 404.14秒 |
| 原15项仅加mask优化 | 420.160秒 | 394.76秒 |
| 当前18项组合 | **409.691秒** | 384.22秒 |

每PDB先完整warm一次、再正式一次，只测一轮。**主指标累计正式Featurizing→succeeded，排除warm、启动及succeeded之后清理。**409.827秒是另一边界的Job扣warm，不混入本表。日志forward包含预测、后处理和写盘，不是纯GPU前向。

当前组合相对原15项吞吐提升4.80%；mask隔离对照提升2.19%；在15项＋mask上同时恢复16、17，联合吞吐提升2.56%，不能分别分摊。三次均同一物理GPU、不同时段单轮；未证明小幅差异稳定。

105个正式结构、含warm共210个结构通过样本数、原子数和坐标有限性检查。尚未确认结构精度等价；仅mask版与原15项的导出坐标存在差异，详细限制见[组合复测报告](../../docs/S5000推理优化组合复测_20261019.md)。

## 验证与索引

~~~bash
python -B -m unittest discover -s optimizations/s5000/tests -v
~~~

启动器生成run.log、manifest.json、iteration_timings.json、wall_timing.json、optimization_stats.json、summary.json和report.md；summary中的formal_full_flow_seconds为上述主指标。

[逐项提交与报告](COMMIT_GUIDE.md)、[当前验证](VALIDATION.md)、[历史单项证据](../../docs/S5000推理优化筛选与新增四项_20261019.md)、[源码迁移来源](SOURCE_INTEGRATION.md)。遵循仓库LICENSE与原版权说明。
