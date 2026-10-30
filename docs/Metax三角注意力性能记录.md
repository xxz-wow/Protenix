# MetaX C500 三角注意力接入与性能记录

验证基线提交：`4c355be4553512f72453ecbfb65e69f4c35d1413`。
该分支基于此提交，便于直接查看实测接入改动。
原生 CU 文件与已验证的 v1 SDK 源码一致，项目通过
`--triangle_attention c500` 选择后端。

## 接入改动

| 路径 | 修改 |
|---|---|
| `protenix/model/triangular/layers.py` | 增加 c500 分派，沿用 Q 缩放、转置、reshape、投影与门控 |
| `protenix/model/tri_attention/c500_backend/` | 原生源码、Python autograd 绑定、构建脚本与依赖 |
| `protenix/data/pipeline/dataset.py` | 将 NumPy 字符串拼接输入显式转换为字符串数组 |
| `protenix/utils/torch_utils.py` | 在 MACA 上按输出容量拆分 direct cdist，保留广播、dtype 和 autograd |

## 环境与口径

- MetaX C500 64 GiB；PyTorch 2.10.0+metax3.8.2.2，Triton 3.6.0，
  driver 3.9.6。
- 同一张空闲 GPU、同一环境，两边 TriangleMultiplication 均为 torch。
- checkpoint：`protenix_base_default_v0.5.0.pt`，BF16，crop 上限 512，H=4、D=32。
- 3ew0、5akv、5mqc、5zin、6hvq，各取首个 protein-protein interface；
  每个 PDB 预热 1 步、正式计时 4 步，共 20 个正式 step。
- max cycles=4，diffusion batch=48，chunk=4，Nstep=20，
  `skip_amp.sample_diffusion_training=true`。
- 每步种子为 `424200 + PDB下标*100 + local_step`。
  两边 25 份原始特征哈希和 20 份正式样本元数据一致。
- 5zin 的实际 crop 随固定种子改变，因此先覆盖完整输入集合预热 JIT/自动调优
  缓存，再完整复测 Triton。首次建立缓存的计时不进入下表。
- 模块微测为 BF16 pair 输入 `[1,512,512,128]`；
  前向预热20次/计时50次，反向预热4次/计时10次，取同步墙钟中位数。
- 完整 step 为20个正式样本的平均值，包含输入传输、模型前向、loss、反向、
  梯度裁剪、优化器/清梯度/调度器、metrics/empty_cache 和 EMA。
  CPU 数据构造及额外验证记录在计时区间外。
- 前向、反向分别采集 GPU trace，样本为5mqc、512 tokens、4017 atoms、
  seed 424401；性能采用非 profiler 计时，GPU trace 用于调用确认。

## 完整 TriangleAttention 模块

| 模块 | Triton前向 ms | C500前向 ms | 加速比 | Triton反向 ms | C500反向 ms | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| starting | 6.1527 | 5.5262 | 1.113× | 16.4554 | 12.6187 | 1.304× |
| ending | 6.3042 | 5.6370 | 1.118× | 16.7990 | 12.9570 | 1.297× |

## 完整模型训练

| 范围 | Triton 秒 | C500 秒 | 加速比 |
|---|---:|---:|---:|
| 模型前向 | 6.4528 | 6.3440 | 1.017× |
| 模型反向 | 12.6409 | 12.3278 | 1.025× |
| 完整step | 19.6130 | 19.1762 | 1.023× |

| PDB | 正式输入token数 | Triton平均step 秒 | C500平均step 秒 | 加速比 |
|---|---|---:|---:|---:|
| 3ew0 | 168 | 6.0730 | 5.9593 | 1.019× |
| 5akv | 512 | 23.1259 | 22.5122 | 1.027× |
| 5mqc | 512 | 23.6660 | 23.0254 | 1.028× |
| 5zin | 328/390/412/471 | 21.4722 | 21.3040 | 1.008× |
| 6hvq | 512 | 23.7280 | 23.0801 | 1.028× |

## 精度与整网梯度限制

- 所测单算子输出及 dQ/dK/dV/dBias2 对独立 FP32 参考的整体相对 L2
  误差均低于 1%；真实输入形状包括 `[1,512,512,4,32]` 和
  `[1,168,168,4,32]`，各实现使用同一份归一化 dO。
- Starting/Ending 完整模块的输出与全部可训练梯度整体相对 L2 误差均≤1%。
- 所有训练 step 的全局梯度范数有限；20个 loss 最大相对差异为
  2.0656%。
- **整网参数梯度未通过相对L2≤3%、余弦≥0.999的严格一致性要求。**
  按确认口径如实记录：

| PDB | 参数梯度整体相对L2差异 | 余弦相似度 |
|---|---:|---:|
| 3ew0 | 50.7140% | 0.971832491 |
| 5akv | 6.9940% | 0.998236449 |
| 5mqc | 8.5189% | 0.996366831 |
| 5zin | 12.7221% | 0.991879267 |
| 6hvq | 10.3390% | 0.995040967 |

固定权重、输入及恢复 Python/NumPy/Torch CPU/CUDA 全局随机状态后，
Triton 自重复参数梯度仍差约12.5%。在该诊断中，以完整模型里的 FP32
注意力计算为参考，Triton 两次运行误差为8.7%/11.3%，C500为9.7%。
注意力前向独立重复、基础随机数和 Dropout 状态恢复均能逐位复现。
整网变化的根因尚未确定；以上局部参考检查不能替代严格的整网梯度等价结论。

## MACA 兼容处理

原 Triton dBias2 自动调优只比较耗时。在真实训练保存状态重放中，部分配置
产生错误大梯度或非有限值；对照使用上游已有的通用
32×32、1 warp、2 stages 配置，所有 shape 一致。数学公式、dtype、BF16
舍入及原 `op.py/forward.py/backward.py` 保持原样；
配置入口在 `c500_backend/validation/triton_compat.py`，
结果应理解为正确性修正后的 Triton 对照。

当前 MACA direct-cdist 路径在4095×4095可运行、4096×4096起拒绝启动。
容量适配按每次最多 `2**24-1` 个输出距离拆分批维、行维或列维，
各块执行原直接距离算法再按原顺序拼接。
实际4097/7180原子坐标通过CPU参考；广播用例输出与原直接算法逐位一致，
梯度相对误差≤9.94e-8。该公共修复在两边完全相同。
上游 grid 实现参见
[PyTorch 2.10 DistanceKernel.cu](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/DistanceKernel.cu)。

数值摘要见 [Metax三角注意力性能摘要.json](Metax三角注意力性能摘要.json)。
