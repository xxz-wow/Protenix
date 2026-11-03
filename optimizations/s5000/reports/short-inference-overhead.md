# 短序列推理调用开销优化

## 实现

直接修改原函数，共两个文件：

- `protenix/model/modules/primitives.py`：`Linear.forward`在MUSA、eval、无梯度、外层autocast关闭且输入/参数已经满足目标dtype时直接执行同一个`F.linear`；`_attention`在无梯度且autocast已关闭时省去重复的禁用上下文。
- `protenix/model/layer_norm/layer_norm.py`：`FusedLayerNorm.forward`在MUSA、eval、无梯度条件下直接调用原来的`fast_layer_norm_cuda_v2`前向，保留必要的contiguous和参数dtype转换。训练或有梯度时继续调用原autograd Function。

默认启用。`PROTENIX_SHORT_AMP_FASTPATH=0`和`PROTENIX_SHORT_LN_DIRECT=0`分别回退对应路径。这些开关只控制本页的调用开销优化。

没有新增FP32转BF16/FP16转换，没有替换GEMM、softmax或LayerNorm的数学内核。保留模型原生AMP边界和显式FP32保护，TF32开启。

## 已完成的性能验证

2026-11-02，MTT S5000 80GB，单卡GPU6，v1.0.0、5采样、200 steps、10 cycles、seed101；MSA/cache/efficient fusion/TF32开启，template关闭。输入为合成序列，不是Test2.3或21 PDB。

下表是同一轮A/B：每个case、每个版本先预热一次再正式一次；完整流程包含特征处理、预测、后处理和写盘，排除预热与模型加载。

| 长度/范围 | 基线完整流程（秒） | 两项优化组合（秒） | 吞吐提升 |
|---|---:|---:|---:|
| 300 | 13.318 | 12.532 | 6.27% |
| 400 | 15.818 | 15.182 | 4.19% |
| 500 | 19.215 | 18.842 | 1.98% |
| 全部12例累计 | 1371.062 | 1368.693 | 0.17% |

收益主要集中在短例。累计0.17%不足以证明全部长度存在稳定收益，不能将短例收益外推到长例。全量无显式预热的另一轮数据见`long-sequence-fixes.md`，不与本表混算。

进入全量测试前，300长度另做三轮确认：基线13.448175秒，仅上下文快路径13.075530秒，组合12.457592秒；该组与上表独立统计。

## 数值与输出检查

- 160组LayerNorm dtype/affine/布局/autocast组合及22组加载后真实参数比较，逐元素一致，最大绝对差异为0。
- 固定输入的Linear和Attention比较逐元素一致；原LayerNorm autograd前向、反向和构造函数保留。
- 全量A/B共120个正式结构，坐标与主要置信度字段有限，原子数正确，clash均为false。
- A/B最大allocated峰值均为67.320 GiB。

上述为已完成的实验记录，本次Git同步只执行离线源码检查，没有重跑GPU。模块一致性与输出完整性不代表完整结构质量评分。
