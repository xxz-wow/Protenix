# 优化12：ConditionedTransition局部BF16

直接修改 protenix/model/modules/transformer.py::ConditionedTransitionBlock.forward。在MUSA无梯度推理、c_a=768、c_s=384及多采样shape满足条件时，对AdaLN、投影、SiLU和gate处理使用局部BF16，返回前恢复输入dtype。训练、其他设备或不匹配shape沿原路径运行。

启用：PROTENIX_ENABLE_MUSA_CT_LOCAL_BF16=1。默认关闭，需显式启用；它会改变局部计算精度。

## 已有21PDB性能证据

2026-10-16，21PDB、10 cycles、200 diffusion steps、5 samples、seed101，MSA/cache/efficient fusion/TF32开启。每个PDB先完整warm再正式执行一次；以原14项组合为对照，其余三个新增项关闭。

| Job循环扣warm | 秒 |
|---|---:|
| 同源码原14项组合对照 | 496.596 |
| ＋本项 | 447.144 |

吞吐提升11.06%；计数器201600次，含warm。原始证据：/workspace/eval21_four_single_ns5_20261016/runs/ct_local_bf16/summary.json。

这是删去06/10/13之前的实验，不代表当前删减版单项或组合已复测。四项单测间复用模型、清理打包参数和pair缓存，对照独立加载；每组仅一轮。

## 数值边界

210份warm/正式CIF通过输出完整性检查，不等于结构精度等价。7uxr的五采样平均pLDDT相对本轮对照下降2.9290；尚未完成结构准确率验收或排除运行非确定性影响。

## 当前组合口径

当前发布18项组合的21PDB五采样正式完整流程为409.691秒；逐条Featurizing→succeeded累计、排除warm。该组合成绩不作为本项独立收益。详见[最新组合复测](../../../docs/S5000推理优化组合复测_20261019.md)。
