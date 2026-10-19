# 优化13：Attention投影局部BF16

直接修改 protenix/model/modules/primitives.py::Attention.forward。仅在MUSA无梯度推理、self-attention、FP32三维输入、多采样、tokens>=400、Q/K/V通道均为768且带标准bias等条件满足时开启局部BF16上下文；输出恢复输入dtype。不符合条件沿原路径执行。

启用：PROTENIX_ENABLE_MUSA_ATTN_LOCAL_BF16=1。默认关闭。此路径覆盖受控Attention计算，不是全模型强制BF16。

## 已有21PDB性能证据

同原14项组合对照496.596秒，本项单开451.028秒，完整Job扣warm吞吐提升10.10%。21PDB、5采样、10 cycles、200 steps、seed101，每PDB完整warm一次，正式一次；MSA/cache/efficient fusion/TF32开启，其余新增三项关闭。计数器105600次（含warm）。

原始证据：/workspace/eval21_four_single_ns5_20261016/runs/attention_local_bf16/summary.json。单项间复用模型并清理相关缓存；对照独立加载。该记录早于旧版不分块、局部展平及AdaLN合并投影删除，不是删减版的新性能成绩。

## 数值边界

400-token随机模块检查返回FP32、shape正确，相对L2误差约0.00652，不等于结构精度验收。7uxr在本项单开组pLDDT下降1.9521，但它只有354 tokens，未命中本路径；当前不能简单归因于此优化。保留显式开关，尚未认定结构精度等价。

## 当前组合口径

当前发布18项组合的21PDB五采样正式完整流程为409.691秒；逐条Featurizing→succeeded累计、排除warm。该组合成绩不作为本项独立收益。详见[最新组合复测](../../../docs/S5000推理优化组合复测_20261019.md)。
