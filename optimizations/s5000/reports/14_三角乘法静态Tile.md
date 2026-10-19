# 优化14：TriangleMultiplication静态tile v2

修改 protenix/model/musa_kernels/triangle_fusion.py::GatedLayout.forward，在原gate/布局kernel调度位置按shape选择tile，不增加forward适配器。

启用：PROTENIX_ENABLE_MUSA_TRIMUL_TILE_V2=1。仅无梯度推理、n>=328、c=128时选择新配置；其他情况保持32×32、4 warps、1 stage。

| shape条件（按顺序判断） | tile | warps | stages |
|---|---|---:|---:|
| n<450 | 32×64 | 8 | 1 |
| 500<=n<640且n可被4整除 | 32×128 | 8 | 1 |
| n为偶数且n<640 | 128×64 | 8 | n<500为1，否则2 |
| n<900 | 64×64 | 8 | 1 |
| 其余 | 32×128 | 8 | 1 |

## 已有性能与验证

21PDB五采样、每PDB完整warm再正式一次。原14项组合对照496.596秒，本项489.387秒；完整Job扣warm吞吐提升1.47%，仅一轮，尚不能证明稳定收益。计数器29120次（含warm）。

原始证据：/workspace/eval21_four_single_ns5_20261016/runs/trimul_tile_v2/summary.json。328/851/964-token算子检查与原32×32布局逐元素一致。该证据不等于完整模型精度验收，也不是移除旧版不分块、局部展平及AdaLN合并投影后的新版性能成绩。

## 当前组合口径

当前发布18项组合的21PDB五采样正式完整流程为409.691秒；逐条Featurizing→succeeded累计、排除warm。该组合成绩不作为本项独立收益。详见[最新组合复测](../../../docs/S5000推理优化组合复测_20261019.md)。
