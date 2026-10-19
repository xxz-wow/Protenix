# 当前18项验证

当前模型函数与409.691秒复测版本逐字节一致，直接修改原函数，不使用forward重绑定或适配器。

## 已完成完整推理

21PDB、5 samples、10 cycles、200 steps、seed101；每PDB完整warm后正式一次，仅一轮。正式Featurizing→succeeded累计409.691秒。105个正式结构、含warm210个结构通过样本数、原子数和坐标有限性检查，输入shape与历史21PDB一致。局部展平50400、AdaLN合并投影403200、mask已知全一快速分支47040次，均含warm。

15项＋mask隔离对照420.160秒，原15项429.346秒。日志forward和Job扣warm是不同边界，不能混算。

## 发布前检查

原函数AST调用链、warm输入协议、mask缓存失效、正式计时边界、Python语法和发布源码哈希检查，结果记录在benchmarks/release_validation_20261019.json。新计时函数使用全部三轮日志重算应分别得到429.346、420.160、409.691秒。

执行：python -B -m unittest discover -s optimizations/s5000/tests -v。手动MUSA模块检查见tests/check_musa_modules.py；旧pruned_module_validation仅为历史15项删除版检查，不代表当前恢复版。

## 数值限制

输出完整性不等于结构精度。仅mask版与原15项存在坐标差异；刚性对齐RMSD中位数0.315319 Å、最大13.526403 Å，尚未排除链置换、同版本运行波动等影响。当前18项没有结构精度等价结论，详情见../../docs/S5000推理优化组合复测_20261019.md。
