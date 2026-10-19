# 优化17：AdaLN scale/shift合并投影

直接修改 protenix/model/modules/primitives.py::AdaptiveLayerNorm.forward，将linear_s和linear_nobias_s的权重及bias打包，一次F.linear得到scale和shift。缓存签名检查参数地址、版本、设备和dtype；参数变化后重新打包。PROTENIX_DISABLE_MUSA_ADAPTIVE_LAYERNORM_FUSION=1可关闭，当前组合默认启用。

## 证据与恢复依据

历史原14项全关基线单开本项：同卡1225.529秒→1226.827秒，吞吐-0.11%，不能标注为单项正收益。

当前15项＋mask为420.160秒，恢复局部展平和本项后为409.691秒；统计为21条正式Featurizing→succeeded累计，21PDB、5 samples、10 cycles、200 steps、seed101，各自排除完整warm。两项联合吞吐提升2.56%，不能拆分归因。当前组合adaln计数403200次，包含warm。

选择保留以本次组合对照为依据，不宣称-0.11%的历史单项结果失效，也不把联合收益记作本项独立收益。所有轮次仅一次，仍受运行波动影响。

见[组合复测](../../../docs/S5000推理优化组合复测_20261019.md)。合并投影可能改变舍入，输出完整性不能代替结构精度验证。
