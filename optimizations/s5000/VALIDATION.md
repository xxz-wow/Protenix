# 验证范围与计时口径

## 源码与交付检查

固定开源基线为 `4c355be4553512f72453ecbfb65e69f4c35d1413`，直接修改原模型函数。当前源码和复测脚本的摘要见 `SOURCE_MANIFEST.json`，离线检查结果见 `SOURCE_VALIDATION.json`。

`tests/test_source_delivery.py`检查136个源码文件、13类计算功能与对应说明、禁止交付的路径、复测脚本摘要及Python/Shell语法，不需要GPU。源码检查不等于完整结构精度验证。

## 配套Test2.3与21 PDB复测

- Test2.3使用7r6r、7wux、7pzb；eval21使用配套21 PDB输入。
- 固定v0.5.0、5采样、10 cycles、200 diffusion steps、seed101，MSA/cache/efficient fusion/TF32开启。
- 每套数据新进程，每个PDB完整warm一次后正式一次，各一轮。
- 主指标为正式Featurizing到succeeded完整流程耗时之和，包含特征、预测、后处理和写盘，排除warm与模型加载。
- `formal_log_seconds`是预测日志窗口，不与完整流程混算。
- 正式结构数量应分别为15及105，warm数量相同。检查坐标有限性、置信度字段、clash和优化路径计数。

长Diffusion布局、五采样长Softmax和chunk640适用于Test2.3长例；21 PDB最长964 tokens。13项是计算功能分类，不是13组共用同一基线的独立消融，收益不能相加。

## 验证边界

当前精度策略为模型原生BF16 AMP及显式FP32保护，TF32开启。完整GPU复测需要真实配套输入、MSA、权重和校验文件；缺失或摘要不符时停止。

8ech界面精度待确认项保留。模块逐元素一致和结构输出完整性检查均不能替代完整结构质量评价。
