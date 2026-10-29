# 来源与范围

Protenix开源源码基于bytedance/Protenix commit `4c355be4553512f72453ecbfb65e69f4c35d1413`，保留根目录Apache-2.0许可证与原版权信息。模型权重版本v0.5.0与源码commit分别标识。

本分支直接修改原函数并保留必要算子源码；不存在forward适配器。runtime_audit仅用于状态和路径计数采集。FastLayerNorm的完整MUSA构建源不在仓库中，运行依赖边界见native-dependencies.md。

不包含数据、权重、MSA、原生二进制、内部镜像地址、运行库摘要和内部运行证据。已发布交付镜像未在此次同步中重新打包；源代码清理不代表旧镜像内物理文件已删除。
