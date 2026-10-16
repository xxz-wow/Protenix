# 来源说明

核心 MUSA 兼容源码来自本次实测的官方 Protenix MUSA 镜像快照，保留原有 Apache-2.0 版权声明。源码内融合实现、运行入口和结果记录来自本项目的 S5000 优化实验；v21源码来自项目已有版本。该分支并不把所有镜像兼容改动声明为本轮新增优化。

fast LayerNorm 和 Mutlass 的镜像预编译扩展不在此分发，通过 prepare_native.py 从使用者已安装的匹配环境准备。来源及核验信息见 SOURCE_SNAPSHOT.json。
