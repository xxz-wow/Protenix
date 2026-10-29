# 配套校验文件

公开仓库不包含内部镜像地址、运行库摘要或运行环境验证元数据。完整复测需要从匹配交付环境取得以下文件，原样复制到本目录，再将整套 benchmarks 复制到 `/workspace/benchmarks/`：

- `delivery_manifest.json`：代码、扩展、模型、输入和MSA完整性校验。
- `benchmark_manifest.json`：复测文件完整性校验。
- `library_reference.json`：匹配运行库校验。
- `best_practice_reference.json`：必要路径及数量校验。

这些文件缺失时，复测入口会停止。不要用空文件或重新计算任意运行库摘要来绕过验证。运行前根据当前实际源码及匹配依赖核对配套校验文件；数据、权重和原生扩展也需要另行取得。
