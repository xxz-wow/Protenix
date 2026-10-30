# MetaX C500 TriangleAttention

为 Protenix 增加 `c500` 注意力后端，复用已验证的 v1 原生前向与一阶反向。
本次实现及性能数据对应 Protenix 提交
`4c355be4553512f72453ecbfb65e69f4c35d1413`。

## 构建与启用

在已安装 MACA SDK、cu-bridge 和对应 PyTorch 的 C500 Linux 环境执行：

```bash
MACA_PATH=/opt/maca bash protenix/model/tri_attention/c500_backend/build.sh
```

产物为本目录 `build/libtri_attention_backward.so`。在原训练命令中添加
`--triangle_attention c500`，或把配置的 `triangle_attention` 设为 `c500`。

调用链为：

```text
TriangleAttention.forward
  -> Attention.forward / _prep_qkv
  -> _tri_attention
  -> c500_backend.TriAttentionFunction.apply
  -> colleague_forward_train

loss.backward()
  -> TriAttentionFunction.backward
  -> colleague_backward
```

Q 在原项目 `_prep_qkv` 中缩放一次。原生 Function 直接使用传入的 Q，
保存 Q/K/V、偏置、O 和 log2-LSE，并由 PyTorch autograd 自动调用反向。

## 接口与验证范围

- Q/K/V/O：BF16 `[B,N,S,H,32]`，当前模型验证为 H=4、D=32。
- N/S 是 pair 表征的两条 token 轴，H 是注意力头数。
- Bias1：BF16 `[B,N,1,1,S]` 或 `[B,N,S]`，作为不求梯度的加性 mask。
- Bias2：BF16 `[B,1,H,S,S]` 或 `[B,H,S,S]`，反向返回原形状和 dtype。
- 支持一阶 autograd。已验证单进程单 GPU、默认 CUDA 兼容 stream；
  同一共享库的全局偏置工作区绑定一张 GPU，顺序复用。
- 公共入口与项目 Triton 入口一致：

```python
from protenix.model.tri_attention.c500_backend import TriAttentionFunction

output = TriAttentionFunction.apply(Q, K, V, Bias1, Bias2)
output.backward(doutput)
```

## 文件职责

| 文件 | 职责 |
|---|---|
| `src/triangle_attention_forward.cu` | 原生训练前向和 log2-LSE 保存 |
| `src/triangle_attention_backward.cu` | dQ/dK/dV、偏置梯度、归约、工作区与调度 |
| `src/pair_bias_cache.cu` | 偏置 BF16→FP32 工作区转换，每次刷新内容 |
| `python/api.py` | ctypes C 接口、张量检查和 autograd Function |
| `third_party/mctlass/` | 构建所需的 CuTe/mctlass 头文件及许可证 |
| `build.sh` | 使用 MACA cu-bridge 编译三个单元 |

项目接入还包含两项公共兼容修复：
`protenix/data/pipeline/dataset.py` 的 NumPy 字符串数组转换，
以及 `protenix/utils/torch_utils.py` 的 MACA direct-cdist 启动容量分块。
两条注意力后端在完整训练对比中使用相同修复。

## 测量结果与限制

完整 TriangleAttention 模块前向约 1.11～1.12×，反向约 1.30×；
五个 PDB 的 20 个正式训练 step 平均为 Triton 19.6130 秒、
C500 19.1762 秒，加速比 1.023×。

独立 FP32 注意力参考下，所测输出及各项有效梯度相对 L2 误差均低于 1%；
完整训练 loss 最大相对差异为 2.07%，所有 step 的梯度范数有限。
**整网参数梯度最大相对 L2 差异为 50.7%，未通过严格一致性要求，
按本轮确认口径单独记录，不能宣称整网参数梯度严格等价。**

Triton 对照显式使用经过正确性验证的通用 dBias2 配置：

```python
from protenix.model.tri_attention.c500_backend.validation.triton_compat import (
    configure_triton_baseline,
)

configure_triton_baseline()
```

该设置只用于对照实验，所有 shape 均采用 32×32、1 warp、2 stages；
其余 Triton 内核沿用自动调优。性能统计在完整目标输入集合的 JIT/调优缓存
预热后复测，每个 PDB 仍保持预热 1 步、正式 4 步。

完整表格、输入口径及精度限制见
[Metax 三角注意力性能记录](../../../../docs/Metax三角注意力性能记录.md)。
