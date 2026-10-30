"""MACA 上用于有效训练对照的通用 Triton dBias2 配置。

上游自动调优仅比较耗时，不检查数值。真实训练输入及输出预填测试发现
部分多 warp 配置会产生错误梯度。此处只选择上游已有的一个配置，
不修改公式、dtype、舍入或 Python API；所有 shape 使用相同配置。
"""
CONFIG = dict(BLOCK_M=32, BLOCK_N=32, num_warps=1, num_stages=2)


def configure_triton_baseline():
    import triton
    from protenix.model.tri_attention.backward import _attention_bwd_dbias2
    _attention_bwd_dbias2.configs = [triton.Config(
        {'BLOCK_M': CONFIG['BLOCK_M'], 'BLOCK_N': CONFIG['BLOCK_N']},
        num_warps=CONFIG['num_warps'], num_stages=CONFIG['num_stages'])]
    return dict(dbias2=CONFIG, scope='All shapes; upstream mathematical kernel unchanged',
                reason='Exclude MACA configurations that fail real-input FP32 reference validation')
