"""TF32 Confidence kernel copied from the measured LY BM128/W8 implementation.
Original FP32 input/output tensor types are preserved. Only enabled by explicit TF32 guards.
"""
import ctypes
from collections import Counter
import math
from pathlib import Path
import torch
COUNTS=Counter()
_LIBS={}
_LAUNCHES={}
def _loader(special=False):
    assert not special
    if "v2" not in _LAUNCHES:
        lib=ctypes.CDLL(str(Path(__file__).with_name("libtri_attn_musa_tf32_bm128_w8.so")))
        launch=lib.launch_triangle_attention_tf32_v2
        launch.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*5+[ctypes.c_void_p]
        launch.restype=None
        _LIBS["v2"]=lib
        _LAUNCHES["v2"]=launch
    return _LAUNCHES["v2"]

def forward(q, k, v, biases, mask_all_one=False):
    """Match triangular.layers._tri_attention's input/output layout."""
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)
    original_shape = q.shape

    def reshape(x):
        prefix = len(x.shape[:-3])
        if prefix < 2:
            return x.reshape(*((1,) * (2 - prefix) + x.shape))
        if prefix > 2:
            return x.reshape(*((x.shape[0], -1) + x.shape[-3:]))
        return x

    q, k, v = [reshape(x).contiguous() for x in (q, k, v)]
    bias1, bias2 = [reshape(x).float().contiguous() for x in biases]
    b, n, s, h, d = q.shape
    # The native ABI addresses B1 as [B,N,S] and B2 as [B,H,S,S].
    # Materialize legal broadcast bias layouts before passing raw pointers.
    bias1 = bias1.expand(b, n, 1, 1, s).contiguous()
    bias2 = bias2.expand(b, 1, h, s, s).contiguous()
    out = torch.empty_like(q)
    lse = torch.empty((b, n, s, h), device=q.device, dtype=torch.float32)
    special = False
    _loader(special)(*[x.data_ptr() for x in (q, k, v, bias1, bias2, out, lse)],
                     b * n, s, h, d, n, torch.musa.current_stream().musa_stream)
    COUNTS["calls"] += 1
    COUNTS["v6_calls" if special else "v2_calls"] += 1
    COUNTS[f"tag_{((s + 31) // 32) * 32}"] += 1
    return out.reshape(original_shape)
