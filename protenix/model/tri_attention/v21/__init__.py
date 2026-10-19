"""Native v21 inference kernel; loaded lazily by _tri_attention."""
import ctypes
from pathlib import Path
import torch
_library = None
_launch = None

def forward(q, k, v, bias1, bias2):
    global _library, _launch
    if _launch is None:
        path = Path(__file__).with_name("libv21_training_fast_p36q32_repro.so")
        _library = ctypes.CDLL(str(path))
        _launch = _library.launch_triangle_attention_fwd_musa_lse
        _launch.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        _launch.restype = None
    q, k, v = [x.contiguous() for x in (q, k, v)]
    bias1, bias2 = bias1.float().contiguous(), bias2.float().contiguous()
    b, n, s, h, d = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((b, n, s, h), device=q.device, dtype=torch.float32)
    stream = torch.musa.current_stream().musa_stream
    _launch(*[x.data_ptr() for x in (q, k, v, bias1, bias2, out, lse)],
            b * n, s, h, d, n, stream)
    return out
