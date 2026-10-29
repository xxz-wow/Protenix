"""Fuse BF16 Q/K/V deinterleave, head-major layout, and Q scaling."""
import math
from collections import Counter
import torch
import triton
import triton.language as tl

COUNTS = Counter()

@triton.jit
def _qkv_layout(p, q, k, v, total: tl.constexpr, seq: tl.constexpr,
                scale: tl.constexpr, block: tl.constexpr):
    off = tl.program_id(0) * block + tl.arange(0, block)
    mask = off < total
    row = off // 128
    channel = off % 128
    batch = row // seq
    token = row % seq
    head = channel // 32
    dim = channel % 32
    dst = ((batch * 4 + head) * seq + token) * 32 + dim
    base = row * 512 + channel
    qv = tl.load(p + base, mask=mask).to(tl.float32) * scale
    kv = tl.load(p + base + 128, mask=mask)
    vv = tl.load(p + base + 256, mask=mask)
    tl.store(q + dst, qv, mask=mask)
    tl.store(k + dst, kv, mask=mask)
    tl.store(v + dst, vv, mask=mask)

def qkv_layout(p, block=1024, num_warps=4):
    if not (p.ndim == 3 and p.shape[-1] == 512 and p.dtype == torch.bfloat16
            and p.device.type == "musa" and p.is_contiguous()):
        raise ValueError("qkv_layout expects contiguous MUSA BF16 [B,S,512]")
    b, s, _ = p.shape
    q = torch.empty((b,4,s,32), device=p.device, dtype=p.dtype)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    total = b*s*128
    _qkv_layout[(triton.cdiv(total, block),)](
        p,q,k,v,total,s,1.0/math.sqrt(32.0),block,
        num_warps=num_warps,num_stages=1,enable_fp_fusion=False)
    COUNTS[f"b{block}_w{num_warps}"] += 1
    return q,k,v
