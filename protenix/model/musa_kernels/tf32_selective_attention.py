"""Measured LY selective FP32 attention kernels; original tensor precision retained."""
import os
from collections import Counter
import torch
import triton
import triton.language as tl
COUNTS=Counter()
@triton.jit
def _bias_softmax(scores, bias, output, width: tl.constexpr,
                  d1: tl.constexpr, d2: tl.constexpr, d3: tl.constexpr,
                  s0: tl.constexpr, s1: tl.constexpr, s2: tl.constexpr,
                  s3: tl.constexpr, s4: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    value = tl.load(scores + row * width + col, col < width, other=-float("inf")).to(tl.float32)
    offset = ((row % d3) * s3 + ((row // d3) % d2) * s2
              + ((row // (d3 * d2)) % d1) * s1 + (row // (d3 * d2 * d1)) * s0)
    value += tl.load(bias + offset + col * s4, col < width, other=0.0).to(tl.float32)
    numerator = tl.exp(value - tl.max(value, axis=0))
    tl.store(output + row * width + col, numerator / tl.sum(numerator, axis=0), col < width)

@triton.jit
def _bias_softmax_batch5(scores, bias, output, heads: tl.constexpr,
                         width: tl.constexpr, block: tl.constexpr):
    hr = tl.program_id(0)
    col = tl.arange(0, block)
    mask = col < width
    shared_bias = tl.load(bias + hr * width + col, mask, other=0.0).to(tl.float32)
    base0 = hr * width
    step = heads * width * width
    x0 = tl.load(scores + base0 + col, mask, other=-float("inf")).to(tl.float32) + shared_bias
    x1 = tl.load(scores + base0 + step + col, mask, other=-float("inf")).to(tl.float32) + shared_bias
    x2 = tl.load(scores + base0 + 2*step + col, mask, other=-float("inf")).to(tl.float32) + shared_bias
    x3 = tl.load(scores + base0 + 3*step + col, mask, other=-float("inf")).to(tl.float32) + shared_bias
    x4 = tl.load(scores + base0 + 4*step + col, mask, other=-float("inf")).to(tl.float32) + shared_bias
    p0=tl.exp(x0-tl.max(x0,axis=0)); p1=tl.exp(x1-tl.max(x1,axis=0))
    p2=tl.exp(x2-tl.max(x2,axis=0)); p3=tl.exp(x3-tl.max(x3,axis=0))
    p4=tl.exp(x4-tl.max(x4,axis=0))
    tl.store(output+base0+col,p0/tl.sum(p0,axis=0),mask)
    tl.store(output+base0+step+col,p1/tl.sum(p1,axis=0),mask)
    tl.store(output+base0+2*step+col,p2/tl.sum(p2,axis=0),mask)
    tl.store(output+base0+3*step+col,p3/tl.sum(p3,axis=0),mask)
    tl.store(output+base0+4*step+col,p4/tl.sum(p4,axis=0),mask)

def softmax_bias(scores, bias, num_warps=4):
    shape = scores.shape
    padded_shape = (1,) * (len(shape) - bias.ndim) + tuple(bias.shape)
    padded_stride = (0,) * (len(shape) - bias.ndim) + bias.stride()
    strides = tuple(0 if dim == 1 else stride for dim, stride in zip(padded_shape, padded_stride))
    dims5 = (1,) * (5 - len(shape)) + tuple(shape)
    strides5 = (0,) * (5 - len(strides)) + strides
    output = torch.empty_like(scores)
    _bias_softmax[(scores.numel() // shape[-1],)](
        scores, bias, output, shape[-1], dims5[1], dims5[2], dims5[3], *strides5,
        block=triton.next_power_of_2(shape[-1]), num_warps=num_warps, enable_fp_fusion=False)
    return output

def softmax_bias_batch5(scores, bias, num_warps=4):
    if not (scores.ndim == 4 and scores.shape[0] == 5 and scores.is_contiguous()
            and bias.is_contiguous()
            and bias.shape == (1, scores.shape[1], scores.shape[2], scores.shape[3])):
        return softmax_bias(scores, bias)
    _, heads, width, _ = scores.shape
    output = torch.empty_like(scores)
    _bias_softmax_batch5[(heads * width,)](
        scores, bias, output, heads, width,
        block=triton.next_power_of_2(width), num_warps=num_warps, enable_fp_fusion=False)
    return output

def fused_core(q, k, v, attn_bias=None, **_):
    scores = q @ k.transpose(-1, -2)
    if (attn_bias is not None and os.getenv("PROTENIX_MUSA_LONG_BATCH5_SOFTMAX") == "1"
            and scores.ndim == 4 and scores.shape[0] == 5 and scores.shape[-1] >= 1024):
        probability = softmax_bias_batch5(scores, attn_bias, num_warps=4)
        COUNTS["long_batch5_softmax"] += 1
    else:
        probability = softmax_bias(scores, attn_bias) if attn_bias is not None else torch.softmax(scores, -1)
    return probability @ v
