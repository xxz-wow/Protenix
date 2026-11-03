"""Five-sample shared-bias specialization; FP32 elementwise arithmetic."""
import torch
import triton
import triton.language as tl
def softmax_bias(scores,bias):
    raise ValueError("Unsupported batch5 layout: caller must retain its generic fallback.")
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
