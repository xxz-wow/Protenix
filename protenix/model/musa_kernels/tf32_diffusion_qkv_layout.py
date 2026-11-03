"""Fuse FP32 Diffusion Q/K/V layout and Q scaling."""
import math
from collections import Counter
import torch
import triton
import triton.language as tl

COUNTS=Counter()

@triton.jit
def _layout(p,q,k,v,total:tl.constexpr,seq:tl.constexpr,scale:tl.constexpr,block:tl.constexpr):
 off=tl.program_id(0)*block+tl.arange(0,block)
 mask=off<total
 row=off//768
 channel=off%768
 batch=row//seq
 token=row%seq
 head=channel//48
 dim=channel%48
 dst=((batch*16+head)*seq+token)*48+dim
 base=row*3072+channel
 qv=tl.load(p+base,mask=mask).to(tl.float32)*scale
 kv=tl.load(p+base+768,mask=mask)
 vv=tl.load(p+base+1536,mask=mask)
 tl.store(q+dst,qv,mask=mask)
 tl.store(k+dst,kv,mask=mask)
 tl.store(v+dst,vv,mask=mask)

def qkv_layout(p,block=1024,num_warps=4):
 if not (p.ndim==3 and p.shape[-1]==3072 and p.dtype==torch.float32
         and p.device.type=="musa" and p.is_contiguous()):
  raise ValueError("expected contiguous MUSA FP32 [B,S,3072]")
 b,s,_=p.shape
 q=torch.empty((b,16,s,48),device=p.device,dtype=p.dtype)
 k=torch.empty_like(q);v=torch.empty_like(q)
 total=b*s*768
 _layout[(triton.cdiv(total,block),)](p,q,k,v,total,s,1.0/math.sqrt(48.0),block,
  num_warps=num_warps,num_stages=1,enable_fp_fusion=False)
 COUNTS[f"b{block}_w{num_warps}"]+=1
 return q,k,v
