"""FP32 bias+softmax only. QK and PV remain native IEEE-FP32 GEMM."""
import torch
import triton
import triton.language as tl
from .runtime import COUNTS

@triton.jit
def _softmax(X,B0,B1,Y,Q:tl.constexpr,H:tl.constexpr,N:tl.constexpr,
             A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,A3:tl.constexpr,
             C0:tl.constexpr,C1:tl.constexpr,C2:tl.constexpr,C3:tl.constexpr,
             HAS0:tl.constexpr,HAS1:tl.constexpr,BLOCK:tl.constexpr):
 row=tl.program_id(0);j=tl.arange(0,BLOCK)
 qr=row%Q;hr=(row//Q)%H;br=row//(Q*H)
 x=tl.load(X+row*N+j,mask=j<N,other=-float('inf'))
 if HAS0:x=x+tl.load(B0+br*A0+hr*A1+qr*A2+j*A3,mask=j<N,other=0)
 if HAS1:x=x+tl.load(B1+br*C0+hr*C1+qr*C2+j*C3,mask=j<N,other=0)
 e=tl.exp(x-tl.max(x,0));p=e/tl.sum(e,0)
 tl.store(Y+row*N+j,p,mask=j<N)

def bias_softmax(x,biases):
 if (x.dtype!=torch.float32 or x.device.type!='musa' or x.ndim not in (3,4)
     or not x.is_contiguous() or not 16<=x.shape[-1]<=2048 or len(biases)>2
     or any(b.dtype!=torch.float32 or b.device!=x.device for b in biases)):return None
 shape=(1,)*(4-x.ndim)+tuple(x.shape);b,h,q,n=shape
 expanded=[]
 for bias in biases:
  z=bias.expand(x.shape)
  expanded.append(z.reshape((1,)*(4-z.ndim)+tuple(z.shape)) if z.ndim<4 else z)
 while len(expanded)<2:expanded.append(None)
 ptr=[z if z is not None else x for z in expanded]
 strides=[tuple(z.stride()) if z is not None else (0,0,0,0) for z in expanded]
 out=torch.empty_like(x);block=triton.next_power_of_2(n)
 _softmax[(b*h*q,)](x,*ptr,out,q,h,n,*strides[0],*strides[1],expanded[0] is not None,expanded[1] is not None,block,num_warps=4 if block<=1024 else 8,enable_fp_fusion=False)
 COUNTS['fp32_bias_softmax_calls']+=1
 return out
