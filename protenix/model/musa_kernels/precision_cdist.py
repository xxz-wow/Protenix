"""Dynamic-N FP32 coordinate distances; adapted from the referenced training fix."""
import torch
import triton
import triton.language as tl
from protenix.model.musa_kernels.runtime import STATS

@triton.jit(do_not_specialize=["N"])
def direct_kernel(X,Y,N,BM:tl.constexpr,BN:tl.constexpr):
    i=tl.program_id(0)*BM+tl.arange(0,BM)
    j=tl.program_id(1)*BN+tl.arange(0,BN)
    batch=tl.program_id(2);base=X+batch*N*3
    x0=tl.load(base+i*3,i<N,0);x1=tl.load(base+i*3+1,i<N,0);x2=tl.load(base+i*3+2,i<N,0)
    y0=tl.load(base+j*3,j<N,0);y1=tl.load(base+j*3+1,j<N,0);y2=tl.load(base+j*3+2,j<N,0)
    d0=x0[:,None]-y0[None,:];d1=x1[:,None]-y1[None,:];d2=x2[:,None]-y2[None,:]
    dist=tl.sqrt((d0*d0+d1*d1)+d2*d2)
    tl.store(Y+batch*N*N+i[:,None]*N+j[None,:],dist,(i[:,None]<N)&(j[None,:]<N))

def cdist(a,b=None,p=2,compute_mode="donot_use_mm_for_euclid_dist"):
    eligible = ((b is None or b is a) and p == 2 and a.device.type=="musa"
        and a.dtype==torch.float32 and a.ndim>=2 and a.shape[-1]==3
        and a.shape[-2]>0 and a.numel()>0 and a.is_contiguous() and not a.requires_grad)
    if eligible:
        n=a.shape[-2];batch=a.numel()//(n*3)
        out=torch.empty((*a.shape[:-2],n,n),device=a.device,dtype=torch.float32)
        direct_kernel[(triton.cdiv(n,32),triton.cdiv(n,64),batch)](a,out,n,32,64,num_warps=4,enable_fp_fusion=False)
        STATS["precision_cdist_calls"]+=1
        return out
    STATS["precision_cdist_native"]+=1
    return torch.cdist(a,a if b is None else b,p=p,compute_mode="donot_use_mm_for_euclid_dist")
