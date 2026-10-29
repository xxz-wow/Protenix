"""Independent MUSA gated-projection/layout fusion for triangle multiplication.
Preserves BF16 sigmoid/multiply rounding; no CUDA/cuequivariance source copied.
"""
import os
from protenix.model.musa_kernels.runtime import STATS
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gates_fwd(P,A,B,R:tl.constexpr,N2:tl.constexpr,C:tl.constexpr,
              BR:tl.constexpr,BC:tl.constexpr):
    r=tl.program_id(0)*BR+tl.arange(0,BR)
    c=tl.program_id(1)*BC+tl.arange(0,BC)
    mask=(r[:,None]<R)&(c[None,:]<C)
    off=r[:,None]*(4*C)+c[None,:]
    ag=tl.load(P+off,mask,0).to(tl.float32)
    ap=tl.load(P+off+C,mask,0).to(tl.float32)
    bg=tl.load(P+off+2*C,mask,0).to(tl.float32)
    bp=tl.load(P+off+3*C,mask,0).to(tl.float32)
    sa=(1./(1.+tl.exp(-ag))).to(P.dtype.element_ty).to(tl.float32)
    sb=(1./(1.+tl.exp(-bg))).to(P.dtype.element_ty).to(tl.float32)
    dest=(r[:,None]//N2*C+c[None,:])*N2+r[:,None]%N2
    tl.store(A+dest,sa*ap,mask)
    tl.store(B+dest,sb*bp,mask)


@triton.jit
def gates_bwd(P,DA,DB,DP,R:tl.constexpr,N2:tl.constexpr,C:tl.constexpr,
              BR:tl.constexpr,BC:tl.constexpr):
    r=tl.program_id(0)*BR+tl.arange(0,BR)
    c=tl.program_id(1)*BC+tl.arange(0,BC)
    mask=(r[:,None]<R)&(c[None,:]<C)
    off=r[:,None]*(4*C)+c[None,:]
    ag=tl.load(P+off,mask,0).to(tl.float32)
    ap=tl.load(P+off+C,mask,0).to(tl.float32)
    bg=tl.load(P+off+2*C,mask,0).to(tl.float32)
    bp=tl.load(P+off+3*C,mask,0).to(tl.float32)
    sa=(1./(1.+tl.exp(-ag))).to(P.dtype.element_ty).to(tl.float32)
    sb=(1./(1.+tl.exp(-bg))).to(P.dtype.element_ty).to(tl.float32)
    dest=(r[:,None]//N2*C+c[None,:])*N2+r[:,None]%N2
    da=tl.load(DA+dest,mask,0).to(tl.float32)
    db=tl.load(DB+dest,mask,0).to(tl.float32)
    ga=(da*ap).to(P.dtype.element_ty).to(tl.float32)
    gb=(db*bp).to(P.dtype.element_ty).to(tl.float32)
    tl.store(DP+off,ga*(1-sa)*sa,mask)
    tl.store(DP+off+C,da*sa,mask)
    tl.store(DP+off+2*C,gb*(1-sb)*sb,mask)
    tl.store(DP+off+3*C,db*sb,mask)


class GatedLayout(torch.autograd.Function):
    @staticmethod
    def forward(ctx,p):
        assert p.is_contiguous() and p.dtype==torch.bfloat16
        n=p.shape[-2];c=p.shape[-1]//4
        assert p.shape[-3]==n
        shape=(*p.shape[:-3],c,n,n)
        a,b=[torch.empty(shape,device=p.device,dtype=p.dtype) for _ in range(2)]
        r=p.numel()//(4*c)
        br, bc, warps, stages = 32, 32, 4, 1
        if (not torch.is_grad_enabled() and n >= 328 and c == 128
                and os.getenv("PROTENIX_ENABLE_MUSA_TRIMUL_TILE_V2") == "1"):
            if n < 450:
                br, bc, warps = 32, 64, 8
            elif n < 500:
                br, bc, warps = (128, 64, 8) if n % 2 == 0 else (64, 64, 8)
            elif n < 900:
                br, bc, warps = 32, 128, 16
            else:
                br, bc, warps = 128, 64, 16
            stages = 1
            STATS["trimul_tile_v2_calls"] += 1
        gates_fwd[(triton.cdiv(r,br),triton.cdiv(c,bc))](p,a,b,r,n*n,c,br,bc,num_warps=warps,num_stages=stages)
        ctx.save_for_backward(p)
        return a,b
    @staticmethod
    def backward(ctx,da,db):
        (p,)=ctx.saved_tensors;n=p.shape[-2];c=p.shape[-1]//4;r=p.numel()//(4*c)
        dp=torch.empty_like(p)
        gates_bwd[(triton.cdiv(r,32),triton.cdiv(c,32))](p,da.contiguous(),db.contiguous(),dp,r,n*n,c,32,32,num_warps=4,num_stages=1)
        return dp



@triton.jit
def norm_layout(X,W,B,Y,R:tl.constexpr,C:tl.constexpr,EPS:tl.constexpr):
    r=tl.program_id(0)*32+tl.arange(0,32)
    c=tl.arange(0,C)
    x=tl.load(X+c[:,None]*R+r[None,:],r[None,:]<R,0).to(tl.float32)
    mean=tl.sum(x,0)/C
    v=x-mean[None,:]
    inv=tl.rsqrt(tl.sum(v*v,0)/C+EPS)
    w=tl.load(W+c).to(tl.float32)
    b=tl.load(B+c).to(tl.float32)
    tl.store(Y+r[None,:]*C+c[:,None],(v*inv[None,:])*w[:,None]+b[:,None],r[None,:]<R)

@triton.jit
def gate_write(X,G,Z,Y,E:tl.constexpr,ADD:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<E,0).to(tl.float32)
    g=tl.load(G+i,i<E,0).to(tl.float32)
    s=(1./(1.+tl.exp(-g))).to(G.dtype.element_ty).to(tl.float32)
    out=(x*s).to(X.dtype.element_ty).to(tl.float32)
    if ADD: out=tl.load(Z+i,i<E,0).to(tl.float32)+out
    tl.store(Y+i,out,i<E)

@triton.jit
def output_forward(X, G, Y, E: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x = tl.load(X+i,i<E,0).to(tl.float32)
    g = tl.load(G+i,i<E,0).to(tl.float32)
    s = (1./(1.+tl.exp(-g))).to(G.dtype.element_ty).to(tl.float32)
    tl.store(Y+i,x*s,i<E)


@triton.jit
def output_backward(X, G, DY, DX, DG, E: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x = tl.load(X+i,i<E,0).to(tl.float32)
    g = tl.load(G+i,i<E,0).to(tl.float32)
    dy = tl.load(DY+i,i<E,0).to(tl.float32)
    s = (1./(1.+tl.exp(-g))).to(G.dtype.element_ty).to(tl.float32)
    grad_s = (dy*x).to(G.dtype.element_ty).to(tl.float32)
    tl.store(DX+i,dy*s,i<E)
    tl.store(DG+i,grad_s*(1.-s)*s,i<E)


class OutputGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,g):
        y = torch.empty_like(x)
        output_forward[(triton.cdiv(x.numel(),1024),)](x,g,y,x.numel(),1024,num_warps=4)
        ctx.save_for_backward(x,g)
        return y

    @staticmethod
    def backward(ctx,dy):
        x,g = ctx.saved_tensors
        dx,dg = torch.empty_like(x),torch.empty_like(g)
        output_backward[(triton.cdiv(x.numel(),1024),)](x,g,dy.contiguous(),dx,dg,x.numel(),1024,num_warps=4)
        return dx,dg
