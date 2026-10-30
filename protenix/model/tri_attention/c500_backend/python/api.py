"""CuTe C500 attention with first-order autograd; Q is pre-scaled by default.

BF16 Q/K/V: [B,N,S,H,32]. BF16 Bias1: [B,N,S] (or [B,N,1,1,S]);
Bias2: [B,H,S,S] (or [B,1,H,S,S]). Output matches Q's layout/dtype.
The colleague's global bias workspaces require one GPU and sequential streams
per library in a process. Only the default stream has been validated here.
"""
import ctypes
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:
    triton=tl=None

if triton is not None:
    @triton.jit
    def _raw_prepare(Q,B1,B2,QS,B1S,B2S,QC:tl.constexpr,C1:tl.constexpr,C2:tl.constexpr,BLOCK:tl.constexpr):
        x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        q=tl.load(Q+x,x<QC,0).to(tl.float32)*0.17677669529663689
        tl.store(QS+x,q,x<QC)
        tl.store(B1S+x,tl.load(B1+x,x<C1,0),x<C1)
        tl.store(B2S+x,tl.load(B2+x,x<C2,0),x<C2)

    @triton.jit
    def _raw_restore(DQS,BS,DQ,B,QC:tl.constexpr,BC:tl.constexpr,BIAS:tl.constexpr,BLOCK:tl.constexpr):
        x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        dq=tl.load(DQS+x,x<QC,0).to(tl.float32)*0.17677669529663689
        tl.store(DQ+x,dq,x<QC)
        if BIAS:
            db=tl.load(BS+x,x<BC,0).to(tl.bfloat16).to(tl.float32)
            tl.store(B+x,db,x<BC)

ROOT=Path(__file__).resolve().parents[1]/"build"
_DEVICES={}
_DEFAULT=None


class Backend:
    def __init__(self,library=None):
        self.path=str(Path(library or ROOT/"libtri_attention_backward.so").resolve())
        self.library=ctypes.CDLL(self.path)
        self.fwd=self.library.colleague_forward_train
        self.fwd.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*4+[ctypes.c_void_p]
        self.fwd.restype=ctypes.c_int
        self.bwd=self.library.colleague_backward
        self.bwd.argtypes=[ctypes.c_void_p]*16+[ctypes.c_int]*5+[ctypes.c_void_p]
        self.bwd.restype=ctypes.c_int

    def validate(self,q,k,v,b1,b2):
        if q.ndim!=5:raise ValueError("Q must be [B,N,S,H,32]")
        b,n,s,h,d=q.shape
        if min(b,n,s,h)<=0 or d!=32 or b*n>65535 or s>65535 or h>65535:
            raise ValueError("Requires positive B/N/S/H, D=32 and B*N/S/H<=65535")
        for name,x,shape in (("q",q,q.shape),("k",k,q.shape),("v",v,q.shape),
                             ("bias1",b1,(b,n,s)),("bias2",b2,(b,h,s,s))):
            if x.shape!=shape or x.dtype!=torch.bfloat16 or not x.is_contiguous() or x.device!=q.device or not x.is_cuda:
                raise ValueError(f"{name} must be contiguous BF16, shape {tuple(shape)}, on Q's GPU")
        device=q.device.index
        previous=_DEVICES.setdefault(self.path,device)
        if previous!=device:raise RuntimeError("This library's bias cache is bound to one GPU per process")
        return b,n,s,h

    @staticmethod
    def _ptrs(tensors):
        return [ctypes.c_void_p(x.data_ptr()) if x is not None else None for x in tensors]

    def prepare_forward(self,q,k,v,b1,b2,*,out=None,lse=None):
        b,n,s,h=self.validate(q,k,v,b1,b2)
        if out is None:out=torch.empty_like(q)
        if lse is None:lse=torch.empty(b,n,s,h,device=q.device,dtype=torch.float32)
        if out.shape!=q.shape or out.dtype!=q.dtype or out.device!=q.device or not out.is_contiguous():
            raise ValueError("out must be contiguous and match Q")
        if lse.shape!=(b,n,s,h) or lse.dtype!=torch.float32 or lse.device!=q.device or not lse.is_contiguous():
            raise ValueError("lse must be contiguous FP32 [B,N,S,H]")
        if any(out.untyped_storage().data_ptr()==x.untyped_storage().data_ptr() for x in (q,k,v,b1,b2,lse)):
            raise ValueError("Output must not share storage with inputs or LSE")
        with torch.cuda.device(q.device):
            stream=torch.cuda.current_stream(q.device)
            pointers=self._ptrs((q,k,v,b1,b2,out,lse))
            def call():
                error=self.fwd(*pointers,b,n,s,h,ctypes.c_void_p(stream.cuda_stream))
                if error:raise RuntimeError(f"C500 training forward launch error {error}")
            call._tensors=(q,k,v,b1,b2,out,lse)
            for x in call._tensors:x.record_stream(stream)
        return call,out,lse

    def forward(self,q,k,v,b1,b2):
        call,out,lse=self.prepare_forward(q,k,v,b1,b2)
        call()
        return out,lse

    def workspace(self,q,*,need_bias1=True,need_bias2=True,groups=None):
        b,n,s,h,d=q.shape
        if groups is None:
            choose=getattr(self.library,'colleague_backward_group_count',None)
            if choose is not None:
                choose.argtypes=[ctypes.c_int]*4;choose.restype=ctypes.c_int
                with torch.cuda.device(q.device):groups=choose(b,n,s,h)
            else:groups=min(n,8) # Compatibility with older libraries.
        else:groups=int(groups)
        if not 1<=groups<=n:raise ValueError("groups must lie in [1,N]")
        alloc=lambda shape:torch.empty(shape,device=q.device,dtype=torch.float32)
        return dict(dq=torch.empty_like(q),dk=torch.empty_like(q),dv=torch.empty_like(q),
            db1=alloc((b,n,s)) if need_bias1 else None,
            db2=alloc((b,h,s,s)) if need_bias2 else None,
            delta=alloc((b,n,s,h)),db1_head=alloc((b,n,h,s)) if need_bias1 else None,
            db2_partial=alloc((b,groups,h,s,s)) if need_bias2 else None,groups=groups)

    def prepare_backward(self,q,k,v,b1,b2,out,lse,dout,*,workspace=None,need_bias1=True,need_bias2=True):
        b,n,s,h=self.validate(q,k,v,b1,b2)
        for name,x in (("out",out),("dout",dout)):
            if x.shape!=q.shape or x.dtype!=q.dtype or x.device!=q.device or not x.is_contiguous():
                raise ValueError(f"{name} must be contiguous and match Q")
        if lse.shape!=(b,n,s,h) or lse.dtype!=torch.float32 or lse.device!=q.device or not lse.is_contiguous():
            raise ValueError("Invalid LSE tensor")
        w=workspace or self.workspace(q,need_bias1=need_bias1,need_bias2=need_bias2)
        groups=w["groups"]
        specs={"dq":(q.shape,q.dtype),"dk":(q.shape,q.dtype),"dv":(q.shape,q.dtype),
               "delta":((b,n,s,h),torch.float32),"db1":((b,n,s),torch.float32),
               "db1_head":((b,n,h,s),torch.float32),"db2":((b,h,s,s),torch.float32),
               "db2_partial":((b,groups,h,s,s),torch.float32)}
        if not 1<=groups<=n:raise ValueError("Invalid workspace group count")
        for name,(shape,dtype) in specs.items():
            x=w[name]
            if x is None:
                if name not in ("db1","db1_head","db2","db2_partial"):raise ValueError(f"Missing workspace {name}")
            elif x.shape!=shape or x.dtype!=dtype or x.device!=q.device or not x.is_contiguous():
                raise ValueError(f"Invalid workspace {name}")
        if (w['db1'] is None)!=(w['db1_head'] is None) or (w['db2'] is None)!=(w['db2_partial'] is None):
            raise ValueError("Bias outputs and workspaces must be requested together")
        inputs=(q,k,v,b1,b2,out,lse,dout)
        outputs=tuple(w[name] for name in ("dq","dk","dv","db1","db2","delta","db1_head","db2_partial"))
        input_storage={x.untyped_storage().data_ptr() for x in inputs}
        output_storage=[x.untyped_storage().data_ptr() for x in outputs if x is not None]
        if len(set(output_storage))!=len(output_storage) or input_storage.intersection(output_storage):
            raise ValueError("Backward workspaces must be disjoint from each other and all inputs")
        with torch.cuda.device(q.device):
            stream=torch.cuda.current_stream(q.device)
            pointers=self._ptrs((*inputs,*outputs))
            def call():
                error=self.bwd(*pointers,b,n,s,h,groups,ctypes.c_void_p(stream.cuda_stream))
                if error:raise RuntimeError(f"C500 backward launch error {error}")
            call._tensors=(*inputs,*(x for x in outputs if x is not None))
            for x in call._tensors:x.record_stream(stream)
        return call,w

    def backward(self,q,k,v,b1,b2,out,lse,dout,*,need_bias1=True,need_bias2=True):
        call,w=self.prepare_backward(q,k,v,b1,b2,out,lse,dout,
            need_bias1=need_bias1,need_bias2=need_bias2)
        call()
        return w['dq'],w['dk'],w['dv'],w['db1'],w['db2']

    def prepare_backward_raw(self,q,k,v,b1,b2,out,lse,dout,*,workspace=None,
                             need_bias2=True,prepare_inputs=True):
        """Preallocate raw-Q backward: BF16 dQ/dK/dV, rounded FP32 dBias2.

        Bias1 is a mask; both input biases are contiguous FP32. Saved O/LSE
        must come from BF16-rounded Q/sqrt(32) and BF16-rounded biases.
        Reuse the returned workspace for repeated calls. prepare_inputs=False
        requires its qs/b1s/b2s buffers to already contain those conversions.
        This entry requires the optimized candidate library; the prepared-Q
        backward and autograd APIs retain their original contract.
        This API chooses fused/native or compiled-helper adapters from kernel
        capacity and GPU residency. Benchmark code does not select the path.
        """
        raw=getattr(self.library,'colleague_backward_raw_workspace',None)
        if raw is None:raise NotImplementedError('Load the optimized raw-backward library')
        if q.ndim!=5:raise ValueError('Q must be [B,N,S,H,32]')
        b,n,s,h,d=q.shape
        for name,x,shapes in (('bias1',b1,((b,n,s),(b,n,1,1,s))),
                              ('bias2',b2,((b,h,s,s),(b,1,h,s,s)))):
            if tuple(x.shape) not in shapes or x.dtype!=torch.float32 or x.device!=q.device or not x.is_contiguous():
                raise ValueError(f'{name} must be contiguous FP32 on Q\'s GPU with the original bias layout')
        if workspace is None:
            workspace=dict(core=self.workspace(q,need_bias1=False,need_bias2=need_bias2),
                qs=torch.empty_like(q),b1s=torch.empty(b,n,s,device=q.device,dtype=q.dtype),
                b2s=torch.empty(b,h,s,s,device=q.device,dtype=q.dtype),dq=torch.empty_like(q),
                db2=torch.empty_like(b2) if need_bias2 else None)
        w=workspace['core'];qs,b1s,b2s=[workspace[x] for x in ('qs','b1s','b2s')]
        self.validate(q,k,v,b1s,b2s)
        prepared_call,_=self.prepare_backward(qs,k,v,b1s,b2s,out,lse,dout,workspace=w)
        if w['db1'] is not None or w['db1_head'] is not None:
            raise ValueError('Raw backward treats Bias1 as a non-differentiable mask')
        for name,shape,dtype in (('dq',q.shape,q.dtype),('db2',(b,h,s,s),torch.float32)):
            x=workspace[name]
            if name=='db2' and x is None and w['db2'] is None:continue
            shapes=(shape,(b,1,h,s,s)) if name=='db2' else (shape,)
            if x is None or tuple(x.shape) not in shapes or x.dtype!=dtype or x.device!=q.device or not x.is_contiguous():
                raise ValueError(f'Invalid raw workspace {name}')
        if (workspace['db2'] is None)!=(w['db2'] is None):raise ValueError('Bias output and core workspace must agree')
        inputs=(q,k,v,b1,b2,out,lse,dout)
        temporaries=tuple(x for x in w.values() if isinstance(x,torch.Tensor))+(qs,b1s,b2s,workspace['dq'])
        if workspace['db2'] is not None:temporaries+=(workspace['db2'],)
        stores=[x.untyped_storage().data_ptr() for x in temporaries]
        if len(set(stores))!=len(stores) or set(stores).intersection(x.untyped_storage().data_ptr() for x in inputs):
            raise ValueError('Raw workspaces must be disjoint from each other and all inputs')
        raw.argtypes=[ctypes.c_void_p]*21+[ctypes.c_int]*6+[ctypes.c_void_p];raw.restype=ctypes.c_int
        with torch.cuda.device(q.device):
            stream=torch.cuda.current_stream(q.device)
            pointers=self._ptrs((*inputs,workspace['dq'],w['dk'],w['dv'],None,workspace['db2'],
                w['delta'],None,w['db2_partial'],qs,b1s,b2s,w['dq'],w['db2']))
            def call():
                error=raw(*pointers,b,n,s,h,w['groups'],int(prepare_inputs),ctypes.c_void_p(stream.cuda_stream))
                if error:raise RuntimeError(f'C500 raw backward launch error {error}')
            choose=getattr(self.library,'colleague_backward_raw_policy',None)
            if choose is not None:
                choose.argtypes=[ctypes.c_int]*3+[ctypes.c_int64];choose.restype=ctypes.c_int
                path=choose(n,s,w['groups'],q.numel())
            else:path=1
            call._path='fused_small_raw' if path==2 else 'native_raw_workspace_dq_epilogue'
            prepare_raw=self.library.colleague_prepare_raw
            prepare_raw.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int64]*3+[ctypes.c_void_p];prepare_raw.restype=ctypes.c_int
            restore_raw=self.library.colleague_restore_raw
            restore_raw.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int64]*2+[ctypes.c_void_p];restore_raw.restype=ctypes.c_int
            pp=self._ptrs((q,b1,b2,qs,b1s,b2s));rp=self._ptrs((None,w['db2'],None,workspace['db2']))
            def prepare_component():
                error=prepare_raw(*pp,q.numel(),b1.numel(),b2.numel(),ctypes.c_void_p(stream.cuda_stream))
                if error:raise RuntimeError(f'C500 raw preparation error {error}')
            def restore_component():
                if workspace['db2'] is not None:
                    error=restore_raw(*rp,0,b2.numel(),ctypes.c_void_p(stream.cuda_stream))
                    if error:raise RuntimeError(f'C500 bias restoration error {error}')
            call._prepare=prepare_component;call._restore=restore_component
            if path==0 and triton is not None:
                def bind(kernel,grid,values):
                    grid=tuple(grid)+(1,)*(3-len(grid))
                    compiled=kernel[grid](*values,num_warps=4,num_stages=1)
                    return lambda:compiled[grid](*values,stream=stream.cuda_stream)
                prep=bind(_raw_prepare,(triton.cdiv(max(q.numel(),b1.numel(),b2.numel()),1024),),
                    (q,b1,b2,qs,b1s,b2s,q.numel(),b1.numel(),b2.numel(),1024))
                has_bias=workspace['db2'] is not None
                restore=bind(_raw_restore,(triton.cdiv(max(q.numel(),b2.numel() if has_bias else 0),1024),),
                    (w['dq'],w['db2'] if has_bias else lse,workspace['dq'],workspace['db2'] if has_bias else lse,
                     q.numel(),b2.numel(),has_bias,1024))
                def call():
                    if prepare_inputs:prep()
                    prepared_call();restore()
                call._path='compiled_raw_helpers';call._prepare=prep;call._restore=restore
            call._tensors=inputs+temporaries
            for x in call._tensors:x.record_stream(stream)
        return call,workspace


def backend():
    global _DEFAULT
    if _DEFAULT is None:_DEFAULT=Backend()
    return _DEFAULT


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,b1,b2):
        op=backend()
        out,lse=op.forward(q,k,v,b1,b2)
        ctx.save_for_backward(q,k,v,b1,b2,out,lse)
        ctx.backend=op
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,dout):
        q,k,v,b1,b2,out,lse=ctx.saved_tensors
        needs=ctx.needs_input_grad
        gradients=ctx.backend.backward(q,k,v,b1,b2,out,lse,dout.contiguous(),
            need_bias1=needs[3],need_bias2=needs[4])
        return tuple(g.to(x.dtype) if need and g is not None else None
                     for g,x,need in zip(gradients,(q,k,v,b1,b2),needs))


def attention(q,k,v,bias1,bias2):
    """First-order attention. Q must already contain its 1/sqrt(D) scaling."""
    if q.ndim!=5:raise ValueError("Q must be [B,N,S,H,32]")
    b,n,s,h,d=q.shape
    if tuple(bias1.shape) not in ((b,n,s),(b,n,1,1,s)):
        raise ValueError("Bias1 must be [B,N,S] or [B,N,1,1,S]")
    if tuple(bias2.shape) not in ((b,h,s,s),(b,1,h,s,s)):
        raise ValueError("Bias2 must be [B,H,S,S] or [B,1,H,S,S]")
    return _Attention.apply(q.contiguous(),k.contiguous(),v.contiguous(),
                            bias1.reshape(b,n,s).contiguous(),bias2.reshape(b,h,s,s).contiguous())


def attention_raw(q,k,v,bias1,bias2):
    """Raw-Q convenience path: PyTorch records scaling/cast for its chain rule."""
    scaled=(q.float()*0.17677669529663689).to(q.dtype)
    return attention(scaled,k,v,bias1,bias2)


class TriAttentionFunction(torch.autograd.Function):
    """替换 Triton 同名入口；Q/K/V/O 为 [B,N,S,H,32]，使用 BF16。

    Q 按传入值直接参与计算，内部不缩放。Bias1 是加性 mask，不求梯度。
    当前原生范围为 BF16 Q/K/V/Bias1/Bias2；不隐式降低 FP32 bias 精度。
    deterministic=True/False 均使用同一个无浮点原子加的固定顺序实现。
    """
    @staticmethod
    def forward(ctx, Q, K, V, Bias1, Bias2, deterministic=False):
        if Q.ndim != 5:
            raise ValueError('Q must be [B,N,S,H,32]')
        b, n, s, h, d = Q.shape
        if tuple(Bias1.shape) not in ((b,n,s), (b,n,1,1,s)):
            raise ValueError('Bias1 must be [B,N,S] or [B,N,1,1,S]')
        if tuple(Bias2.shape) not in ((b,h,s,s), (b,1,h,s,s)):
            raise ValueError('Bias2 must be [B,H,S,S] or [B,1,H,S,S]')
        if Bias1.dtype != torch.bfloat16 or Bias2.dtype != torch.bfloat16:
            raise ValueError('This verified native interface requires BF16 Bias1/Bias2; FP32 biases are not silently rounded')
        q, k, v = (x.contiguous() for x in (Q,K,V))
        b1 = Bias1.reshape(b,n,s).contiguous()
        b2 = Bias2.reshape(b,h,s,s).contiguous()
        op = backend()
        out, lse = op.forward(q,k,v,b1,b2)
        ctx.save_for_backward(q,k,v,b1,b2,out,lse)
        ctx.backend = op
        ctx.bias2_shape = Bias2.shape
        ctx.deterministic = bool(deterministic)
        ctx.input_count = len(ctx.needs_input_grad)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, DO):
        q,k,v,b1,b2,out,lse = ctx.saved_tensors
        # Bias1 在原 Triton 注释中是 mask；不返回其未初始化的 DBias1。
        dq,dk,dv,_,db2 = ctx.backend.backward(
            q,k,v,b1,b2,out,lse,DO.contiguous(),
            need_bias1=False,need_bias2=ctx.needs_input_grad[4])
        gradients = tuple(g if need else None for g,need in zip(
            (dq,dk,dv),ctx.needs_input_grad[:3]))
        if db2 is not None:
            db2 = db2.to(b2.dtype).reshape(ctx.bias2_shape)
        gradients += (None,db2)
        # apply 可省略有默认值的 deterministic，也可显式传入第六个参数。
        return gradients + (None,) if ctx.input_count == 6 else gradients


class TriAttention(torch.nn.Module):
    """与原 Triton 模块相同的构造和五张量 forward 调用。"""
    def __init__(self, deterministic=False):
        super().__init__()
        self.deterministic = deterministic

    def forward(self, Q, K, V, Bias1, Bias2):
        return TriAttentionFunction.apply(Q,K,V,Bias1,Bias2,self.deterministic)
