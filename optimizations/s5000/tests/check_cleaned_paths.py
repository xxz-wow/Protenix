"""Compare retained MUSA paths against the reference image; no dataset needed.

Run in the matching MUSA environment. The reference tree supplies unchanged
dependencies and native libraries. The four cleaned modules load separately.
"""
import argparse,importlib,importlib.util,json,os,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--clean-root',type=Path,required=True)
    p.add_argument('--flags',type=Path,required=True)
    p.add_argument('--output-json',type=Path,required=True)
    a=p.parse_args()
    os.environ.update(json.loads(a.flags.read_text()))
    os.environ['LAYERNORM_TYPE']='fast_layernorm'
    os.environ['OMP_NUM_THREADS']='4'
    sys.path.insert(0,str(a.reference_root))
    import torch,torch_musa
    torch.manual_seed(101);torch.musa.manual_seed_all(101)
    paths={'primitives':'protenix/model/modules/primitives.py',
           'transformer':'protenix/model/modules/transformer.py',
           'layers':'protenix/model/triangular/layers.py',
           'triangular':'protenix/model/triangular/triangular.py'}
    old={k:importlib.import_module(v[:-3].replace('/','.')) for k,v in paths.items()}
    new={}
    for key,rel in paths.items():
        spec=importlib.util.spec_from_file_location('cleaned_'+key,a.clean_root/rel)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);new[key]=m
    new['triangular'].Attention=new['layers'].Attention
    new['transformer'].AdaptiveLayerNorm=new['primitives'].AdaptiveLayerNorm
    def pair(old_class,new_class,*args,**kwargs):
        before=old_class(*args,**kwargs).eval().to('musa')
        after=new_class(*args,**kwargs).eval().to('musa')
        with torch.no_grad():
            for param in before.parameters():param.normal_(0,0.02)
        after.load_state_dict(before.state_dict())
        return before,after
    def tf32(value):
        torch.backends.cuda.matmul.allow_tf32=value
        torch.backends.musa.matmul.allow_tf32=value
        torch.backends.mudnn.allow_tf32=value
        torch.set_float32_matmul_precision('high' if value else 'highest')
    def role(module,value):
        for part in module.modules():part._musa_confidence=value
    records=[]
    def compare(label,before,after,call,amp=False):
        with torch.no_grad(),torch.autocast('musa',enabled=amp,dtype=torch.bfloat16):
            for phase in ['cold','cache_hit','parameter_update']:
                if phase=='parameter_update':
                    # An actual parameter update must invalidate packed caches.
                    for module in [before,after]:
                        if hasattr(module,'linear_q'):weight=module.linear_q.weight
                        elif hasattr(module,'mha'):weight=module.mha.linear_q.weight
                        elif hasattr(module,'linear_a_p'):weight=module.linear_a_p.weight
                        else:weight=module.adaln.linear_s.weight
                        weight.add_(0.0001)
                x=call(before);y=call(after)
                torch.testing.assert_close(x,y,rtol=0,atol=0,check_dtype=True)
                assert torch.isfinite(y).all().item()
                records.append(dict(case=label,phase=phase,tf32=bool(torch.backends.musa.matmul.allow_tf32),
                                    dtype=str(y.dtype),shape=list(y.shape),max_abs_difference=float((x.float()-y.float()).abs().max())))
    for enabled in [True,False]:
        tf32(enabled)
        for local,tokens in [(False,64),(False,400),(True,64)]:
            kw=dict(c_q=768,c_k=768,c_v=768,c_hidden=48,num_heads=16,zero_init=False,
                    local_attention_method='local_cross_attention' if local else 'global_attention_with_bias')
            before,after=pair(old['primitives'].Attention,new['primitives'].Attention,**kw)
            x=torch.randn(5,tokens,768,device='musa')
            bias=None if local else torch.randn(1,16,tokens,tokens,device='musa')*0.01
            params=dict(n_queries=32,n_keys=128,chunk_size=16) if local else {}
            compare('local_attention' if local else 'token_attention_'+str(tokens),before,after,
                    lambda m:m(x,x,attn_bias=bias,**params))
        before,after=pair(old['transformer'].ConditionedTransitionBlock,new['transformer'].ConditionedTransitionBlock,768,384)
        x=torch.randn(5,16,768,device='musa');s=torch.randn(5,16,384,device='musa')
        compare('conditioned_transition_fp32',before,after,lambda m:m(x,s))
        for conf in [False,True]:
            for starting in [False,True]:
                before,after=pair(old['triangular'].TriangleAttention,new['triangular'].TriangleAttention,128,32,4,starting=starting)
                role(before,conf);role(after,conf)
                z=torch.randn(32,32,128,device='musa')
                compare('triangle_attention_'+str(conf)+'_'+str(starting),before,after,
                        lambda m:m(z.clone(),triangle_attention='triattention',chunk_size=None),amp=not conf)
            for outgoing in [False,True]:
                for inplace in [False,True]:
                    before,after=pair(old['triangular'].TriangleMultiplicativeUpdate,new['triangular'].TriangleMultiplicativeUpdate,128,128,_outgoing=outgoing)
                    role(before,conf);role(after,conf)
                    z=torch.randn(32,32,128,device='musa')
                    compare('triangle_multiplication_'+str(conf)+'_'+str(outgoing)+'_'+str(inplace),before,after,
                            lambda m:m(z.clone(),triangle_multiplicative='torch',inplace_safe=inplace,_add_with_inplace=inplace),amp=not conf)
    torch.musa.synchronize()
    forbidden=[x for x in sys.modules if 'mutlass_triangle_bmm' in x or x.startswith('protenix.model.tri_attention.v21')]
    assert not forbidden,forbidden
    result=dict(status='pass',scope='Retained module output equality; not full PDB performance or structural accuracy validation',
                seed=101,tf32_modes=[True,False],comparisons=len(records),all_outputs_bitwise_equal=True,
                all_outputs_finite=True,retired_modules_loaded=forbidden,cases=records)
    a.output_json.parent.mkdir(parents=True,exist_ok=True)
    a.output_json.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
if __name__=='__main__':main()
