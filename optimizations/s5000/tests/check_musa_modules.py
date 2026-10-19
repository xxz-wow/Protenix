"""Manual MUSA module checks; set MUSA_VISIBLE_DEVICES to an idle GPU before running."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
os.environ['LAYERNORM_TYPE']='torch'
os.environ['PROTENIX_DISABLE_MUSA_LOCAL_SDPA_FLATTEN']='0'
os.environ['PROTENIX_DISABLE_MUSA_ADAPTIVE_LAYERNORM_FUSION']='0'
import torch,torch_musa
from protenix.model.modules.primitives import Attention,AdaptiveLayerNorm,_local_attention
from protenix.model.modules.transformer import ConditionedTransitionBlock,AttentionPairBias
from protenix.model.musa_kernels.triangle_fusion import GatedLayout
from protenix.model.musa_kernels.runtime import COUNTS,STATS
torch.manual_seed(101)
results=[]
def check(name,tensor,shape,dtype):
 assert tuple(tensor.shape)==tuple(shape),(name,tensor.shape,shape)
 assert tensor.dtype==dtype,(name,tensor.dtype,dtype)
 assert bool(torch.isfinite(tensor).all().item()),name
 results.append(dict(name=name,shape=list(shape),dtype=str(dtype),finite=True))
with torch.no_grad():
 ct=ConditionedTransitionBlock(768,384).eval().to('musa')
 a=torch.randn(2,32,768,device='musa');s=torch.randn(2,32,384,device='musa')
 os.environ['PROTENIX_ENABLE_MUSA_CT_LOCAL_BF16']='1'
 STATS.clear();o=ct(a,s);check('ct_bf16_restored_adaln',o,a.shape,a.dtype)
 assert STATS['ct_local_bf16_calls']==1
 assert ct.adaln._projection_weight is not None
 assert COUNTS.get('adaln',0)>0
 os.environ['PROTENIX_DISABLE_MUSA_ADAPTIVE_LAYERNORM_FUSION']='1'
 COUNTS.clear();ln=ct.adaln(a,s);expected=torch.sigmoid(ct.adaln.linear_s(ct.adaln.layernorm_s(s)))*ct.adaln.layernorm_a(a)+ct.adaln.linear_nobias_s(ct.adaln.layernorm_s(s))
 torch.testing.assert_close(ln,expected,rtol=0,atol=0)
 assert not COUNTS.get('adaln',0)
 results.append(dict(name='adaln_separate_projection_fallback',exact=True))
 os.environ['PROTENIX_DISABLE_MUSA_ADAPTIVE_LAYERNORM_FUSION']='0'
 COUNTS.clear();packed=ct.adaln(a,s)
 torch.testing.assert_close(packed,expected,rtol=2e-5,atol=2e-5)
 assert COUNTS.get('adaln',0)==1
 # BF16 scope is inactive on CPU and in training.
 cpu=ConditionedTransitionBlock(768,384).eval();STATS.clear()
 check('ct_cpu_fallback',cpu(a.cpu(),s.cpu()),a.shape,a.dtype)
 assert STATS.get('ct_local_bf16_calls',0)==0
 ct.train();STATS.clear();check('ct_training_fallback',ct(a,s),a.shape,a.dtype)
 assert STATS.get('ct_local_bf16_calls',0)==0
 att=Attention(768,768,768,32,4,zero_init=False).eval().to('musa')
 x=torch.randn(2,400,768,device='musa');bias=torch.randn(1,4,400,400,device='musa')*.01
 os.environ['PROTENIX_ENABLE_MUSA_ATTN_LOCAL_BF16']='1';COUNTS.clear()
 o=att(x,x,attn_bias=bias);check('attention_400_bf16',o,x.shape,x.dtype)
 assert COUNTS['attn_local_bf16_calls']==1
 COUNTS.clear();short=x[:,:32].contiguous()
 check('attention_short_fallback',att(short,short,attn_bias=bias[:,:,:32,:32]),short.shape,short.dtype)
 assert COUNTS.get('attn_local_bf16_calls',0)==0
 # Cached bias is built once and reused.
 pair=AttentionPairBias(has_s=False,n_heads=4,c_a=768,c_z=128).eval().to('musa')
 q=torch.randn(2,16,768,device='musa');z=torch.randn(1,128,16,16,device='musa')
 cache={'channels':z,'biases':{}}
 os.environ['PROTENIX_ENABLE_MUSA_PAIR_BIAS_BF16_CACHE']='1';STATS.clear()
 first=pair.standard_multihead_attention(q,q,z,inplace_safe=False,enable_efficient_fusion=True,pair_bias_cache=cache)
 second=pair.standard_multihead_attention(q,q,z,inplace_safe=False,enable_efficient_fusion=True,pair_bias_cache=cache)
 check('pair_bias_bf16_cached',second,q.shape,q.dtype)
 assert cache['biases'][pair].dtype==torch.bfloat16
 assert STATS['pair_bias_bf16_builds']==1 and STATS['pair_bias_hits']==1
 torch.testing.assert_close(first,second,rtol=0,atol=0)
 # Original chunk path remains available; also verify restored flatten path.
 os.environ['PROTENIX_DISABLE_MUSA_LOCAL_SDPA_FLATTEN']='1'
 q=torch.randn(2,2,64,32,device='musa')*.1
 COUNTS.clear()
 local=_local_attention(q,q,q,n_queries=32,n_keys=64,chunk_size=1)
 whole=_local_attention(q,q,q,n_queries=32,n_keys=64,chunk_size=None)
 check('restored_local_chunk_path',local,q.shape,q.dtype)
 torch.testing.assert_close(local,whole,rtol=2e-2,atol=2e-2)
 assert COUNTS.get('local_flatten',0)==0
 os.environ['PROTENIX_DISABLE_MUSA_LOCAL_SDPA_FLATTEN']='0'
 COUNTS.clear()
 flat=_local_attention(q,q,q,n_queries=32,n_keys=64,chunk_size=1,use_efficient_implementation=True)
 torch.testing.assert_close(flat,whole,rtol=2e-2,atol=2e-2)
 assert COUNTS.get('local_flatten',0)==1
 check('flattened_local_attention',flat,q.shape,q.dtype)
 for n in [328,851,964]:
  p=torch.randn(1,n,n,512,device='musa',dtype=torch.bfloat16)
  os.environ['PROTENIX_ENABLE_MUSA_TRIMUL_TILE_V2']='0';old=GatedLayout.apply(p)
  os.environ['PROTENIX_ENABLE_MUSA_TRIMUL_TILE_V2']='1';STATS.clear();new=GatedLayout.apply(p)
  assert STATS['trimul_tile_v2_calls']==1
  for x,y in zip(old,new):torch.testing.assert_close(x,y,rtol=0,atol=0)
  results.append(dict(name='tile_'+str(n),exact=True))
  del p,old,new
torch.musa.synchronize()
out=dict(device=str(torch.musa.get_device_name(0)),visible_devices=os.environ.get('MUSA_VISIBLE_DEVICES'),checks=results,accuracy_equivalence_claimed=False,full_eval21_retested=False,scope='module checks only; release benchmark evidence is separate')
print(json.dumps(out,indent=2))
