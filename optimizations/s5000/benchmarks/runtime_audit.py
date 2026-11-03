from pathlib import Path
import os,sys,atexit,json,runpy
T=Path('/workspace/benchmarks')
sys.path.insert(0,str(T))
import torch,torch_musa
from audit_common import checked_paths,identities
from protenix.model.protenix import Protenix
REPO=Path('/workspace/Protenix')
def tf32():
 return dict(tf32_musa=bool(torch.backends.musa.matmul.allow_tf32),tf32_cuda=bool(torch.backends.cuda.matmul.allow_tf32),tf32_mudnn=bool(torch.backends.mudnn.allow_tf32))
initial=checked_paths();records=[];init_records=[]
original_init=Protenix.__init__
def init(self,*args,**kwargs):
 original_init(self,*args,**kwargs)
 flags=tf32();assert all(flags.values()),flags
 from collections import Counter
 from protenix.model.triangular.layers import fastln_is_installed
 kinds=dict(Counter(type(m).__name__ for m in self.modules() if 'LayerNorm' in type(m).__name__))
 assert os.getenv('LAYERNORM_TYPE')=='fast_layernorm' and fastln_is_installed
 assert kinds.get('FusedLayerNorm',0)>0,kinds
 init_records.append(dict(no_runtime_override=True,flags=flags,layernorm_types=kinds,layernorm_environment=os.getenv('LAYERNORM_TYPE')))
Protenix.__init__=init
original=Protenix.forward
def forward(self,*args,**kwargs):
 flags=tf32();assert all(flags.values()),flags
 before=checked_paths();torch.musa.reset_peak_memory_stats()
 out=original(self,*args,**kwargs)
 assert all(tf32().values()),tf32()
 pred=out[0] if isinstance(out,tuple) else out
 dtypes={k:str(v.dtype) for k,v in pred.items() if isinstance(v,torch.Tensor)} if isinstance(pred,dict) else {}
 records.append(dict(allocated=torch.musa.max_memory_allocated(),reserved=torch.musa.max_memory_reserved(),library_paths=checked_paths(),output_dtypes=dtypes,**flags))
 return out
Protenix.forward=forward
def record():
 modules={}
 for name in ['tf32_confidence','tf32_pairformer_qkv_layout','tf32_diffusion_qkv_layout','tf32_selective_attention']:
  module=sys.modules.get('protenix.model.musa_kernels.'+name)
  if module is not None:modules[name]=dict(module.COUNTS)
 out=dict(**tf32(),memory_model_forward=records,initial_library_paths=initial,loaded_libraries=identities(),torch=str(torch.__version__),torch_musa=str(torch_musa.__version__),expected_library_root=os.environ['MUDNN_EXPECT_ROOT'],float32_matmul_precision=torch.get_float32_matmul_precision(),tf32_override=os.environ.get('TORCH_ALLOW_TF32_MUBLAS_OVERRIDE'),model_init_audit=init_records,native_counters=modules)
 (Path(os.environ['S5000_RUN_DIR'])/'runtime_audit.json').write_text(json.dumps(out,indent=2))
atexit.register(record)
runpy.run_path(str(REPO/'runner/inference.py'),run_name='__main__')
