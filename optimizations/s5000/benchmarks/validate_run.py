from pathlib import Path
import json,re,datetime,math,sys
def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False))
def summarize(name, run):
 inputs=read(run/'input_warm_formal.json');formal=[x['name'] for x in inputs if not x['name'].endswith('_warm')]
 from _summary_impl import summarize as summary
 result=summary(run);start={};end={}
 for line in (run/'run.log').read_text().splitlines():
  stamp=re.match(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})',line)
  if not stamp:continue
  dt=datetime.datetime.strptime(stamp[1],'%Y-%m-%d %H:%M:%S,%f')
  m=re.search(r'Featurizing (\S+)\.\.\.',line)
  if m:start[m[1]]=dt
  m=re.search(r'\] (\S+) \[seed:101\] succeeded\.',line)
  if m:end[m[1]]=dt
 assert len(end)==len(inputs) and set(end)=={x['name'] for x in inputs}
 durations={n:(end[n]-start[n]).total_seconds() for n in formal}
 for c in result['cases']:c['formal_full_process_seconds']=durations[c['pdb_id']]
 folders=list((run/'output').glob('*/seed_101/predictions'));assert len(folders)==len(inputs)
 assert all(len(list(p.glob('*.cif')))==5 for p in folders)
 stats=read(run/'optimization_stats.json');counts={**stats,**stats.get('counts',{})}
 forbidden=['std_fp32_to_bf16','confidence_mutlass_calls','conf_v21','v21_native_calls','ct_local_bf16_calls','attn_local_bf16_calls','pair_bias_bf16_builds']
 assert not any(counts.get(k,0) for k in forbidden),{k:counts.get(k,0) for k in forbidden}
 for key in ['aligned_tf32_confidence_calls','aligned_qkvg_fp32_calls','aligned_mul_fp32_calls','aligned_confidence_nochunk_calls','aligned_qkv_layout_calls','precision_cdist_calls','full_fp32_local_flatten_calls','full_bf16_qkvg_cache_calls','full_bf16_trimul_cache_calls']:
  assert counts.get(key,0)>0,(key,counts)
 expected={'test23':{'core':36120,'selective':19200,'local':7200,'conf':240},'eval21':{'core':252840,'selective':105600,'local':50400,'conf':1680}}[name]
 assert counts.get('full_fp32_core_calls',0)==expected['core'],('core',counts)
 assert counts.get('full_selective_fp32_core_calls',0)==expected['selective'],('selective',counts)
 assert counts.get('full_fp32_local_flatten_calls',0)==expected['local'],('local',counts)
 assert counts.get('aligned_tf32_confidence_calls',0)==expected['conf']
 assert counts.get('full_bf16_qkvg_cache_calls',0)==counts['qkvg_calls']
 assert counts.get('full_bf16_trimul_cache_calls',0)==counts['mul_output_calls']
 assert counts.get('fp32_local_flatten_batches',0)==0 and counts.get('fp32_local_flatten_calls',0)==0
 assert counts.get('fp32_bias_softmax_calls',0)==0,counts
 long=['aligned_diffusion_qkv_layout_calls','aligned_batch5_softmax_calls','aligned_chunk640_calls']
 if name=='test23':assert all(counts.get(k,0)>0 for k in long),{k:counts.get(k,0) for k in long}
 else:assert not any(counts.get(k,0) for k in long),{k:counts.get(k,0) for k in long}
 audit=read(run/'runtime_audit.json')
 assert all(audit[k] for k in ['tf32_musa','tf32_cuda','tf32_mudnn'])
 assert audit['model_init_audit'][0]['layernorm_environment']=='fast_layernorm'
 assert audit['model_init_audit'][0]['layernorm_types'].get('FusedLayerNorm',0)>0
 assert audit['tf32_override'] is None
 assert audit['float32_matmul_precision']=='high' and len(audit['memory_model_forward'])==len(inputs)
 expected_libs=read('/workspace/benchmarks/checks/library_reference.json')
 assert {Path(x['path']).name:x['sha256'] for x in audit['loaded_libraries']}=={Path(x['path']).name:x['sha256'] for x in expected_libs}
 assert all(all(v[k] for k in ['tf32_musa','tf32_cuda','tf32_mudnn']) for v in audit['memory_model_forward'])
 conf=[]
 for p in (run/'output').rglob('*summary_confidence*.json'):
  d=read(p)
  for k in ['plddt','ptm','iptm','ranking_score','has_clash']:
   assert k in d and math.isfinite(float(d[k])),(str(p),k)
  conf.append(dict(name=p.parents[2].name,file=p.name,**{k:d.get(k) for k in ['plddt','ptm','iptm','ranking_score','has_clash']}))
 result.update(status='completed_and_validated',formal_full_process_seconds=sum(durations.values()),validated_formal_structures=len(formal)*5,validated_warm_structures=len(formal)*5,optimization_stats=stats,forbidden_precision_counters={k:counts.get(k,0) for k in forbidden},runtime_audit=audit,peak_formal_model_allocated_gib=max(x['allocated'] for x in audit['memory_model_forward'][1::2])/1024**3,tf32=True,metric='sum formal Featurizing to succeeded; exclude warm and model loading',formal_has_clash=any(bool(x.get('has_clash')) for x in conf if not x['name'].endswith('_warm')),confidence=conf,structural_accuracy_equivalence_proven=False)
 result['dataset']=name
 assert audit['model_init_audit'][0]['layernorm_types'].get('FusedLayerNorm')==766
 expected_counts=read('/workspace/benchmarks/checks/best_practice_reference.json')['expected_path_counts'][name]
 assert all(counts.get(k,0)==val for k,val in expected_counts.items()), {'actual':counts,'expected':expected_counts}
 save(run/'result.json',result);return result
if __name__=='__main__':
 r=summarize(sys.argv[1],Path(sys.argv[2]))
 print(json.dumps({k:r[k] for k in ['dataset','status','formal_full_process_seconds','formal_log_seconds','validated_formal_structures','peak_formal_model_allocated_gib']}))
