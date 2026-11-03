"""Run both delivery benchmark entry points, one warm and one formal per PDB."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,re,subprocess,sys,time,traceback
B=Path('/workspace/benchmarks')
W=Path('/workspace')
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-root',type=Path,required=True)
 parser.add_argument('--gpu',type=int,default=6)
 args=parser.parse_args()
 if args.gpu not in range(8):parser.error('--gpu must be 0..7')
 root=args.output_root.resolve()
 if not root.is_relative_to(W/'output'):parser.error('--output-root must be under /workspace/output')
 if root.exists() and any(root.iterdir()):parser.error('Use a new, empty output root')
 root.mkdir(parents=True,exist_ok=True)
 done=[];baseline={}
 for name in ['test23','eval21']:
  old=W/'output'/(name+'_validation')/'result.json'
  if old.exists():
   d=read(old);baseline[name]={'sha256':sha(old),'seconds':d['formal_full_process_seconds'],'path':str(old)}
 status=root/'validation_status.json'
 try:
  save(status,{'state':'checking_files','completed':done})
  with (root/'integrity_console.log').open('w') as f:
   p=subprocess.run([sys.executable,'-B',str(B/'verify_installation.py'),'--output-json',str(root/'integrity_check.json')],stdout=f,stderr=subprocess.STDOUT)
  assert p.returncode==0,'Integrity check failed'
  for name in ['test23','eval21']:
   run=root/name
   env={**os.environ,'PROTENIX_GPU':str(args.gpu),'PYTHONDONTWRITEBYTECODE':'1'}
   env['TEST23_OUTPUT_ROOT' if name=='test23' else 'EVAL21_OUTPUT_ROOT']=str(run)
   command=['bash',str(B/('run_test23.sh' if name=='test23' else 'run_eval21.sh'))]
   save(status,dict(state='running',current_set=name,formal_complete=0,warm_complete=0,completed=done))
   with (root/(name+'_console.log')).open('w') as f:
    p=subprocess.Popen(command,cwd=B,env=env,stdout=f,stderr=subprocess.STDOUT)
    while p.poll() is None:
     log=(run/'run.log').read_text(errors='replace') if (run/'run.log').exists() else ''
     hits=re.findall(r'\] (\S+) \[seed:101\] succeeded\.',log)
     starts=re.findall(r'Featurizing (\S+)\.\.\.',log)
     save(status,dict(state='running',current_set=name,formal_complete=sum(not n.endswith('_warm') for n in hits),warm_complete=sum(n.endswith('_warm') for n in hits),current_pdb=starts[-1:] or [],completed=done))
     time.sleep(3)
   assert p.returncode==0,{'dataset':name,'exit':p.returncode,'log':str(root/(name+'_console.log'))}
   d=read(run/'result.json');assert d['status']=='completed_and_validated'
   count=3 if name=='test23' else 21
   assert d['validated_formal_structures']==d['validated_warm_structures']==count*5
   item={k:d[k] for k in ['dataset','formal_full_process_seconds','formal_log_seconds','validated_formal_structures','validated_warm_structures','peak_formal_model_allocated_gib','formal_has_clash']}
   if name in baseline:
    old=baseline[name]
    assert sha(old['path'])==old['sha256'],'Previous results changed'
    item['previous_full_seconds']=old['seconds']
    item['time_change_percent']=(d['formal_full_process_seconds']/old['seconds']-1)*100
   done.append(item)
   print('DATASET_COMPLETE',json.dumps(item),flush=True)
  assert not (W/'reports').exists(),'Unexpected removed directory recreated'
  with (root/'summary.md').open('w') as f:
   p=subprocess.run([sys.executable,'-B',str(B/'summary.py'),str(root/'test23/result.json'),str(root/'eval21/result.json')],stdout=f,stderr=subprocess.STDOUT)
  assert p.returncode==0,'Summary entry failed'
  save(root/'comparison.json',{'status':'complete','gpu':args.gpu,'samples':5,'cycles':10,'steps':200,'seed':101,'rows':done,'removed_directory_recreated':False,'metric':'sum formal Featurizing to succeeded; exclude warm and model loading'})
  save(status,dict(state='complete',completed=done,gpu=args.gpu,samples=5))
 except BaseException:
  save(status,dict(state='failed',completed=done,error=traceback.format_exc()));raise
if __name__=='__main__':main()
