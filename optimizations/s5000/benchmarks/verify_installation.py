"""Verify copied runtime code, inputs, MSA, checkpoint and benchmark files."""
from pathlib import Path
import argparse,hashlib,json
W=Path('/workspace')
CHECKS=W/'benchmarks/checks'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify():
 m=json.loads((CHECKS/'delivery_manifest.json').read_text())
 benchmark=json.loads((CHECKS/'benchmark_manifest.json').read_text())
 for name in m.get('removed_paths',[]):
  assert not (W/name).exists(),'Retired optimization file remains: '+name
 checked={}
 for section in [m['files'],m['generated_files'],benchmark]:
  for name,record in section.items():
   p=W/name
   assert p.is_file() and not p.is_symlink(),str(p)
   assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],str(p)
   checked[name]=record['sha256']
 links=[]
 for root in ['Protenix','data','benchmarks']:
  for p in (W/root).rglob('*'):
   if p.is_symlink():
    assert p.exists() and p.resolve().is_relative_to(W),str(p)
    links.append(str(p))
 for dataset,total in [('test23',3),('eval21',21)]:
  inputs=json.loads((W/'data'/dataset/'input.json').read_text())
  assert isinstance(inputs,list) and len(inputs)==total,dataset
  def walk(x):
   if isinstance(x,dict):
    for k,v in x.items():
     if k=='precomputed_msa_dir':
      p=Path(v)
      assert p.is_dir() and p.resolve().is_relative_to(W/'data'),v
     walk(v)
   elif isinstance(x,list):
    for v in x:walk(v)
  walk(inputs)
 assert not any((Path('/app')/x).exists() for x in ['protenix','output','protenix_10case_input'])
 assert not any(Path('/home').iterdir())
 assert not Path('/root/checkpoint').is_symlink() and not Path('/root/common').is_symlink()
 return {'status':'pass','verified_files':len(checked),'manifest_files':m['file_count'],
         'checkpoint_sha256':checked['data/checkpoint/protenix_base_default_v0.5.0.pt'],
         'external_symlinks':False,'links':links,'runtime_root':str(W),
         'network_required':False,'host_data_mount_required':False}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output-json',type=Path,default=W/'output/integrity_check.json')
 args=p.parse_args();result=verify()
 args.output_json.parent.mkdir(parents=True,exist_ok=True)
 args.output_json.write_text(json.dumps(result,indent=2))
 print(json.dumps(result))
if __name__=='__main__':main()
