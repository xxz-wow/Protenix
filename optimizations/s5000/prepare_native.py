#!/usr/bin/env python3
"""Copy and verify prebuilt native dependencies from the matching image tree."""
import argparse,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source-root',type=Path,required=True,help='Protenix tree in the matching delivered image')
 p.add_argument('--target-root',type=Path,required=True,help='Patched upstream Protenix tree')
 a=p.parse_args();m=json.loads((HERE/'benchmarks/checks/delivery_manifest.json').read_text())
 files={k.removeprefix('Protenix/'):v for k,v in m['files'].items() if k.startswith('Protenix/') and k.endswith('.so')}
 for name,r in files.items():
  s=a.source_root/name
  if not s.is_file() or s.stat().st_size!=r['bytes'] or digest(s)!=r['sha256']:
   raise ValueError('Missing or mismatched native dependency: '+name)
 # Validate every dependency before writing any target.
 for name,r in files.items():
  s=a.source_root/name;t=a.target_root/name
  if s.resolve()!=t.resolve():
   t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,t)
  if digest(t)!=r['sha256']:raise ValueError('Copy verification failed: '+name)
 print('Verified native dependencies:',len(files))
if __name__=='__main__':main()
