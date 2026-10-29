"""Print validated formal complete-process results; do not rerun inference."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('result_json',type=Path,nargs='+');a=p.parse_args()
 print('| Dataset | Samples | Formal PDB | Structures | Complete process (s) | Prediction window (s) |')
 print('|---|---:|---:|---:|---:|---:|')
 for path in a.result_json:
  d=json.loads(path.read_text())
  assert d['status']=='completed_and_validated' and d['formal_rounds']==1
  assert d['validated_formal_structures']==d['formal_inputs']*d['samples']
  print(f"| {d['dataset']} | {d['samples']} | {d['formal_inputs']} | {d['validated_formal_structures']} | {d['formal_full_process_seconds']:.3f} | {d['formal_log_seconds']:.3f} |")
 print('Complete = formal Featurizing to succeeded sum; warm and model loading excluded.')
if __name__=='__main__':main()
