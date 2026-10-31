"""CPU-only full import preflight and missing pure-Python dependency repair."""
import json,os,subprocess
from pathlib import Path
from restore_overlay import restore,digest
ROOT=Path('/data/geneval-20261101');PY=Path('/opt/fmrg-geneval/venv/bin/python')

def main():
 record=json.loads((ROOT/'build.json').read_text());a=record['overlay']
 restore(a['path'],a['sha256'],a['bytes'])
 requirements=['terminaltables==3.1.10','scikit-learn==1.2.2','joblib==1.2.0','threadpoolctl==3.1.0']
 missing=subprocess.run([str(PY),'-c',"import terminaltables,sklearn,joblib,threadpoolctl"],capture_output=True).returncode!=0
 if missing:
  subprocess.run([str(PY),'-m','pip','install','--no-deps',*requirements],check=True,timeout=90)
 subprocess.run([str(PY),'-c',"import torch,mmcv,mmdet,open_clip; from mmdet.apis import init_detector,inference_detector; from clip_benchmark.metrics import zeroshot_classification; from mmcv.ops import MultiScaleDeformableAttention; print('FULL_IMPORT_PREFLIGHT_PASS')"],check=True,timeout=90)
 if missing:
  original=ROOT/'build-initial.json'
  if not original.exists():original.write_text(json.dumps(record,indent=2))
  freeze=subprocess.check_output([str(PY),'-m','pip','freeze','--all'],text=True)
  Path('/opt/fmrg-geneval/resolved-freeze.txt').write_text(freeze)
  archive=ROOT/'geneval-overlay-v2.tar.gz'
  subprocess.run(['tar','-I','gzip -1','-cf',str(archive),'-C','/','opt/fmrg-geneval'],check=True,timeout=90)
  record['overlay']=dict(path=str(archive),bytes=archive.stat().st_size,sha256=digest(archive))
  record['overlay_dependency_repair']=requirements
 record['cpu_full_import_preflight']='PASS'
 (ROOT/'build.json').write_text(json.dumps(record,indent=2))
 result=dict(status='PASS',repaired_terminaltables=missing,overlay=record['overlay'],full_launch_allowed=False)
 (ROOT/'preflight.json').write_text(json.dumps(result,indent=2));print('FMRG_GENEVAL_PREFLIGHT='+json.dumps(result),flush=True)

if __name__=='__main__':main()
