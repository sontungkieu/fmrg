"""Extract one pinned pure-Python wheel and point torch at verified local weights."""
import argparse,hashlib,json,os,shutil,tarfile,zipfile
from pathlib import Path,PurePosixPath
def prepare(root):
 root=Path(root).resolve();receipt=json.loads((root/'download-receipt.json').read_text())
 assert receipt['status']=='DOWNLOADS_HASH_VERIFIED'
 from verify_assets import verify
 verify(root,['metrics'])
 wheel=root/'metrics/torch_fidelity-0.3.0-py3-none-any.whl'
 assert hashlib.sha256(wheel.read_bytes()).hexdigest()=='d01284825595feb7dc3eae3dc9a0d8ced02be764813a3483f109bc142b52a1d3'
 overlay=root/'python-overlay';overlay.mkdir(exist_ok=True)
 with zipfile.ZipFile(wheel) as z:
  for m in z.infolist():
   name=PurePosixPath(m.filename)
   assert not name.is_absolute() and '..' not in name.parts
   if m.is_dir():continue
   target=overlay.joinpath(*name.parts);target.parent.mkdir(parents=True,exist_ok=True)
   data=z.read(m)
   if target.exists():assert target.read_bytes()==data
   else:target.write_bytes(data)
 checkpoints=root/'torch/hub/checkpoints';checkpoints.mkdir(parents=True,exist_ok=True)
 for name in ['vgg16-397923af.pth','weights-inception-2015-12-05-6726825d.pth']:
  source=root/'metrics'/name;assert source.is_file()
  dest=checkpoints/name
  if dest.exists():assert dest.resolve()==source.resolve()
  else:dest.symlink_to(source)
 return dict(python_overlay=str(overlay),TORCH_HOME=str(root/'torch'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--assets',required=True);a=p.parse_args()
 print(json.dumps(prepare(a.assets),indent=2))
