"""Extract pinned evaluator source and check a separately supplied native env.

No installation, GPU inference, or benchmark is performed here.
"""
import argparse,importlib.metadata,json,tarfile
from pathlib import Path,PurePosixPath
from verify_assets import verify

def extract(assets,dest):
 verify(assets,['sources','geneval'])
 dest=Path(dest).resolve();assert not dest.exists(),'New source directory required'
 archive=Path(assets)/'sources/evaluator-source.tar.gz'
 with tarfile.open(archive) as tar:
  members=tar.getmembers()
  for m in members:
   p=PurePosixPath(m.name)
   assert not p.is_absolute() and '..' not in p.parts and (m.isfile() or m.isdir())
  dest.mkdir(parents=True)
  tar.extractall(dest,members=members,filter='data')
 return dict(source_root=str(dest),status='SOURCE_EXTRACTED_NOT_NATIVE_QUALIFIED',full_launch_allowed=False)

def env_check():
 import torch,mmcv,mmdet,open_clip
 from mmcv.ops import MultiScaleDeformableAttention
 versions={k:importlib.metadata.version(k) for k in ['torch','mmcv-full','mmdet','open-clip-torch','clip-benchmark','timm']}
 assert mmcv.__version__=='1.7.1' and mmdet.__version__=='2.28.2'
 return dict(status='IMPORTS_ONLY_NOT_KERNEL_QUALIFIED',versions=versions,cuda=torch.version.cuda,
             gpu_available=torch.cuda.is_available(),full_launch_allowed=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--assets');p.add_argument('--dest');p.add_argument('--check-env',action='store_true');a=p.parse_args()
 if a.check_env:r=env_check()
 else:
  assert a.assets and a.dest
  r=extract(a.assets,a.dest)
 print(json.dumps(r,indent=2))
