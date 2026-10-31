"""Bounded technical inventory, with no generation or benchmark."""
import importlib.metadata,json,os,shutil,subprocess
from pathlib import Path

def run(out):
 import torch
 from torch.utils.cpp_extension import CUDA_HOME
 from huggingface_hub import hf_hub_download
 r=dict(torch=torch.__version__,cuda=torch.version.cuda,cuda_home=CUDA_HOME,
        nvcc=shutil.which('nvcc'),gcc=shutil.which('gcc'),python=os.sys.executable,
        packages={k:None for k in ['mmcv-full','mmdet','open-clip-torch','timm','numpy']},full_launch_allowed=False)
 for k in r['packages']:
  try:r['packages'][k]=importlib.metadata.version(k)
  except importlib.metadata.PackageNotFoundError:pass
 for name in ['/usr/local/cuda','/usr/local/cuda-12.8','/opt/fmrg']:
  r[name]=Path(name).exists()
 try:
  p=hf_hub_download('Skywork/Skywork-VL-Reward-7B','config.json',revision='cfc6496e451f4e636d7e78ce6249a4ee0efb89a1',local_dir='/data/geneval-20261101/vlm-access-probe')
  r['vlm_access']='CONFIG_ACCESS_GRANTED';r['vlm_config_bytes']=Path(p).stat().st_size
 except Exception as e:r['vlm_access']=type(e).__name__
 out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2))
 print('FMRG_GENEVAL_PROBE='+json.dumps(r),flush=True)
 return r

if __name__=='__main__':run(os.sys.argv[1])
