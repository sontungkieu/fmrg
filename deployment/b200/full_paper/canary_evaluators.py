"""Small evaluator canaries only. Never launches a generation matrix."""
import argparse,json,os,sys,time
from pathlib import Path
def metrics(assets,device):
 from prepare_metric_overlay import prepare
 setup=prepare(assets);os.environ['TORCH_HOME']=setup['TORCH_HOME'];sys.path.insert(0,setup['python_overlay'])
 import torch,lpips
 from download_assets import digest
 assert digest(Path(lpips.__file__).parent/'weights/v0.1/vgg.pth')==digest(Path(assets)/'metrics/lpips-v0.1-vgg.pth')
 from torchmetrics.image.fid import FrechetInceptionDistance
 torch.manual_seed(314)
 network=lpips.LPIPS(net='vgg',verbose=False).to(device).eval()
 a=torch.rand(1,3,224,224,device=device)*2-1
 with torch.no_grad():identity=float(network(a,a).mean())
 assert abs(identity)<1e-6
 b=(a+.05*torch.randn_like(a)).clamp(-1,1).detach().requires_grad_(True)
 loss=network(a,b).mean();g,=torch.autograd.grad(loss,b)
 assert torch.isfinite(loss) and torch.isfinite(g).all() and g.norm()>0
 fid=FrechetInceptionDistance(feature=2048,normalize=False).to(device)
 fid.update(torch.randint(0,256,(2,3,299,299),dtype=torch.uint8,device=device),real=True)
 assert torch.isfinite(fid.real_features_sum).all() and int(fid.real_features_num_samples)==2
 return dict(kind='SYNTHETIC_METRIC_CODE_CANARY',device=device,lpips_identity=identity,
  lpips_perturbed=float(loss.detach()),input_gradient_norm=float(g.norm()),fid_feature_samples=2,
  fid_features_finite=True,fid_dataset_score='NOT_COMPUTED',scientific_result=False)
def reward(assets,core,device):
 import torch
 from offline_rewards import load_ensemble
 assert device=='cuda' and torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(10,0)
 torch.manual_seed(314);start=time.perf_counter();ensemble=load_ensemble(assets,core,device)
 x=torch.rand(1,3,224,224,device=device,requires_grad=True)
 loss,parts=ensemble(x,'a photo of a dog');g,=torch.autograd.grad(loss,x)
 assert torch.isfinite(g).all() and g.norm()>0 and all(__import__('math').isfinite(v) for v in parts.values())
 return dict(kind='SYNTHETIC_REWARD_INPUT_GRADIENT_CANARY',device=device,losses=parts,
  gradient_norm=float(g.norm()),seconds=time.perf_counter()-start,
  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,scientific_result=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('kind',choices=['metrics','reward']);p.add_argument('--assets',required=True)
 p.add_argument('--core-assets');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();assert not a.out.exists(),'New attempt output required'
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_XET='1')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 try:
  record=metrics(a.assets,a.device) if a.kind=='metrics' else reward(a.assets,a.core_assets,a.device)
  record['status']='PASS'
 except Exception as e:
  a.out.write_text(json.dumps(dict(status='FAIL',error_type=type(e).__name__,error=str(e)),indent=2));raise
 a.out.write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
