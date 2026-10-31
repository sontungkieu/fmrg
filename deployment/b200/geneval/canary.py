"""Bounded evaluator qualification on frozen fixtures. Never generates FLUX images."""
import argparse,gc,json,os,time
from pathlib import Path
from offline_adapter import load,sha

def native():
 import torch
 from mmcv.ops import nms
 from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction,multi_scale_deformable_attn_pytorch
 torch.manual_seed(911)
 shapes=torch.tensor([[2,2],[2,2]],dtype=torch.long)
 starts=torch.tensor([0,4],dtype=torch.long)
 values=torch.randn(1,8,2,4,dtype=torch.float64,requires_grad=True)
 locations=torch.rand(1,3,2,2,2,2,dtype=torch.float64,requires_grad=True)
 weights=torch.rand(1,3,2,2,2,dtype=torch.float64,requires_grad=True)
 gpu=[x.detach().cuda().requires_grad_() for x in [values,locations,weights]]
 reference=multi_scale_deformable_attn_pytorch(values,shapes,locations,weights)
 actual=MultiScaleDeformableAttnFunction.apply(gpu[0],shapes.cuda(),starts.cuda(),gpu[1],gpu[2],64)
 forward=float((actual.cpu()-reference).abs().max());assert forward<1e-7
 cpu_grad=torch.autograd.grad(reference.square().sum(),[values,locations,weights])
 gpu_grad=torch.autograd.grad(actual.square().sum(),gpu)
 gradient=[float((a.cpu()-b).abs().max()) for a,b in zip(gpu_grad,cpu_grad)]
 assert max(gradient)<1e-7 and all(torch.isfinite(g).all() for g in gpu_grad)
 boxes=torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.],[30.,30.,40.,40.]],device='cuda')
 _,keep=nms(boxes,torch.tensor([.9,.8,.7],device='cuda'),.5)
 assert keep.tolist()==[0,2]
 return dict(status='PASS',msda_forward_max_abs=forward,msda_gradient_max_abs=gradient,nms_keep=keep.tolist())

def run(assets,out,prefix='/opt/fmrg-geneval'):
 import numpy as np,torch
 from PIL import Image
 out=Path(out);assert not out.exists(),'New canary receipt path required'
 assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(10,0)
 start=time.monotonic();record=dict(status='RUNNING',kind='GENEVAL_NATIVE_AND_FIXTURE_CANARY',
  torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=[10,0],
  full_launch_allowed=False,scientific_result=False)
 record['code_sha256']={name:sha(Path(__file__).parent/name) for name in ['canary.py','offline_adapter.py','source-pins.json']}
 try:
  record['native']=native();record['profiles']=[]
  fixture=Path(prefix)/'source/sources/mmdetection/demo/demo.jpg';assert fixture.is_file()
  record['fixture_sha256']=sha(fixture)
  for profile in ['r50','swin_s']:
   with load(assets,profile,prefix) as (ns,identity):
    metadata=dict(tag='single_object',prompt='a photo of a car',include=[{'class':'car','count':1}])
    row=ns['evaluate_image'](str(fixture),metadata)
    details=json.loads(row['details'])
    assert isinstance(row['correct'],bool) and all(np.isfinite(np.asarray(v)).all() for v in details.values())
    # Force the official color classifier path even if this fixture has no selected car.
    color=ns['color_classification'](Image.new('RGB',(64,64),'red'),[(np.array([0,0,63,63,1.]),None)],'car')
    assert len(color)==1 and color[0] in ns['COLORS']
    identity.update(status='PASS',detected_classes=sorted(details),synthetic_color_label=color[0],
                    fixture_correct=row['correct'],fixture_score_interpretation='NO_EFFICACY_CLAIM')
    record['profiles'].append(identity)
   del ns;gc.collect();torch.cuda.empty_cache()
  torch.cuda.synchronize();record['status']='PASS'
 except Exception as e:
  record.update(status='FAIL',error_type=type(e).__name__,error=str(e));raise
 finally:
  record.update(seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
  out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,indent=2))
  print('FMRG_GENEVAL_CANARY='+json.dumps(record),flush=True)
 return record

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--out',required=True);p.add_argument('--prefix',default='/opt/fmrg-geneval');a=p.parse_args()
 run(a.assets,a.out,a.prefix)
