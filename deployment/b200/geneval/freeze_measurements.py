"""CPU-only observation preparation for the reconstructed image split.

No FLUX, VAE, reward model, generated image or efficacy evaluation is loaded.
"""
import hashlib,json,os,random,sys,tarfile,time
from pathlib import Path
ROOT=Path('/data/geneval-20261101');OUT=ROOT/'measurements'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def seed_for(text):return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little')

def prepare():
 import numpy as np,torch
 from PIL import Image
 from torchvision.transforms.functional import to_tensor
 from safetensors.torch import save_file
 from munch import Munch
 from functions.degradation import get_degradation
 from strict_evidence import validate_inverse_split
 from prepare_additions import publish
 torch.set_num_threads(4)
 manifest=json.loads((ROOT/'reconstructed/manifest.json').read_text());assert manifest['claim']=='reconstructed'
 validate_inverse_split(manifest,require_measurements=False)
 pins=json.loads((Path(__file__).parent/'measurement-source-pins.json').read_text())
 source=Path('/opt/fmrg/source')
 for name,digest in pins['files'].items():assert sha(source/name)==digest,'Frozen operator source mismatch'
 OUT.mkdir(exist_ok=True);operators={};meta={};namespace=manifest['namespace']
 for task,(name,scale) in dict(sr4=('sr_avgpool',4),motion61=('deblur_motion',61),mask64=('box_inpainting',64)).items():
  op_seed=seed_for(namespace+'/operator/'+task)
  random.seed(op_seed);np.random.seed(op_seed%(2**32));torch.manual_seed(op_seed)
  op=get_degradation(name,Munch(channels=3,image_size=256,deg_scale=scale),'cpu')
  state={k:v.detach().contiguous().cpu() for k,v in vars(op).items() if isinstance(v,torch.Tensor)}
  if task=='motion61':state.update({'conv.'+k:v.detach().contiguous().cpu() for k,v in op.conv.state_dict().items()})
  if task=='mask64':state['mask']=op.get_mask((1,3,256,256)).contiguous()
  assert state,'Operator state must be explicit'
  path=OUT/'operators'/(task+'.safetensors');path.parent.mkdir(exist_ok=True);save_file(state,str(path))
  config=dict(task=task,source_name=name,deg_scale=scale,channels=3,resolution=256,noise_sigma=.03,
   operator_seed=op_seed,operator_state_sha256=sha(path),source=pins,device='cpu',dtype='float32',
   motion_intensity=.5 if task=='motion61' else None)
  config_path=path.with_suffix('.json');config_path.write_text(json.dumps(config,indent=2))
  operators[task]=op;meta[task]=dict(operator_config_sha256=sha(config_path),operator_state_sha256=sha(path),operator_seed=op_seed)
 def read_image(row):
  p=ROOT/'reconstructed'/row['path'];assert sha(p)==row['png_sha256']
  image=Image.open(p).convert('RGB');assert image.size==(256,256)
  return (to_tensor(image)*2-1).unsqueeze(0)
 # Discriminative timing/finite/shape check before processing the whole split.
 image=read_image(manifest['rows'][0]);timings={}
 with torch.no_grad():
  for task,op in operators.items():
   start=time.monotonic();value=op.A(image).contiguous();timings[task]=time.monotonic()-start
   assert torch.isfinite(value).all() and value.dtype==torch.float32
   assert value.numel()==(3*64*64 if task=='sr4' else 3*256*256)
 estimated=sum(timings.values())*len(manifest['rows'])
 print('MEASUREMENT_CPU_PROBE='+json.dumps(dict(seconds_per_image=timings,estimated_operator_seconds=estimated)),flush=True)
 assert estimated<1000,'CPU observation freeze exceeds this bounded preparation; preserve image-only split'
 start=time.monotonic()
 with torch.no_grad():
  for index,row in enumerate(manifest['rows']):
   image=read_image(row);observations={}
   for task,op in operators.items():
    clean=op.A(image).contiguous()
    noise_seed=seed_for(namespace+'/'+row['dataset']+'/'+row['original_id']+'/'+task)
    noise=torch.randn(clean.shape,dtype=torch.float32,generator=torch.Generator(device='cpu').manual_seed(noise_seed))
    value=(clean+.03*noise).contiguous();assert torch.isfinite(value).all()
    path=OUT/row['dataset']/row['split']/task/(row['original_id']+'.safetensors');path.parent.mkdir(parents=True,exist_ok=True)
    save_file({'measurement':value},str(path))
    observations[task]=dict(**meta[task],noise_seed=noise_seed,noise_sigma=.03,measurement_sha256=sha(path),
     tensor_payload_sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),shape=list(value.shape),dtype='float32',
     path=path.relative_to(OUT).as_posix())
   row['observations']=observations;row['observations_status']='CPU_FROZEN_RECONSTRUCTED_NOT_AUTHOR_EXACT'
   if index%100==0:print('MEASUREMENTS_FROZEN',index+1,'of',len(manifest['rows']),flush=True)
 manifest['measurement_tensors']='CPU_FROZEN_6300';manifest['measurement_rng']=dict(algorithm='torch2.7.1 CPU torch.randn per-image/per-task independent generator',
  rule='first8 bytes little-endian of SHA256(namespace/dataset/original_id/task)',torch=torch.__version__,numpy=np.__version__)
 manifest['measurement_source']=pins;manifest['full_launch_allowed']=False
 result=validate_inverse_split(manifest)
 (OUT/'measurement-manifest.json').write_text(json.dumps(manifest,indent=2))
 (OUT/'measurement-validation.json').write_text(json.dumps(dict(**result,measurements=6300,seconds=time.monotonic()-start,probe=timings),indent=2))
 archive=ROOT/'reconstructed-measurements.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
  for p in sorted(OUT.rglob('*')):
   if p.is_file():tar.add(p,arcname=p.relative_to(OUT))
 return publish([archive,OUT/'measurement-manifest.json',OUT/'measurement-validation.json'],'measurements')

if __name__=='__main__':prepare()
