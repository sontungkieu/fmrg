"""512px official FMRG sampler with pinned CLIP-only reward."""
import itertools,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
PROMPTS=["A watercolor painting of a red fox in a snowy forest.","A studio photograph of a blue ceramic teapot on a wooden table."]
def run(out,deadline=1000):
 import imagereward_compat
 from fluxfm_sampler_reward import FluxFlowMapSampler,create_clip_preprocess
 from transformers import CLIPModel,AutoProcessor
 out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 data=Path(os.environ.get("FMRG_DATA_ROOT","/data"));a=json.loads((data/"assets.json").read_text());ra=json.loads((data/"pilot-assets.json").read_text())
 s=FluxFlowMapSampler(model_id=a["base_path"],lora_path=a["lora_512"],prompt=PROMPTS[0],additional_prompts=PROMPTS)
 class Reward:
  def __init__(self):
   self.model=CLIPModel.from_pretrained(ra["clip_path"],local_files_only=True).to("cuda").eval().requires_grad_(False)
   self.model.gradient_checkpointing_enable();self.processor=AutoProcessor.from_pretrained(ra["clip_path"],local_files_only=True)
   self.preprocess=create_clip_preprocess(224);self.calls=0
  def __call__(self,image,prompt):
   self.calls+=1;im=self.model.get_image_features(pixel_values=self.preprocess(image.float()))
   tokens=self.processor(text=prompt,return_tensors="pt",padding=True,max_length=77,truncation=True).to("cuda")
   text=self.model.get_text_features(**tokens);score=(F.normalize(im,dim=-1)@F.normalize(text,dim=-1).T).mean()*self.model.logit_scale.exp()
   loss=(100-score)*.01;return loss,dict(clip=float((100-score).detach()),total=float(loss.detach()))
 reward=Reward();load=time.perf_counter()-start;counts=dict(transformer_forward=0,block_forward=0,autograd_calls=0)
 def hook(key):
  def inner(*args):counts[key]+=1
  return inner
 s.pipeline.transformer.register_forward_pre_hook(hook("transformer_forward"))
 for group in ["transformer_blocks","single_transformer_blocks"]:
  for b in getattr(s.pipeline.transformer,group,[]):b.register_forward_pre_hook(hook("block_forward"))
 x=torch.rand(1,3,512,512,device="cuda",requires_grad=True);loss,_=reward(x,PROMPTS[0]);g,=torch.autograd.grad(loss,x)
 assert torch.isfinite(g).all() and g.norm()>0
 canary=dict(loss=float(loss.detach()),gradient_norm=float(g.norm()),finite=True);del x,loss,g
 original_grad=torch.autograd.grad
 def checked_grad(*args,**kw):
  result=original_grad(*args,**kw);counts["autograd_calls"]+=1
  assert all(g is not None and torch.isfinite(g).all() for g in result),"nonfinite reward gradient"
  return result
 torch.autograd.grad=checked_grad
 rows=[];features=[];status="COMPLETE"
 for number,(pi,seed,n,arm) in enumerate(itertools.product(range(2),[0,1],[6,12],["unguided","J","J_K3"])):
  if time.perf_counter()-start>deadline-60:status="PARTIAL_DEADLINE";break
  s.set_prompt_embeddings(PROMPTS[pi]);counts.update(transformer_forward=0,block_forward=0,autograd_calls=0);reward.calls=0
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
  image=s.sample_reward_guided(reward_model=reward,num_steps=n,guidance_scale=3.5,step_size=0. if arm=="unguided" else 1.,
   img_shape=(512,512),grad_mode="jac",sample_mode="flow_map1",seed=seed,num_optim_iters=1,lmbda=0.,use_adam=True,
   early_stop=0,noise_mode="deterministic",grad_norm_mode="normalize",enable_callback=False,grad_checkpointing=True,
   warmup_steps=1 if arm=="J_K3" else 0,warmup_particles=3 if arm=="J_K3" else 1,unguided_steps=0)
  torch.cuda.synchronize();assert torch.isfinite(image).all()
  unit=((image.detach().float()+1)/2).clamp(0,1)
  with torch.no_grad():
   loss,_=reward(unit,PROMPTS[pi]);feat=F.normalize(reward.model.get_image_features(pixel_values=reward.preprocess(unit)),dim=-1)
  filename=f"{number:03d}-p{pi}-s{seed}-n{n}-{arm}.png"
  Image.fromarray(np.rint(unit[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save(out/filename)
  row=dict(index=number,prompt_index=pi,prompt=PROMPTS[pi],seed=seed,main_grid_nfe=n,arm=arm,clip_loss=float(loss),
   clip_logit=100-float(loss)/.01,seconds=time.perf_counter()-began,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
   counts=counts.copy(),reward_evaluations_including_final=reward.calls,image_file=filename)
  rows.append(row);features.append(feat.float().cpu().numpy()[0])
  with (out/"rows.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
  print("FMRG_REWARD_CELL_JSON="+json.dumps(row),flush=True);del image,unit,feat
 if features:np.save(out/"clip-image-features.npy",np.stack(features))
 summary=dict(status=status,expected=24,completed=len(rows),load_seconds=load,canary=canary,gpu=torch.cuda.get_device_name(),
  reward="CLIP-only weight0.01; not paper ensemble",scope="two-prompt reward/runtime pilot; not GenEval reproduction")
 (out/"summary.json").write_text(json.dumps(summary,indent=2));print("FMRG_REWARD_JSON="+json.dumps(summary),flush=True)
 assert len(rows)==24,status
 return summary
if __name__=="__main__":run(sys.argv[1],float(sys.argv[2]) if len(sys.argv)>2 else 1000)
