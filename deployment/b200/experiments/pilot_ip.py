"""I/J paired screening using the untouched official sampler."""
import hashlib,itertools,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity
def instrument(s):
 state={"step":-1,"cutoff":None,"counts":{}}
 names=["predict_forward","transformer_forward","block_forward","vae_decode","autograd_calls"]
 p0,e0,j0,d0=s.predict_vector,s.fmrg_gradient_step_euclidean,s.fmrg_gradient_step_jacobian,s.decode
 def skip():return state["cutoff"] is not None and state["step"]>=state["cutoff"]
 def predict(*args,**kw):
  state["step"]+=1;state["counts"]["predict_forward"]+=1
  if skip():
   with torch.no_grad():return p0(*args,**kw)
  return p0(*args,**kw)
 def e(z,*args,**kw):return (z.detach(),0.) if skip() else e0(z,*args,**kw)
 def j(z,*args,**kw):return (torch.zeros_like(z),0.,z.detach()) if skip() else j0(z,*args,**kw)
 def decode(*args,**kw):
  state["counts"]["vae_decode"]+=1;return d0(*args,**kw)
 s.predict_vector,s.fmrg_gradient_step_euclidean,s.fmrg_gradient_step_jacobian,s.decode=predict,e,j,decode
 def hook(key):
  def count(*args):state["counts"][key]+=1
  return count
 s.pipeline.transformer.register_forward_pre_hook(hook("transformer_forward"))
 for group in ["transformer_blocks","single_transformer_blocks"]:
  for block in getattr(s.pipeline.transformer,group,[]):block.register_forward_pre_hook(hook("block_forward"))
 state["names"]=names;return state
def run(kind,out,deadline=1000):
 from fluxfm_sampler_ip import FluxFlowMapSampler
 start=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=True)
 data=Path(os.environ.get("FMRG_DATA_ROOT","/data"));assets=json.loads((data/"assets.json").read_text())
 s=FluxFlowMapSampler(model_id=assets["base_path"],lora_path=assets["lora_256"],prompt="a photo of a dog")
 s.pipeline.transformer.enable_gradient_checkpointing();load=time.perf_counter()-start;state=instrument(s)
 original_grad=torch.autograd.grad
 def checked_grad(*args,**kw):
  grads=original_grad(*args,**kw);state["counts"]["autograd_calls"]+=1
  assert all(g is not None and torch.isfinite(g).all() for g in grads),"nonfinite gradient"
  return grads
 torch.autograd.grad=checked_grad
 class Operator:
  def A(self,x):return F.avg_pool2d(x,4)
 op=Operator();images=["corgi.png","afhq_dog_1.png","afhq_dog_2.png"];cases=[]
 if kind=="I":
  for im,seed,n,arm in itertools.product(images,[0,1],[30,3,6,12],["unguided","E","Euler_E"]):cases.append((im,seed,n,arm,0. if arm=="unguided" else 10.))
 else:
  for im,seed,n,strength,arm in itertools.product(images,[0,1],[6,12],[.1,1.],["E_all","J_all","E_half","J_half"]):cases.append((im,seed,n,arm,strength))
 rows=[];status="COMPLETE";measurements=data/"benchmark-inputs";measurements.mkdir(exist_ok=True)
 try:
  for number,(name,seed,n,arm,strength) in enumerate(cases):
   if time.perf_counter()-start>deadline-45:status="PARTIAL_DEADLINE";break
   source_root=Path(os.environ.get("FMRG_SOURCE_ROOT","/opt/fmrg-source" if Path("/opt/fmrg-source").exists() else "/opt/fmrg/source"));source=source_root/"data/inverse_examples"/name
   image=Image.open(source).convert("RGB").resize((256,256),Image.Resampling.BICUBIC)
   gt=torch.from_numpy(np.array(image).copy()).permute(2,0,1).float().unsqueeze(0)/127.5-1
   gen=torch.Generator().manual_seed(1000+images.index(name)*10+seed)
   measurement=op.A(gt)+.03*torch.randn((1,3,64,64),generator=gen)
   mp=measurements/(source.stem+f"-seed{seed}.pt")
   if not mp.exists():torch.save(dict(measurement=measurement,gt=gt),mp)
   else:
    frozen=torch.load(mp,weights_only=True);assert torch.equal(frozen["measurement"],measurement) and torch.equal(frozen["gt"],gt)
   state["step"]=-1;state["cutoff"]=n//2 if arm.endswith("half") else None;state["counts"]={k:0 for k in state["names"]}
   torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
   result=s.sample_inverse_problem(measurement.to("cuda"),op,"sr",num_steps=n,guidance_scale=3.5,
    step_size=strength,img_shape=(256,256),grad_mode="jac" if arm.startswith("J") else "euc",
    sample_mode="flow_matching" if arm=="Euler_E" else "flow_map1",seed=seed,num_optim_iters=3 if kind=="I" else 1,
    loss_mode="pixel",loss_func="norm",normalize_grad=kind=="J",early_stop=0,enable_callback=False)
   torch.cuda.synchronize();elapsed=time.perf_counter()-began
   assert torch.isfinite(result).all();assert state["counts"]["predict_forward"]==n
   pred=((result.detach().float().cpu()+1)/2).clamp(0,1);label=(gt+1)/2;mse=float((pred-label).square().mean())
   arr=pred[0].permute(1,2,0).numpy();lab=label[0].permute(1,2,0).numpy()
   filename=f"{number:03d}-{source.stem}-s{seed}-n{n}-{arm}-l{strength}.png"
   Image.fromarray(np.rint(arr*255).astype(np.uint8)).save(out/filename)
   row=dict(index=number,image=name,seed=seed,nfe=n,arm=arm,strength=strength,cutoff=state["cutoff"],seconds=elapsed,
    psnr=-10*math.log10(max(mse,1e-20)),ssim=float(structural_similarity(lab,arr,data_range=1.,channel_axis=-1)),
    measurement_rmse=float((op.A(result.float())-measurement.to("cuda")).square().mean().sqrt()),
    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
    counts=state["counts"].copy(),measurement_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),image_file=filename)
   rows.append(row)
   with (out/"rows.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
   print("FMRG_CELL_JSON="+json.dumps(row),flush=True);del result,pred
   if kind=="I" and len(rows)==3 and time.perf_counter()-start+max(r["seconds"] for r in rows)*(len(cases)-3)>deadline:
    status="PARTIAL_FORECAST";break
 except Exception as error:
  status="FAILED_"+type(error).__name__;raise
 finally:
  torch.autograd.grad=original_grad
  summary=dict(kind=kind,status=status,expected=len(cases),completed=len(rows),load_seconds=load,total_seconds=time.perf_counter()-start,
   gpu=torch.cuda.get_device_name(),torch=torch.__version__,scope="3-image paired screening; not full paper benchmark",
   early_stop_contract="fixed grid; guidance zero in second half; same total NFE",
   mean_psnr_by_arm={a:float(np.mean([r["psnr"] for r in rows if r["arm"]==a])) for a in sorted({r["arm"] for r in rows})})
  (out/"summary.json").write_text(json.dumps(summary,indent=2));print("FMRG_PILOT_JSON="+json.dumps(summary),flush=True)
 assert len(rows)==len(cases),summary["status"]
 return summary
if __name__=="__main__":run(sys.argv[1],sys.argv[2],float(sys.argv[3]) if len(sys.argv)>3 else 1000)
