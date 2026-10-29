"""CPU control-flow audit of pinned source with a synthetic network/reward.
No model weights, generated images, scientific scores or GPU execution.
"""
import json,types,inspect,hashlib
from pathlib import Path

def run(out):
 import torch
 from diffusers import FlowMatchEulerDiscreteScheduler
 from diffusers.pipelines.flux.pipeline_flux import calculate_shift,retrieve_timesteps
 from fluxfm_sampler_reward import FluxFlowMapSampler
 base=json.loads(Path('/data/assets.json').read_text())
 cfg=json.loads((Path(base['base_path'])/'scheduler/scheduler_config.json').read_text())
 from diffusers import FluxPipeline
 init_source=inspect.getsource(FluxPipeline.__init__)
 assert '2 ** (len(self.vae.config.block_out_channels) - 1)' in init_source,'Review changed scale formula'
 vae_config=json.loads((Path(base['base_path'])/'vae/config.json').read_text())
 vae_scale_factor=2**(len(vae_config['block_out_channels'])-1)
 assert vae_scale_factor==8,'Recompute sequence length and trace for changed VAE'
 profiles=[dict(name='release_J_tag21',grad='jac',mode='flow_map1',N=32,k=8,w=4,K=3,eta=3,nopt=1),
           dict(name='release_J_tag100',grad='jac',mode='flow_map2',N=208,k=26,w=11,K=3,eta=5,nopt=1),
           dict(name='release_E_tag99',grad='euc',mode='flow_map2',N=200,k=25,w=11,K=3,eta=.7,nopt=3)]
 results=[]
 for p in profiles:
  s=object.__new__(FluxFlowMapSampler)
  s.device='cpu';s.dtype=torch.float32;s.vae_scale_factor=vae_scale_factor;s.prompt_text='synthetic audit'
  s.cached_prompt_embeds=torch.zeros(1,1,1);s.cached_pooled_prompt_embeds=torch.zeros(1,1)
  s.pipeline=types.SimpleNamespace(scheduler=FlowMatchEulerDiscreteScheduler.from_config(cfg))
  calls=[]
  def predict(z,t_cur,**kw):
   calls.append(dict(sigma=float(t_cur),target=float(kw.get('t_next') or 0)))
   return .01*z+.1
  s.predict_vector=predict;s.decode=lambda z:z[:,:3]
  def reward(img,prompt):
   loss=img.square().mean();return loss,dict(total=float(loss.detach()))
  torch.manual_seed(0)
  result=s.sample_reward_guided(reward_model=reward,num_steps=p['N'],step_size=p['eta'],img_shape=(512,512),
    grad_mode=p['grad'],sample_mode=p['mode'],seed=0,num_optim_iters=p['nopt'],use_adam=True,
    early_stop=p['k'],grad_norm_mode='normalize' if p['grad']=='jac' else 'none',enable_callback=False,
    grad_checkpointing=False,warmup_steps=p['w'],warmup_particles=p['K'],unguided_steps=2)
  assert torch.isfinite(result).all()
  scheduler=FlowMatchEulerDiscreteScheduler.from_config(cfg)
  mu=calculate_shift(1024,cfg.get('base_image_seq_len',256),cfg.get('max_image_seq_len',4096),cfg.get('base_shift',.5),cfg.get('max_shift',1.15))
  import numpy as np
  times,_=retrieve_timesteps(scheduler,p['N'],'cpu',sigmas=np.linspace(1,1/p['N'],p['N']),mu=mu)
  sigma=(times/1000).tolist()+[0.]
  results.append(dict(profile=p,actual_predict_vector_calls=len(calls),sigma_grid=sigma,
    last_guided_boundary_index=p['k']-1,effective_data_time_stop=1-sigma[p['k']-1],calls=calls))
 record=dict(kind='CPU_CONTROL_FLOW_ONLY_SYNTHETIC_NETWORK',scientific_inference=False,
   source='be485ba40f1d1c163b95ad61112aca6baa13ed22',scheduler_config=cfg,
   vae_scale_factor=vae_scale_factor,pipeline_init_sha256=hashlib.sha256(init_source.encode()).hexdigest(),results=results)
 Path(out).write_text(json.dumps(record,indent=2))
 print('FMRG_SCHEDULE_TRACE='+json.dumps([dict(name=r['profile']['name'],actual_calls=r['actual_predict_vector_calls'],effective_tstop=r['effective_data_time_stop']) for r in results]),flush=True)
 return record
