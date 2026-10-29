"""Path-only reward loader overlay for the pinned FMRG classes.

Original reward forward functions/preprocessing are reused. HPS constructs the
architecture without downloading base weights; strict full-state loading is
mandatory. This is not a GPU backward/quality certificate.
"""
import json,os
from pathlib import Path
def load_ensemble(assets,core_assets,device='cuda',dtype=None):
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_XET='1')
 import torch
 import imagereward_compat
 import fluxfm_sampler_reward as source
 from transformers import AutoModel,CLIPModel,AutoProcessor,BertTokenizer
 from unittest.mock import patch
 root=Path(assets);core=Path(core_assets)
 from verify_assets import verify
 verify(root,['hps','imagereward','pickscore','bert'])
 if dtype is None:dtype=torch.float32
 for component in ['hps','imagereward','pickscore','bert']:
  assert (root/component).is_dir(),'Run hash-verified downloader first: '+component
 clip_path=Path(json.loads((core/'pilot-assets.json').read_text())['clip_path'])
 tokenizer=AutoProcessor.from_pretrained(clip_path,local_files_only=True)
 ensemble=object.__new__(source.RewardEnsemble)
 ensemble.device=device
 ensemble.dtype=dtype
 ensemble.preprocess=source.create_clip_preprocess(224)
 ensemble.weights={'hps':5.,'imagereward':1.,'pickscore':.05,'clip':.01};ensemble.models={}
 from hpsv2.src.open_clip import create_model,get_tokenizer
 h=object.__new__(source.HPSReward);torch.nn.Module.__init__(h)
 h.model=create_model('ViT-H-14',pretrained=None,precision=dtype,device=device)
 checkpoint=torch.load(root/'hps/HPS_v2.1_compressed.pt',map_location='cpu',weights_only=True)
 h.model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
 h.model=h.model.to(device,dtype=dtype).eval().requires_grad_(False);h.model.set_grad_checkpointing(True)
 h.tokenizer=get_tokenizer('ViT-H-14');h.device=device;ensemble.models['hps']=h
 original=BertTokenizer.from_pretrained
 def local_bert(*args,**kwargs):
  kwargs['local_files_only']=True
  return original(str(root/'bert'),**kwargs)
 import ImageReward as RM
 ir=object.__new__(source.ImageRewardReward);torch.nn.Module.__init__(ir)
 original_load_state=torch.nn.Module.load_state_dict;load_checks=[]
 def checked_state(model,*args,**kwargs):
  result=original_load_state(model,*args,**kwargs)
  if type(model).__name__=='ImageReward':
   assert not result.missing_keys and not result.unexpected_keys,'ImageReward checkpoint key mismatch: '+repr(result)
   load_checks.append(True)
  return result
 with patch.object(BertTokenizer,'from_pretrained',side_effect=local_bert),patch.object(torch.nn.Module,'load_state_dict',checked_state):
  ir.model=RM.load(str(root/'imagereward/ImageReward.pt'),device=device,med_config=str(root/'imagereward/med_config.json'))
 assert load_checks,'ImageReward state loading was not audited'
 ir.model=ir.model.to(device=device,dtype=dtype).eval().requires_grad_(False)
 ir.device=device;ir.dtype=dtype;ensemble.models['imagereward']=ir
 c=object.__new__(source.CLIPReward);torch.nn.Module.__init__(c)
 c.model=CLIPModel.from_pretrained(clip_path,local_files_only=True).to(device,dtype=dtype).eval().requires_grad_(False)
 c.model.gradient_checkpointing_enable();c.tokenizer=tokenizer;c.device=device;ensemble.models['clip']=c
 ps=object.__new__(source.PickScoreReward);torch.nn.Module.__init__(ps)
 ps.model=AutoModel.from_pretrained(root/'pickscore',local_files_only=True).to(device,dtype=dtype).eval().requires_grad_(False)
 ps.model._set_gradient_checkpointing(True);ps.tokenizer=tokenizer;ps.device=device;ensemble.models['pickscore']=ps
 return ensemble
