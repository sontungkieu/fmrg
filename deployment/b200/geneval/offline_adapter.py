"""Path-only adapter for pinned GenEval; retains the OpenAI loader/QuickGELU.

No generation entrypoint. Network connections are denied while loading/evaluating.
"""
import contextlib,hashlib,json,os,runpy,socket,types
from pathlib import Path
from unittest.mock import patch
PINS={
 'r50':('mask2former_r50_lsj_8x2_50e_coco.pth','8e96e88b40be84c1432696ae3183ea0746e749ddc996efb40e55bfad8361cbea','mask2former_r50_lsj_8x2_50e_coco.py'),
 'swin_s':('mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth','743b7d99015f1224c6d57fd4b14d04b15cc8ec72ae7ee7831e7c71d8873b7a54','mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py')}
CLIP_SHA='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'

def bind_evaluator_globals(ns,detector,clip,transform,tokenizer,classes):
 ns.update(object_detector=detector,clip_model=clip,transform=transform,tokenizer=tokenizer,classnames=classes,
  THRESHOLD=.3,COUNTING_THRESHOLD=.9,MAX_OBJECTS=16,NMS_THRESHOLD=1.,POSITION_THRESHOLD=.1)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

@contextlib.contextmanager
def network_denied():
 original=socket.socket.connect;original_ex=socket.socket.connect_ex
 def connect(sock,address):
  if sock.family in (socket.AF_INET,socket.AF_INET6):raise RuntimeError('Evaluator network access denied')
  return original(sock,address)
 def connect_ex(sock,address):
  if sock.family in (socket.AF_INET,socket.AF_INET6):raise RuntimeError('Evaluator network access denied')
  return original_ex(sock,address)
 with patch.object(socket.socket,'connect',connect),patch.object(socket.socket,'connect_ex',connect_ex):yield

@contextlib.contextmanager
def load(assets,profile,prefix='/opt/fmrg-geneval'):
 import torch,open_clip,importlib
 checkpoint_module=importlib.import_module('mmcv.runner.checkpoint')
 assets=Path(assets).resolve();source=Path(prefix)/'source/sources'
 filename,digest,config=PINS[profile];checkpoint=assets/'geneval'/filename
 assert sha(checkpoint)==digest and sha(assets/'geneval/ViT-L-14.pt')==CLIP_SHA
 script=source/'geneval/evaluation/evaluate_images.py'
 expected=json.loads((Path(__file__).parent/'source-pins.json').read_text())
 assert sha(script)==expected['evaluate_images_sha256']
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 original_load=torch.load;original_clip=open_clip.create_model_and_transforms
 original_state=checkpoint_module.load_state_dict;key_audits=[]
 def checked_state(model,state_dict,*args,**kwargs):
  expected=model.state_dict()
  missing=[k for k in expected if k not in state_dict and not k.endswith('.num_batches_tracked')]
  unexpected=[k for k in state_dict if k not in expected and not k.endswith('.num_batches_tracked')]
  bad_shapes=[k for k in expected if k in state_dict and expected[k].shape!=state_dict[k].shape]
  assert not missing and not unexpected and not bad_shapes,'Detector checkpoint/model key or shape mismatch: '+repr((missing,unexpected,bad_shapes))
  key_audits.append(dict(model_keys=len(expected),checkpoint_keys=len(state_dict),missing=missing,unexpected=unexpected,bad_shapes=bad_shapes))
  return original_state(model,state_dict,*args,**kwargs)
 def trusted_load(path,*args,**kwargs):
  if isinstance(path,(str,Path)) and Path(path).resolve()==checkpoint:
   kwargs.setdefault('weights_only',False) # Only the exact hash-verified legacy checkpoint.
  return original_load(path,*args,**kwargs)
 def local_clip(name,*args,**kwargs):
  assert name=='ViT-L-14' and kwargs.get('pretrained')=='openai'
  # Keep pretrained='openai': changing it to a path bypasses the OpenAI model
  # reconstruction and can silently replace QuickGELU with GELU.
  kwargs['cache_dir']=str(assets/'geneval')
  return original_clip(name,*args,**kwargs)
 with network_denied(),patch.object(torch,'load',trusted_load),patch.object(open_clip,'create_model_and_transforms',local_clip),patch.object(checkpoint_module,'load_state_dict',checked_state):
  ns=runpy.run_path(str(script),run_name='fmrg_pinned_geneval')
  ns=ns['load_models'].__globals__
  args=types.SimpleNamespace(model_config=str(source/'mmdetection/configs/mask2former'/config),
   model_path=str(assets/'geneval'),options={'model':filename[:-4]})
  ns['args']=args
  detector,(clip,transform,tokenizer),classes=ns['load_models'](args)
  assert len(key_audits)==1,'Detector state load was not audited'
  assert len(classes)==80
  bind_evaluator_globals(ns,detector,clip,transform,tokenizer,classes)
  quick=sum(type(m).__name__=='QuickGELU' for m in clip.modules());assert quick>0
  yield ns,dict(profile=profile,checkpoint_sha256=digest,clip_sha256=CLIP_SHA,quickgelu_modules=quick,
   evaluator_sha256=sha(script),state_keys=key_audits[0],resolved_config_sha256=hashlib.sha256(detector.cfg.pretty_text.encode()).hexdigest(),
   network='TCP_DENIED',claim='TECHNICAL_ONLY_NOT_PAPER_SCORE',full_launch_allowed=False)
