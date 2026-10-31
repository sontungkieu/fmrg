"""Publish completed technical preparation with immutable references, CPU only."""
import hashlib,io,json,os,tarfile
from pathlib import Path
from prepare_additions import api,sha
ROOT=Path('/data/geneval-20261101');REPO='codemaivanngu/fmrg-b200-portable';PREFIX='full-paper/20261101'

def publish():
 from huggingface_hub import CommitOperationAdd,hf_hub_download
 client=api();here=Path(__file__).resolve().parent
 build=json.loads((ROOT/'build.json').read_text());preflight=json.loads((ROOT/'preflight.json').read_text());canary=json.loads((ROOT/'canary-a3.json').read_text())
 measurement_ready=(ROOT/'measurements.json').exists()
 assert preflight['status']=='PASS' and canary['status']=='PASS' and len(canary['profiles'])==2
 for name,digest in canary['code_sha256'].items():assert sha(here/name)==digest,'Publish only the canary-tested evaluator code'
 assert build['overlay']['sha256']=='16126b76c1540e464cdef993f0e7551432f7203d8de000f470244edeabd0b7f1'
 output=ROOT/'public-update';output.mkdir(exist_ok=True)
 safe_build=json.loads(json.dumps(build))
 for k in ['overlay','wheel']:safe_build[k].pop('path',None)
 safe_build.update(builder_base='nvidia/cuda@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7',
  qualification='Modal B200 technical only; not company qualification or author-evaluator numerical equivalence')
 (output/'build.json').write_text(json.dumps(safe_build,indent=2))
 safe_pre=json.loads(json.dumps(preflight));safe_pre['overlay'].pop('path',None)
 (output/'preflight.json').write_text(json.dumps(safe_pre,indent=2))
 (output/'canary.json').write_text(json.dumps(canary,indent=2))
 if (ROOT/'canary-a2.json').exists():(output/'canary-a2-failed.json').write_bytes((ROOT/'canary-a2.json').read_bytes())
 update=dict(protocol='fmrg-preparation-contract-update-20261101-v1',full_launch_allowed=False,
  parent_contract_url='https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/77f4ae1dee744fbfa906eaf3f9cd00b66a344594/full-paper/20261029/toolkit/CONTRACT.json',
  parent_contract_sha256='1b02425b2c9d6c447bc7350f2d168c35af6b2e06bf6f904d43ca745eb098b3cb',
  scientific_source='be485ba40f1d1c163b95ad61112aca6baa13ed22',
  resolved=['GenEval native sm100 build and Modal B200 technical canary, R50 primary/Swin-S control, offline OpenAI CLIP QuickGELU',
   'Skywork authorized download and pinned public weights mirror',
   'User-authorized reconstructed AFHQ-Dog/FFHQ image splits,2100 images,original/source/preprocessing hashes and FFHQ attribution'],
  pending=['Company GenEval canary and GPU UUID/allocation receipt','Company four-reward ensemble canary receipt',
   'Author-exact splits remain unavailable; reconstructed is a separate claim',
   'Operator/kernel/noise seeds and measurement tensor freeze','Paper-formula sampler adapter',
   'Skywork paper Yes/No reward differentiable adapter and renoiseconstantc','Style16refs+48pairedprompts/configs',
   'Tuning objective/tie-break/full baseline environments and budgets','Complete compute instrumentation and full allocation/deadline/authorization'],
  scoring='strict_evidence rejects missing/duplicate/failed cells; complete six-category aggregation only',
  methodological_limits=['Adapted Torch2.7/cu128 evaluator is not the original Torch1.12/cu113 environment.',
   'Synthetic/native/demo canaries do not measure efficacy or establish evaluator score parity.',
   'No FLUX generation, full benchmark or company-node operation was performed in this update.'])
 if measurement_ready:
  frozen=json.loads((ROOT/'measurements/measurement-validation.json').read_text())
  assert frozen['measurement_hashes_verified'] is True and frozen['measurements']==6300
  update['resolved'].append('6300 reconstructed CPU observations frozen with per-image/task seeds, source/operator state hashes and tensor SHA256')
  update['pending']=[s for s in update['pending'] if not s.startswith('Operator/kernel/noise')]
  update['pending'].append('Runner adapter/canary to consume frozen measurement/operator states; original CLI regenerates observations')
  update['measurement_claim']='RECONSTRUCTED_CPU_OBSERVATIONS_NOT_AUTHOR_EXACT'
 (output/'CONTRACT_UPDATE.json').write_text(json.dumps(update,indent=2))
 names=['canary.py','offline_adapter.py','source-pins.json','restore_overlay.py','download_publication.py',
        'strict_evidence.py','tuning-plan-draft.json','reconstructed-plan.json']
 if measurement_ready:names.append('measurement-source-pins.json')
 toolkit=ROOT/'geneval-toolkit-20261101.tar.gz'
 with tarfile.open(toolkit,'w:gz') as tar:
  for name in names:
   b=(here/name).read_bytes();m=tarfile.TarInfo(name);m.size=len(b);m.mode=0o644;m.mtime=0;tar.addfile(m,io.BytesIO(b))
  for name in ['CONTRACT_UPDATE.json','canary.json','build.json','preflight.json']:
   b=(output/name).read_bytes();m=tarfile.TarInfo(name);m.size=len(b);m.mode=0o644;m.mtime=0;tar.addfile(m,io.BytesIO(b))
 operations=[];rows=[]
 def add(path,file,group):
  file=Path(file);rows.append(dict(group=group,filename=file.name,path=path,bytes=file.stat().st_size,sha256=sha(file)))
  operations.append(CommitOperationAdd(path_in_repo=path,path_or_fileobj=str(file)))
 add(PREFIX+'/runtime/'+Path(build['overlay']['path']).name,build['overlay']['path'],'runtime')
 add(PREFIX+'/runtime/'+Path(build['wheel']['path']).name,build['wheel']['path'],'runtime')
 for name in ['build.json','preflight.json','canary.json','canary-a2-failed.json']:
  if (output/name).exists():add(PREFIX+'/runtime/'+name,output/name,'runtime')
 add(PREFIX+'/CONTRACT_UPDATE.json',output/'CONTRACT_UPDATE.json','toolkit')
 add(PREFIX+'/'+toolkit.name,toolkit,'toolkit')
 for name in names:add(PREFIX+'/toolkit/'+name,here/name,'source')
 for name in ['test_preparation.py','build_overlay.py']:
  add(PREFIX+'/toolkit/'+name,here/name,'source')
 if measurement_ready:add(PREFIX+'/toolkit/freeze_measurements.py',here/'freeze_measurements.py','source')
 first=client.create_commit(REPO,repo_type='dataset',operations=operations,
  commit_message='Publish GenEval cu128 sm100 overlay and B200 technical qualification; no full benchmark').oid
 for row in rows:row['url']=f'https://huggingface.co/datasets/{REPO}/resolve/{first}/'+row['path']
 for kind in ['vlm','reconstructed']+(['measurements'] if measurement_ready else []):
  existing=json.loads((ROOT/(kind+'.json')).read_text())
  rows.extend(dict(group=kind,filename=Path(f['path']).name,**f) for f in existing['files'])
 index=dict(protocol='fmrg-public-preparation-20261101-v1',full_launch_allowed=False,files=rows,
  runtime_status='MODAL_B200_TECHNICAL_PASS_COMPANY_PENDING',dataset_claim='reconstructed_not_author_exact',
  vlm_status='WEIGHTS_STAGED_PAPER_ADAPTER_PENDING')
 if measurement_ready:index['measurement_status']='CPU_FROZEN_RECONSTRUCTED_6300'
 publication=output/'publication.json';publication.write_text(json.dumps(index,indent=2))
 second=client.create_commit(REPO,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=PREFIX+'/publication.json',path_or_fileobj=str(publication))],
  commit_message='Pin evaluator runtime, VLM and reconstructed split downloads in one manifest').oid
 publication_url=f'https://huggingface.co/datasets/{REPO}/resolve/{second}/{PREFIX}/publication.json'
 text=(here/'HANDOFF_TEMPLATE.md').read_text()
 if measurement_ready:
  text=text.replace('Không phải author-exact; measurement tensors chưa sinh.','Không phải author-exact. Đã freeze thêm6300CPU observation tensors theo source operators và per-image/task seeds; không có neural generation.')
  text=text.replace('Reconstructed archive chỉ có ảnh/metadata, không measurement tensors. Freeze operator/kernel/noise-seed/tensor hashes trước tuning.',
   'Reconstructed image archive và measurement archive là hai gói riêng. Group `measurements` chứa6300safetensors, operator-state/config và manifest đã verify đủ2100ảnh×3tasks. Seeds dùng SHA256(namespace/dataset/original_id/task), first8bytes little-endian; Torch2.7.1 CPU torch.randn, sigma.03, input[-1,1]. Một operator/kernel cố định mỗi task cho cả hai datasets. Đây là reconstructed inputs, không khẳng định trùng author kernel/noise. Trước tuning phải verify hashes và qualify runner adapter nạp đúng frozen measurements/operator states; released CLI hiện tự tạo observations.')
  text=text.replace('VLM differentiable Yes/No adapter/renoiseconstantc, complete tuning objective/grid/baselines, instrumentation và full allocation/deadline chưa chốt.',
   'VLM differentiable Yes/No adapter/renoiseconstantc, runner nạp frozen observations, complete tuning objective/grid/baselines, instrumentation và full allocation/deadline chưa chốt.')
  text += '\n## Frozen observation download\n\nDùng lại `download_publication.py` với publication manifest/SHA ở trên và `--group measurements` để tải archive/manifest/validation (khoảng3.2GB, xem bytes thực trong manifest). Chưa cần gói này để chạy GenEval canary. Không tự gọi `freeze_measurements.py` trên company hoặc tái tạo y trong generator rồi gọi là dùng frozen inputs. Sau verify/extract, giữ file measurement theo manifest và kiểm dtype/shape/tensor SHA trước khi runner được qualify.\n'
 replacements=dict(PUBLICATION_URL=publication_url,PUBLICATION_SHA=sha(publication),
  TOOLKIT_URL=f'https://huggingface.co/datasets/{REPO}/resolve/{first}/{PREFIX}/'+toolkit.name,TOOLKIT_SHA=sha(toolkit),
  CONTRACT_URL=f'https://huggingface.co/datasets/{REPO}/resolve/{first}/{PREFIX}/CONTRACT_UPDATE.json')
 for k,v in replacements.items():text=text.replace('__'+k+'__',v)
 assert '__PUBLICATION' not in text and '__TOOLKIT' not in text
 handoff=output/'PREPARATION_HANDOFF.md';handoff.write_bytes(text.encode())
 old=client.repo_info(REPO,repo_type='dataset').sha
 banner='\n## Preparation update — 2026-11-01\n\nRead [the new GenEval/VLM/reconstructed-data handoff](full-paper/20261101/PREPARATION_HANDOFF.md). GenEval Modal B200 technical canary passed; company qualification remains pending. Full launch remains disabled. Older immutable core artifacts are preserved.\n\n'
 pointers=[]
 for name in ['README.md','HANDOFF.md']:
  prior=Path(hf_hub_download(REPO,name,repo_type='dataset',revision=old)).read_text()
  prior=prior.replace(banner,'')
  if prior.startswith('---\n'):
   end=prior.index('\n---',4)+4;value=prior[:end]+'\n'+banner+prior[end:]
  else:value=banner+prior
  dest=output/name;dest.write_text(value);pointers.append(CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(dest)))
 third=client.create_commit(REPO,repo_type='dataset',operations=pointers+[CommitOperationAdd(path_in_repo=PREFIX+'/PREPARATION_HANDOFF.md',path_or_fileobj=str(handoff))],
  commit_message='Update company handoff with qualified GenEval overlay and reconstructed split boundaries').oid
 siblings={f.rfilename:f for f in client.repo_info(REPO,repo_type='dataset',revision=third,files_metadata=True).siblings}
 for row in rows:
  f=siblings[row['path']];assert f.size==row['bytes']
  if f.lfs:assert f.lfs.sha256==row['sha256']
 result=dict(handoff_url=f'https://huggingface.co/datasets/{REPO}/resolve/{third}/{PREFIX}/PREPARATION_HANDOFF.md',handoff_sha256=sha(handoff),
  publication_url=publication_url,publication_sha256=sha(publication),toolkit_url=replacements['TOOLKIT_URL'],toolkit_sha256=sha(toolkit),
  contract_url=replacements['CONTRACT_URL'],runtime_commit=first,publication_commit=second,handoff_commit=third,full_launch_allowed=False)
 (ROOT/'publish.json').write_text(json.dumps(result,indent=2));print('FMRG_PREPARATION_PUBLISHED='+json.dumps(result),flush=True)
 return result

if __name__=='__main__':publish()
