"""Remote CPU asset staging only, including explicitly reconstructed splits."""
import concurrent.futures,hashlib,io,json,os,tarfile,time
from pathlib import Path
ROOT=Path('/data/geneval-20261101');ROOT.mkdir(exist_ok=True)
REPO='codemaivanngu/fmrg-b200-portable';PREFIX='full-paper/20261101'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def api():
 from huggingface_hub import HfApi
 client=HfApi(token=os.environ['HF_TOKEN'])
 assert client.whoami()['name']=='codemaivanngu' and not client.repo_info(REPO,repo_type='dataset').private
 return client

def publish(files,kind):
 from huggingface_hub import CommitOperationAdd
 client=api();rows=[dict(path=PREFIX+'/'+kind+'/'+p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in files]
 commit=client.create_commit(REPO,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=r['path'],path_or_fileobj=str(p)) for r,p in zip(rows,files)],
  num_threads=3,commit_message='Stage '+kind+' assets only; reconstructed data explicitly separated; no full launch').oid
 siblings={f.rfilename:f for f in client.repo_info(REPO,repo_type='dataset',revision=commit,files_metadata=True).siblings}
 for row in rows:
  f=siblings[row['path']];assert f.size==row['bytes']
  if f.lfs:assert f.lfs.sha256==row['sha256']
  row['url']=f'https://huggingface.co/datasets/{REPO}/resolve/{commit}/'+row['path']
 result=dict(repo=REPO,commit=commit,files=rows,full_launch_allowed=False)
 (ROOT/(kind+'.json')).write_text(json.dumps(result,indent=2));print('FMRG_ADDITIONS='+json.dumps(result),flush=True)
 return result

def vlm():
 from huggingface_hub import hf_hub_download
 directory=ROOT/'vlm';directory.mkdir(exist_ok=True)
 revision='cfc6496e451f4e636d7e78ce6249a4ee0efb89a1'
 # Fail before large transfers if the granted account still cannot read config.
 hf_hub_download('Skywork/Skywork-VL-Reward-7B','config.json',revision=revision,local_dir=directory)
 plan=json.loads(Path('/opt/asset-plan.json').read_text())
 entries=[f for f in plan['files'] if f['component']=='vlm']
 paths=[]
 for entry in entries:
  p=Path(hf_hub_download(entry['repo'],entry['filename'],revision=entry['revision'],local_dir=directory))
  if entry.get('bytes'):assert p.stat().st_size==entry['bytes']
  if entry.get('sha256'):assert sha(p)==entry['sha256']
  paths.append(p);print('VLM_HASH_VERIFIED',p.name,p.stat().st_size,flush=True)
 manifest=directory/'manifest.json';manifest.write_text(json.dumps(dict(repo='Skywork/Skywork-VL-Reward-7B',revision=revision,
  status='WEIGHTS_HASH_VERIFIED_NOT_PAPER_REWARD_QUALIFIED',files=[dict(filename=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in paths],
  reward_adapter='PENDING: paper Yes/No logits are distinct from scalar value head',full_launch_allowed=False),indent=2))
 return publish(paths+[manifest],'vlm')

def datasets():
 import numpy as np,pyarrow.parquet as pq,requests
 from PIL import Image
 from huggingface_hub import hf_hub_download
 plan=json.loads((Path(__file__).parent/'reconstructed-plan.json').read_text());assert plan['claim']=='reconstructed'
 root=ROOT/'reconstructed';root.mkdir(exist_ok=True);rawroot=ROOT/'dataset-raw';rawroot.mkdir(exist_ok=True)
 rows=[];content=set()
 preprocessing=dict(rgb=True,resize=[256,256],interpolation='PIL.BICUBIC',crop=None,exif_transpose=False,
                    model_range='[-1,1]',pillow=__import__('PIL').__version__,torchvision='source-compatible PIL Resize')
 pre_sha=hashlib.sha256(json.dumps(preprocessing,sort_keys=True).encode()).hexdigest()
 def save_image(data,dataset,identifier,split,source,attribution=None):
  image=Image.open(io.BytesIO(data)).convert('RGB');decoded=hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
  image=image.resize((256,256),Image.Resampling.BICUBIC)
  pixels=hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
  target=root/dataset/split/(identifier+'.png');target.parent.mkdir(parents=True,exist_ok=True)
  image.save(target,format='PNG',compress_level=6)
  return dict(dataset=dataset,split=split,original_id=identifier,source_repo=source['repo'],source_revision=source['revision'],
   source_file=source['file'],raw_sha256=hashlib.sha256(data).hexdigest(),decoded_rgb_sha256=decoded,
   preprocessed_sha256=pixels,png_sha256=sha(target),path=target.relative_to(root).as_posix(),
   preprocessing_sha256=pre_sha,attribution=attribution,observations_status='NOT_GENERATED')
 # AFHQ original mirror has label1=dog. Keep original parquet row identity.
 candidates=[]
 for shard_index,f in enumerate(plan['afhq']['files']):
  p=Path(hf_hub_download(plan['afhq']['repo'],f['source_file'],revision=plan['afhq']['revision'],repo_type='dataset',local_dir=rawroot/'afhq'))
  assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
  table=pq.read_table(p,columns=['image','label'])
  for i,(img,label) in enumerate(zip(table.column('image').to_pylist(),table.column('label').to_pylist())):
   if label==1:
    identifier=f'shard{shard_index}-row{i:05d}'
    rank=hashlib.sha256((plan['namespace']+'/AFHQ-Dog/'+identifier).encode()).hexdigest()
    candidates.append((rank,identifier,img,f['source_file']))
 candidates.sort(key=lambda x:x[0]);assert len(candidates)>=1050
 for i,(_,identifier,img,source_file) in enumerate(candidates[:1050]):
  row=save_image(img['bytes'],'AFHQ-Dog',identifier,'tuning' if i<50 else 'evaluation',
   dict(repo=plan['afhq']['repo'],revision=plan['afhq']['revision'],file=source_file),
   dict(license='CC-BY-NC-4.0',credit='NAVER Corporation; Choi, Uh, Yoo, Ha. StarGAN v2, CVPR2020.',changes='RGB, direct BICUBIC resize256; selected reconstructed subset'))
  row['source_image_path']=img.get('path');assert row['preprocessed_sha256'] not in content,'Duplicate content: abort without replacement'
  content.add(row['preprocessed_sha256']);rows.append(row)
 print('AFHQ_RECONSTRUCTED_1050_READY',flush=True)
 # Preserve original FFHQ per-image authors/licenses; verify official metadata MD5.
 meta=rawroot/'ffhq-dataset-v2.json';spec=plan['ffhq']['metadata']
 if not meta.exists():
  with requests.get(spec['url'],stream=True,timeout=(30,120)) as r:
   r.raise_for_status()
   with meta.open('wb') as out:
    for b in r.iter_content(8<<20):out.write(b)
 assert meta.stat().st_size==spec['bytes'] and hashlib.md5(meta.read_bytes()).hexdigest()==spec['md5'],'Official FFHQ metadata mismatch'
 original=json.loads(meta.read_text());assert len(original)==70000
 def get_ff(entry):
  p=Path(hf_hub_download(plan['ffhq']['repo'],entry['source_file'],revision=plan['ffhq']['revision'],repo_type='dataset',local_dir=rawroot/'ffhq'))
  assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
  data=p.read_bytes();info=original[str(int(entry['original_id']))]
  expected=info['image']
  assert hashlib.md5(data).hexdigest()==expected['file_md5'],'Mirror does not match original aligned FFHQ image'
  return save_image(data,'FFHQ',entry['original_id'],entry['split'],dict(repo=plan['ffhq']['repo'],revision=plan['ffhq']['revision'],file=entry['source_file']),
   dict(metadata=info,changes='Aligned1024 original to RGB BICUBIC256; reconstructed subset, not paper IDs'))
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for row in pool.map(get_ff,plan['ffhq']['files']):
   assert row['preprocessed_sha256'] not in content,'Duplicate content: abort without replacement'
   content.add(row['preprocessed_sha256']);rows.append(row)
 rows.sort(key=lambda r:(r['dataset'],r['split'],r['original_id']))
 assert len(rows)==2100
 manifest=dict(protocol='fmrg-dataset-freeze-v1',claim='reconstructed',full_launch_allowed=False,rows=rows,
  preprocessing=preprocessing,preprocessing_sha256=pre_sha,selection=plan['selection'],namespace=plan['namespace'],
  af_hq_version=plan['afhq']['version'],author_exact_split=False,measurement_tensors='NOT_GENERATED',
  licenses='AFHQ CC-BY-NC4; FFHQ dataset CC-BY-NC-SA4 and per-image original terms in attribution; noncommercial research use.')
 from strict_evidence import validate_inverse_split
 receipt=validate_inverse_split(manifest,require_measurements=False)
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
 (root/'validation.json').write_text(json.dumps(receipt,indent=2))
 (root/'selection-plan.json').write_text(json.dumps(plan,indent=2))
 for name,url in [('AFHQ-LICENSE.txt',f"https://raw.githubusercontent.com/clovaai/stargan-v2/{plan['afhq']['official_repo_commit']}/LICENSE"),
                  ('FFHQ-LICENSE.txt',f"https://raw.githubusercontent.com/NVlabs/ffhq-dataset/{plan['ffhq']['official_repo_commit']}/LICENSE.txt")]:
  r=requests.get(url,timeout=60);r.raise_for_status();(root/name).write_bytes(r.content)
 archive=ROOT/'reconstructed-splits-256.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
  for p in sorted(root.rglob('*')):
   if p.is_file():tar.add(p,arcname=p.relative_to(root))
 return publish([archive,root/'manifest.json',root/'validation.json',root/'selection-plan.json',root/'AFHQ-LICENSE.txt',root/'FFHQ-LICENSE.txt'],'reconstructed')

if __name__=='__main__':
 import sys
 {'vlm':vlm,'datasets':datasets}[sys.argv[1]]()
