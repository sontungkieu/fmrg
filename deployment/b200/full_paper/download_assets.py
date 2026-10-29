"""Company CPU preparation: only immutable HF URLs through an explicit proxy."""
import argparse,hashlib,json,os,re,urllib.request
from pathlib import Path,PurePosixPath
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def select_files(manifest,wanted):
 failures=[x for x in manifest['failures'] if x['component'] in wanted]
 assert not failures,'Selected component incomplete; resolve gate before download: '+repr(failures)
 files=[f for f in manifest['files'] if f['component'] in wanted]
 assert wanted=={f['component'] for f in files},'Unknown/missing component'
 for f in files:
  rel=PurePosixPath(f['component'])/f['filename']
  assert not rel.is_absolute() and '..' not in rel.parts and len(rel.parts)==2
  assert '\\' not in str(rel) and ':' not in str(rel)
  assert re.fullmatch(r'[0-9a-f]{64}',f['sha256']) and isinstance(f['bytes'],int) and f['bytes']>=0
  assert f['path_in_repo']=='full-paper/20261029/assets/'+str(rel),'Unexpected asset path'
 return files
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest-url',required=True);p.add_argument('--sha256',required=True)
 p.add_argument('--dest',type=Path,required=True);p.add_argument('--component',action='append',required=True)
 a=p.parse_args();proxy=os.environ.get('FMRG_PROXY');assert proxy,'FMRG_PROXY required'
 assert re.fullmatch(r'https://huggingface.co/datasets/[\w.-]+/[\w.-]+/resolve/[0-9a-f]{40}/full-paper/20261029/asset-manifest.json',a.manifest_url)
 assert re.fullmatch(r'[0-9a-f]{64}',a.sha256)
 root=a.dest.resolve();assert a.dest.is_absolute() and root!=Path('/') and len(root.parts)>=3
 root.mkdir(parents=True,exist_ok=True)
 # Empty proxy bypass list is intentional, including for signed HF CDN redirects.
 for key in ['NO_PROXY','no_proxy']:os.environ[key]=''
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy}))
 with opener.open(a.manifest_url,timeout=120) as r:raw=r.read(2_000_000)
 assert hashlib.sha256(raw).hexdigest()==a.sha256,'Manifest SHA mismatch'
 manifest=json.loads(raw);wanted=set(a.component)
 files=select_files(manifest,wanted)
 base=a.manifest_url.rsplit('/full-paper/',1)[0]+'/';receipts=[]
 for f in files:
  rel=PurePosixPath(f['component'])/f['filename']
  assert not rel.is_absolute() and '..' not in rel.parts and len(rel.parts)==2
  path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
  assert path.resolve().is_relative_to(root) and not path.is_symlink(),'Unsafe asset destination'
  if path.exists():assert path.is_file() and path.stat().st_size==f['bytes'] and digest(path)==f['sha256'],'Preserve mismatched existing file'
  else:
   partial=path.with_suffix(path.suffix+'.part')
   assert not partial.is_symlink(),'Unsafe partial destination'
   url=base+f['path_in_repo']
   with opener.open(url,timeout=180) as src,partial.open('wb') as dst:
    for b in iter(lambda:src.read(8<<20),b''):dst.write(b)
   assert partial.stat().st_size==f['bytes'] and digest(partial)==f['sha256'],'File hash mismatch'
   assert not path.exists(),'Destination appeared during download'
   partial.rename(path)
  receipts.append(dict(component=f['component'],filename=f['filename'],path=str(path),sha256=f['sha256']))
  print('VERIFIED',f['component'],f['filename'],flush=True)
 record=dict(status='DOWNLOADS_HASH_VERIFIED',manifest_url=a.manifest_url,manifest_sha256=a.sha256,files=receipts,
             neural_canary='NOT_RUN',full_launch_allowed=False)
 (root/'download-receipt.json').write_text(json.dumps(record,indent=2))
 (root/'asset-manifest.json').write_bytes(raw)
if __name__=='__main__':main()
