"""Download selected public immutable HF groups through an explicit proxy."""
import argparse,hashlib,json,os,re,urllib.request
from pathlib import Path,PurePosixPath

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def select(manifest,wanted):
 files=[f for f in manifest['files'] if f['group'] in wanted]
 assert {f['group'] for f in files}==wanted
 seen=set()
 for f in files:
  assert re.fullmatch(r'[0-9a-f]{64}',f['sha256']) and type(f['bytes']) is int and f['bytes']>=0
  assert re.fullmatch(r'https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/[0-9a-f]{40}/full-paper/20261101/[\w./-]+',f['url'])
  rel=PurePosixPath(f['group'])/f['filename'];assert len(rel.parts)==2 and '..' not in rel.parts and not rel.is_absolute()
  assert '\\' not in str(rel) and ':' not in str(rel) and rel not in seen
  seen.add(rel)
 return files

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest-url',required=True);p.add_argument('--sha256',required=True)
 p.add_argument('--dest',type=Path,required=True);p.add_argument('--group',action='append',required=True);a=p.parse_args()
 assert re.fullmatch(r'https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/[0-9a-f]{40}/full-paper/20261101/publication.json',a.manifest_url)
 assert re.fullmatch(r'[0-9a-f]{64}',a.sha256)
 proxy=os.environ.get('FMRG_PROXY');assert proxy,'Approved FMRG_PROXY required'
 for k in ['NO_PROXY','no_proxy']:os.environ[k]=''
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy}))
 with opener.open(a.manifest_url,timeout=120) as r:raw=r.read(2_000_000)
 assert hashlib.sha256(raw).hexdigest()==a.sha256
 files=select(json.loads(raw),set(a.group));root=a.dest.resolve()
 assert a.dest.is_absolute() and len(root.parts)>=3 and root!=Path('/')
 root.mkdir(parents=True,exist_ok=True)
 receipts=[]
 for f in files:
  target=root/f['group']/f['filename'];target.parent.mkdir(parents=True,exist_ok=True)
  assert target.resolve().is_relative_to(root) and not target.is_symlink()
  if target.exists():assert target.is_file() and target.stat().st_size==f['bytes'] and sha(target)==f['sha256'],'Preserve mismatched existing file'
  else:
   partial=target.with_suffix(target.suffix+'.part');assert not partial.is_symlink()
   with opener.open(f['url'],timeout=180) as src,partial.open('wb') as out:
    for b in iter(lambda:src.read(8<<20),b''):out.write(b)
   assert partial.stat().st_size==f['bytes'] and sha(partial)==f['sha256']
   assert not target.exists();partial.rename(target)
  receipts.append(dict(group=f['group'],filename=f['filename'],sha256=f['sha256'],bytes=f['bytes']))
  print('VERIFIED',f['group'],f['filename'],flush=True)
 (root/'download-20261101-receipt.json').write_text(json.dumps(dict(status='DOWNLOADS_VERIFIED',manifest_url=a.manifest_url,manifest_sha256=a.sha256,files=receipts,full_launch_allowed=False),indent=2))

if __name__=='__main__':main()
