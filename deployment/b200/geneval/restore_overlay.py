"""Restore the hash-pinned evaluator at a new fixed prefix; no installs/GPU calls."""
import argparse,hashlib,json,os,posixpath,subprocess,tarfile
from pathlib import Path,PurePosixPath

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def validate_members(members):
 prefix=PurePosixPath('opt/fmrg-geneval')
 names=[m.name.rstrip('/') for m in members];assert len(names)==len(set(names)),'Duplicate archive members'
 links={PurePosixPath(m.name) for m in members if m.issym()}
 for m in members:
  path=PurePosixPath(m.name)
  assert not path.is_absolute() and '..' not in path.parts and path.is_relative_to(prefix)
  assert not any(parent in links for parent in path.parents),'Archive file below a symlink'
  assert not m.mode & 0o7000,'Special permission bits disallowed'
  if m.issym():
   target=PurePosixPath(m.linkname)
   if target.is_absolute():assert target.is_relative_to('/opt/fmrg/python') and '..' not in target.parts,'Unexpected external runtime link'
   else:
    normalized=PurePosixPath(posixpath.normpath(str(path.parent/target)))
    assert normalized.is_relative_to(prefix) and '..' not in normalized.parts
  else:assert m.isfile() or m.isdir()

def restore(archive,sha256,size):
 archive=Path(archive).resolve();dest=Path('/opt/fmrg-geneval')
 assert not dest.exists() and not dest.is_symlink(),'Preserve existing evaluator; validate/reuse its receipt or use a fresh container'
 assert Path('/opt').resolve()==Path('/opt')
 assert archive.stat().st_size==size and digest(archive)==sha256
 core=Path('/opt/fmrg/venv/bin/python');assert core.is_file(),'Restore and qualify pinned core runtime first'
 info=json.loads(subprocess.check_output([str(core),'-c',"import json,sys,torch; print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda)))"],text=True))
 assert info==dict(python='3.11.14',torch='2.7.1+cu128',cuda='12.8'),'Core ABI mismatch'
 with tarfile.open(archive) as tar:
  members=tar.getmembers();validate_members(members)
  # Every member and link is checked against the fixed prefix and inherited
  # interpreter; a generic tar data filter would reject the deliberate core link.
  tar.extractall('/',members=members,filter='fully_trusted')
 record=dict(status='RESTORED_NOT_COMPANY_GPU_QUALIFIED',overlay_sha256=sha256,bytes=size,core_versions=info,
  core_runtime_required_sha256='06ef365e64d7dcccd7507ac3435f46bf25b724679b65c8e1a8d7a060958f43ff',
  existing_core_gate_receipt_required=True,full_launch_allowed=False)
 (dest/'restore-receipt.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--sha256',required=True);p.add_argument('--bytes',required=True,type=int);a=p.parse_args()
 restore(a.archive,a.sha256,a.bytes)
