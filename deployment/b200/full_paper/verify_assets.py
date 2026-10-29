"""Verify selected downloaded components without trusting an existence check."""
import json
from pathlib import Path
from download_assets import digest,select_files

def verify(root,components):
 root=Path(root).resolve()
 receipt=json.loads((root/'download-receipt.json').read_text())
 assert receipt['status']=='DOWNLOADS_HASH_VERIFIED'
 raw=(root/'asset-manifest.json').read_bytes()
 import hashlib
 assert hashlib.sha256(raw).hexdigest()==receipt['manifest_sha256']
 files=select_files(json.loads(raw),set(components))
 for f in files:
  p=root/f['component']/f['filename']
  assert p.resolve().is_relative_to(root) and p.is_file() and not p.is_symlink()
  assert p.stat().st_size==f['bytes'] and digest(p)==f['sha256'],'Asset identity mismatch: '+f['filename']
 return files
