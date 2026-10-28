"""Offline assets restore: explicit destination, hash and safe regular files."""
import argparse,hashlib,json,os,tarfile,uuid
from pathlib import Path
def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(8<<20),b""):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--archive",type=Path,required=True);p.add_argument("--sha256",required=True);p.add_argument("--dest",type=Path,required=True);a=p.parse_args()
 archive=a.archive.resolve(strict=True);dest=a.dest.resolve()
 if dest.exists() or not dest.is_absolute() or dest==Path("/") or len(dest.parts)<3:raise SystemExit("Use a new explicit data directory")
 if not dest.parent.is_dir():raise SystemExit("Destination parent must exist")
 if digest(archive)!=a.sha256:raise SystemExit("Archive SHA256 mismatch")
 stage=dest.with_name(dest.name+".staging-"+uuid.uuid4().hex[:8]);stage.mkdir()
 digests={};sizes={}
 with tarfile.open(archive,mode="r|*",bufsize=8<<20) as tf:
  for m in tf:
   relative=Path(m.name)
   if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0]!="assets":raise SystemExit("Unsafe archive path")
   if not m.isfile():raise SystemExit("Only regular files allowed in assets bundle")
   target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():raise SystemExit("Duplicate archive member")
   h=hashlib.sha256()
   with tf.extractfile(m) as src,target.open("wb") as out:
    for b in iter(lambda:src.read(8<<20),b""):out.write(b);h.update(b)
   digests[m.name]=h.hexdigest();sizes[m.name]=m.size
 manifest=json.loads((stage/"assets/assets-manifest.json").read_text())
 for f in manifest["files"]:
  if digests.get(f["path"])!=f["sha256"] or sizes.get(f["path"])!=f["bytes"]:raise SystemExit("Asset identity mismatch")
 if set(digests)!={f["path"] for f in manifest["files"]}|{"assets/assets-manifest.json"}:raise SystemExit("Unexpected asset files")
 stage.rename(dest);root=dest/"assets"
 base_files=[dict(file=f["path"].removeprefix("assets/flux/"),bytes=f["bytes"],sha256=f["sha256"]) for f in manifest["files"] if f["path"].startswith("assets/flux/")]
 base=dict(base_path=str(root/"flux"),base_revision=manifest["base_revision"],flowmap_revision=manifest["flowmap_revision"],
           lora_256=str(root/"lora/256"),lora_512=str(root/"lora/512"),files=base_files)
 (root/"assets.json").write_text(json.dumps(base,indent=2))
 (root/"pilot-assets.json").write_text(json.dumps(dict(clip_path=str(root/"clip"),clip_revision=manifest["clip_revision"]),indent=2))
 print(json.dumps(dict(status="RESTORED_HASH_VERIFIED",data_root=str(root),archive_sha256=a.sha256,files=len(manifest["files"])),indent=2))
if __name__=="__main__":main()
