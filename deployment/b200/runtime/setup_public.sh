#!/usr/bin/env bash
# Run only inside the operator's authorized company allocation/container.
set -euo pipefail
manifest_url="${1:?immutable publication manifest HF URL}"
manifest_sha="${2:?publication manifest SHA256}"
download_dir="${3:?existing absolute download directory}"
data_dest="${4:?new absolute assets restore directory}"
run_dir="${5:?new absolute qualification output directory}"
: "${FMRG_PROXY:?Set the approved HF proxy without printing credentials}"
[[ "$(uname -s)" = Linux && "$(uname -m)" = x86_64 ]]
[[ "$manifest_url" =~ ^https://huggingface.co/datasets/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/resolve/[0-9a-f]{40}/publication.json$ ]]
[[ "$manifest_sha" =~ ^[0-9a-f]{64}$ ]]
[[ "$download_dir" = /* && -d "$download_dir" ]]
[[ "$data_dest" = /* && ! -e "$data_dest" && ! -L "$data_dest" ]]
[[ "$run_dir" = /* && ! -e "$run_dir" && ! -L "$run_dir" ]]
[[ ! -e /opt/fmrg && ! -L /opt/fmrg ]] || { echo 'Preserve existing /opt/fmrg; setup refuses overwrite' >&2; exit 2; }
command -v python3 >/dev/null
command -v sha256sum >/dev/null
command -v curl >/dev/null
manifest="$download_dir/fmrg-publication-$manifest_sha.json"
curl -fL --retry 2 --connect-timeout 30 --max-time 120 --proxy "$FMRG_PROXY" --noproxy '' "$manifest_url" -o "$manifest.part"
printf '%s  %s\n' "$manifest_sha" "$manifest.part" | sha256sum -c -
mv -- "$manifest.part" "$manifest"
plan="$download_dir/fmrg-download-plan-$manifest_sha.tsv"
python3 - "$manifest" > "$plan" <<'PY'
import json,re,sys
d=json.load(open(sys.argv[1]))
assert d['private'] is False
prefix=f"https://huggingface.co/datasets/{d['repo']}/resolve/{d['commit']}/"
assert re.fullmatch(r'[A-Za-z0-9._-]+/[A-Za-z0-9._-]+',d['repo'])
assert re.fullmatch(r'[0-9a-f]{40}',d['commit'])
for path,name in [('runtime/runtime.tar.gz','runtime.tar.gz'),('assets/assets.tar','assets.tar'),('source/setup-source.tar.gz','setup-source.tar.gz')]:
    matches=[f for f in d['files'] if f['path']==path]
    assert len(matches)==1
    f=matches[0]
    assert f['url']==prefix+path and re.fullmatch(r'[0-9a-f]{64}',f['sha256'])
    assert isinstance(f['bytes'],int) and f['bytes']>0
    print(name,f['sha256'],f['bytes'],f['url'],sep='\t')
PY
while IFS=$'\t' read -r name digest size url; do
  file="$download_dir/$name"
  if [[ -f "$file" ]] && [[ "$(stat -c %s "$file")" = "$size" ]] && printf '%s  %s\n' "$digest" "$file" | sha256sum -c - >/dev/null 2>&1; then
    printf 'Reusing verified %s\n' "$name"
    continue
  fi
  [[ ! -e "$file" ]] || { echo "Existing mismatched $name; preserve and resolve it" >&2; exit 2; }
  curl -fL --retry 2 --connect-timeout 30 --max-time 7200 --proxy "$FMRG_PROXY" --noproxy '' "$url" -o "$file.part"
  [[ "$(stat -c %s "$file.part")" = "$size" ]]
  printf '%s  %s\n' "$digest" "$file.part" | sha256sum -c -
  mv -- "$file.part" "$file"
done < "$plan"
source_dir="$download_dir/fmrg-setup-$manifest_sha"
python3 - "$download_dir/setup-source.tar.gz" "$source_dir" <<'PY'
import pathlib,sys,tarfile
dest=pathlib.Path(sys.argv[2])
assert not dest.exists()
dest.mkdir()
with tarfile.open(sys.argv[1]) as tf:
    for m in tf:
        p=pathlib.PurePosixPath(m.name)
        assert m.isfile() and not p.is_absolute() and '..' not in p.parts
        assert p.parts[0]=='fmrg-setup'
        target=dest.joinpath(*p.parts)
        assert not target.exists()
        target.parent.mkdir(parents=True,exist_ok=True)
        with tf.extractfile(m) as src,target.open('wb') as out:
            import shutil
            shutil.copyfileobj(src,out)
PY
# Reuse the already verified local runtime; original installer validation remains
# authoritative. No second large HF download is required.
python3 - "$download_dir/runtime.tar.gz" <<'PY'
import pathlib,sys,tarfile
root=pathlib.Path('/opt/fmrg')
with tarfile.open(sys.argv[1]) as tf:
    for m in tf:
        assert m.isfile() or m.isdir() or m.issym() or m.islnk()
        target=(pathlib.Path('/')/m.name).resolve()
        assert target==root or root in target.parents
        if m.issym() or m.islnk():
            link=pathlib.Path(m.linkname)
            to=(link if link.is_absolute() else (pathlib.Path('/')/link if m.islnk() else target.parent/link)).resolve()
            assert to==root or root in to.parents
PY
[[ ! -e /opt/fmrg && ! -L /opt/fmrg ]] || { echo '/opt/fmrg appeared during download; preserve it' >&2; exit 2; }
tar -xzf "$download_dir/runtime.tar.gz" -C /
pilot_root="$source_dir/fmrg-setup"
/opt/fmrg/python-fmrg.sh "$pilot_root/experiments/restore_assets.py" \
  --archive "$download_dir/assets.tar" \
  --sha256 "$(awk -F '\t' '$1=="assets.tar" {print $2}' "$plan")" --dest "$data_dest"
export FMRG_COMPANY=1 FMRG_DATA_ROOT="$data_dest/assets" FMRG_SOURCE_ROOT=/opt/fmrg/source
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$data_dest/assets/hf"
mkdir -- "$run_dir"
python3 - "$manifest" "$run_dir" "$pilot_root" "$FMRG_DATA_ROOT" <<'PY'
import json,pathlib,sys
d=json.load(open(sys.argv[1]));d.update(pilot_root=sys.argv[3],data_root=sys.argv[4],company_status='GATES_PENDING')
pathlib.Path(sys.argv[2],'publication-used.json').write_text(json.dumps(d,indent=2))
PY
for stage in imports kernel tiny full; do
  /opt/fmrg/python-fmrg.sh /opt/fmrg/tools/gates.py "$stage" --data "$FMRG_DATA_ROOT" --out "$run_dir/$stage.json"
done
python3 - "$run_dir" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
records={stage:json.loads((root/(stage+'.json')).read_text()) for stage in ['imports','kernel','tiny','full']}
assert all(r['status']=='PASS' for r in records.values())
path=root/'publication-used.json'
d=json.loads(path.read_text());d['company_status']='TECHNICAL_PASS'
path.write_text(json.dumps(d,indent=2))
(root/'setup-summary.json').write_text(json.dumps(dict(status='TECHNICAL_PASS',gates={k:r['status'] for k,r in records.items()}),indent=2))
PY
printf 'FMRG_COMPANY_TECHNICAL_PASS\nData root: %s\nPilot source: %s\nReceipts: %s\n' "$FMRG_DATA_ROOT" "$pilot_root" "$run_dir"
