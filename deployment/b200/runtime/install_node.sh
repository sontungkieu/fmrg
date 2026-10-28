#!/usr/bin/env bash
# Run on the authorized company node, not on Windows/WSL laptop.
# URL is supplied only after an authorized HF publication; never fall back to GitHub.
set -euo pipefail
url="${1:?immutable HF datasets resolve URL required}"
sha="${2:?archive sha256 required}"
download_dir="${3:?existing absolute download directory required}"
[[ "$(uname -s)" = Linux ]]
[[ "$url" =~ ^https://huggingface.co/datasets/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/resolve/[0-9a-f]{40}/[A-Za-z0-9._/-]+$ ]]
[[ "$sha" =~ ^[0-9a-f]{64}$ ]]
[[ "$download_dir" = /* && -d "$download_dir" ]]
[[ -n "${FMRG_PROXY:-}" ]] || { echo 'FMRG_PROXY is required' >&2; exit 2; }
[[ ! -e /opt/fmrg && ! -L /opt/fmrg ]] || { echo '/opt/fmrg already exists; preserve it and resolve prefix deliberately' >&2; exit 2; }
archive="$download_dir/fmrg-runtime-$sha.tar.gz"
curl -fL --retry 2 --connect-timeout 30 --max-time 1800 --proxy "$FMRG_PROXY" --noproxy '' "$url" -o "$archive"
printf '%s  %s\n' "$sha" "$archive" | sha256sum -c -
# Validate all paths before path-preserving extraction; do not overwrite an existing prefix.
python3 - "$archive" <<'PY'
import pathlib,sys,tarfile
root=pathlib.Path('/opt/fmrg')
with tarfile.open(sys.argv[1]) as tf:
    for member in tf:
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise SystemExit('unsupported archive member type')
        target=(pathlib.Path('/')/member.name).resolve()
        if target!=root and root not in target.parents:
            raise SystemExit('archive target outside /opt/fmrg')
        if member.issym() or member.islnk():
            link=pathlib.Path(member.linkname)
            dest=(link if link.is_absolute() else (pathlib.Path('/')/link if member.islnk() else target.parent/link)).resolve()
            if dest!=root and root not in dest.parents:
                raise SystemExit('archive symlink outside /opt/fmrg')
PY
tar -xzf "$archive" -C /
/opt/fmrg/python-fmrg.sh -c 'import torch,diffusers,transformers; print(torch.__version__,diffusers.__version__,transformers.__version__)'
printf '%s\n' 'FMRG_INSTALLED_IMPORT_ONLY; run allocation-bound gates separately'
