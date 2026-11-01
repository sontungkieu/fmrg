"""Restore all2100 pinned PNGs from the already staged public archive."""
import argparse
import json
from pathlib import Path, PurePosixPath
import tarfile

from inverse_campaign import dump_new, safe_relative, sha


def main():
    p=argparse.ArgumentParser()
    for name in ['archive','manifest','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    assert sha(a.archive)=='18b57d8d6d58b8bb9fb5c61837752f6666fa5028233a02768fade040a8f76c86'
    assert sha(a.manifest)=='f0a6b32ea1fd1ba6a1f0ff0e7055680ee32ae39f40f4e9bf44046fd29effac7a'
    man=json.loads(a.manifest.read_text())
    expected={r['path']:r['png_sha256'] for r in man['rows']}
    assert len(expected)==2100
    a.out.mkdir(exist_ok=False);seen=set()
    with tarfile.open(a.archive,'r:gz') as archive:
        for item in archive:
            name=PurePosixPath(item.name)
            assert not name.is_absolute() and '..' not in name.parts
            assert item.isdir() or item.isfile(),'No links/special files'
            if item.name not in expected:continue
            assert item.isfile() and item.name not in seen and item.size<2**22
            target=safe_relative(a.out,item.name);target.parent.mkdir(parents=True,exist_ok=True)
            stream=archive.extractfile(item);assert stream is not None
            data=stream.read();assert len(data)==item.size
            with target.open('xb') as output:output.write(data)
            assert sha(target)==expected[item.name]
            seen.add(item.name)
    assert seen==set(expected),'Missing expected image in archive'
    dump_new(a.out/'restore-receipt.json',dict(status='ALL_2100_PNG_HASHES_VERIFIED',images=2100,
        archive_sha256=sha(a.archive),manifest_sha256=sha(a.manifest),full_launch_allowed=False))
    print('INVERSE_IMAGES_RESTORED=2100',flush=True)


if __name__=='__main__':main()
