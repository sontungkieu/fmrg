"""Remote CPU build of an isolated evaluator overlay; no GPU or generation."""
import hashlib,json,os,shutil,subprocess,sys,tarfile,time,posixpath
from pathlib import Path,PurePosixPath
ROOT=Path('/data/geneval-20261101');PREFIX=Path('/opt/fmrg-geneval');PY=PREFIX/'venv/bin/python'
CORE_SHA='06ef365e64d7dcccd7507ac3435f46bf25b724679b65c8e1a8d7a060958f43ff'
MMCV_SHA='1e2b3cbb4a9d31ffe6039fd5dc3b13edef1f7795f38fd12b8ee72c50e2639adb'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def command(args,timeout=600,cwd=None,env=None):
 args=[str(x) for x in args]
 print('BUILD_STEP='+json.dumps(args[:5]),flush=True)
 subprocess.run(args,check=True,timeout=timeout,cwd=cwd,env=env)

def extract(archive,dest):
 with tarfile.open(archive) as tar:
  for m in tar.getmembers():
   p=PurePosixPath(m.name);assert not p.is_absolute() and '..' not in p.parts
   if m.issym():
    target=PurePosixPath(posixpath.normpath(str(p.parent/m.linkname)))
    assert not target.is_absolute() and '..' not in target.parts,'Escaping archive symlink'
   else:assert m.isfile() or m.isdir()
  tar.extractall(dest,filter='data')

def build():
 ROOT.mkdir(exist_ok=True);start=time.monotonic()
 core=Path('/data/publication/runtime-public.tar.gz');assert sha(core)==CORE_SHA
 if not Path('/opt/fmrg').exists():command(['tar','-xzf',core,'-C','/'],timeout=180)
 assert not PREFIX.exists(),'Fresh build container required'
 PREFIX.mkdir()
 command(['/opt/fmrg/venv/bin/python','-m','venv',PREFIX/'venv'])
 site=PREFIX/'venv/lib/python3.11/site-packages'
 (site/'core-runtime.pth').write_text('/opt/fmrg/venv/lib/python3.11/site-packages\n')
 command([PY,'-m','pip','install','--no-deps','setuptools==69.5.1','wheel==0.45.1','numpy==1.26.4',
  'addict==2.4.0','yapf==0.32.0','opencv-python-headless==4.8.1.78','pycocotools==2.0.7','terminaltables==3.1.10',
  'scikit-learn==1.2.2','joblib==1.2.0','threadpoolctl==3.1.0',
  'open-clip-torch==2.20.0','timm==0.9.2','clip-benchmark==1.4.0','braceexpand==0.1.7','webdataset==0.2.48'])
 source=PREFIX/'source';source.mkdir()
 external=Path('/data/full-paper-20261029/assets/sources/evaluator-source.tar.gz')
 assert sha(external)=='dd7a0640f881c23f3d2c8f5f7efe1c4275b38d2a7b17b05bed9bda254cb37d80'
 extract(external,source)
 mmcv_archive=Path('/opt/mmcv-source.tar.gz');assert sha(mmcv_archive)==MMCV_SHA
 extract(mmcv_archive,source)
 mmcv=source/'mmcv'
 setup=mmcv/'setup.py';before=setup.read_text();after=before.replace('-std=c++14','-std=c++17')
 assert before!=after;setup.write_text(after)
 cache=ROOT/'mmcv-build-torch271-cu128-sm100';cache.mkdir(exist_ok=True)
 (mmcv/'build').symlink_to(cache,target_is_directory=True)
 patch=dict(source_commit='7a13f995784087ace7cf17ae5f965b6190fe0aea',source_archive_sha256=MMCV_SHA,
  changes=[dict(file='setup.py',before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=sha(setup),
   change='C++14 -> C++17 required by pinned PyTorch2.7 headers; no numerical kernel edits')])
 (PREFIX/'native-patch.json').write_text(json.dumps(patch,indent=2))
 env=os.environ.copy();env.update(CUDA_HOME='/usr/local/cuda',FORCE_CUDA='1',MMCV_WITH_OPS='1',
  TORCH_CUDA_ARCH_LIST='10.0',MAX_JOBS='4',MMCV_CUDA_ARGS='-std=c++17',PIP_NO_CACHE_DIR='1')
 env['PATH']='/usr/local/cuda/bin:'+env['PATH']
 wheels=ROOT/'wheels';wheels.mkdir(exist_ok=True)
 command([PY,'-m','pip','wheel','--no-deps','--no-build-isolation','-v','--wheel-dir',wheels,mmcv],timeout=1200,env=env)
 wheel=next(wheels.glob('mmcv_full-1.7.1-*.whl'))
 command([PY,'-m','pip','install','--no-deps',wheel])
 mmdet=next(p.parent for p in source.rglob('setup.py') if (p.parent/'mmdet').is_dir())
 command([PY,'-m','pip','install','--no-deps','--no-build-isolation',mmdet])
 command([PY,'-c',"import torch,mmcv,mmdet,open_clip; from mmcv.ops import MultiScaleDeformableAttention; from mmdet.apis import init_detector,inference_detector; from clip_benchmark.metrics import zeroshot_classification; print('NATIVE_IMPORT_PASS',torch.__version__,mmcv.__version__,mmdet.__version__)"])
 lock=subprocess.check_output([str(PY),'-m','pip','freeze','--all'],text=True)
 (PREFIX/'resolved-freeze.txt').write_text(lock)
 # Build files are intentionally retained outside the portable venv for diagnosis.
 for p in [PREFIX/'source/mmcv',mmdet/'build']:
  if p.exists() and p.resolve().is_relative_to(PREFIX/'source'):
   # Move generated compiler objects to the remote work area, never delete evidence.
   target=ROOT/('build-evidence-'+p.parent.name+'-'+p.name)
   if not target.exists():shutil.move(str(p),str(target))
 # Retain the patched MMCV source separately in persistent evidence; its wheel is installed.
 archive=ROOT/'geneval-overlay.tar.gz'
 command(['tar','-I','gzip -1','-cf',archive,'-C','/','opt/fmrg-geneval'],timeout=180)
 record=dict(status='BUILT_CPU_IMPORTS_PASS_NOT_GPU_QUALIFIED',full_launch_allowed=False,
  core_runtime_sha256=CORE_SHA,overlay_prefix=str(PREFIX),torch='2.7.1+cu128',python='3.11',
  native_arch='sm_100',compiler_cuda='12.8.1',mmcv_source_commit=patch['source_commit'],patch=patch,
  overlay=dict(bytes=archive.stat().st_size,sha256=sha(archive),path=str(archive)),
  wheel=dict(bytes=wheel.stat().st_size,sha256=sha(wheel),path=str(wheel)),seconds=time.monotonic()-start)
 (ROOT/'build.json').write_text(json.dumps(record,indent=2))
 print('FMRG_GENEVAL_BUILD='+json.dumps(record),flush=True)
 return record

if __name__=='__main__':build()
