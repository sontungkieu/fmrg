"""Strict complete-group scoring of frozen inverse outputs; no generation."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from inverse_campaign import dump_new, find_complete, safe_relative, sha


def mean_ci(values):
    import numpy as np
    x=np.asarray(values,dtype=np.float64)
    assert x.shape==(1000,) and np.isfinite(x).all()
    rng=np.random.Generator(np.random.PCG64(20261102))
    boots=np.concatenate([x[rng.integers(0,1000,size=(100,1000))].mean(1) for _ in range(100)])
    return dict(mean=float(x.mean()),std_population=float(x.std()),
                ci95_percentile=[float(v) for v in np.quantile(boots,[.025,.975])],images=1000)


def main():
    p=argparse.ArgumentParser()
    for name in ['contract','plan','launch-spec','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--roots',type=Path,nargs='+',required=True)
    p.add_argument('--dataset',choices=['AFHQ-Dog','FFHQ'],required=True)
    p.add_argument('--task',choices=['sr4','motion61','mask64'],required=True)
    p.add_argument('--method',choices=['E','J'],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--canary',action='store_true',help='First16 tuning outputs only; no dataset FID/KID/CI')
    a=p.parse_args();start=time.monotonic()
    c=json.loads(a.contract.read_text());plan=json.loads(a.plan.read_text());spec=json.loads(a.launch_spec.read_text())
    cs=sha(a.contract)
    assert cs==plan['contract_sha256']==spec['contract_sha256']
    assert sha(a.plan)==spec['plan_sha256']
    assert sha(Path(plan['paths']['source'])/'scripts/solve_inverse_problem.py')=='924288546c5c3e027d3a86a9d4a17a65b3bff4a1e25b4590b4806889ba5b8c1b'
    assert sha(Path(plan['paths']['source'])/'scripts/aggregate_metrics.py')=='1dc65de15f5eb480b8884e57c1337496f53716c80b55d11a43b0bbba64adab43'
    split='tuning' if a.canary else 'eval'
    rows_file=a.plan.parent/(split+'.jsonl');assert sha(rows_file)==plan[split+'_jsonl_sha256']
    cells=[json.loads(x) for x in rows_file.read_text().splitlines()]
    cells=[x for x in cells if (x['dataset'],x['task'],x['method'])==(a.dataset,a.task,a.method)]
    assert len(cells)==len({x['original_id'] for x in cells})==(50 if a.canary else 1000)
    if a.canary:cells=cells[:16]
    # Completeness and integrity are checked BEFORE any group metric is computed.
    paths=[]
    for cell in cells:
        record=find_complete(a.roots,cell,cs,spec['code_sha256']['run_inverse_campaign.py'])
        assert record is not None,'INCOMPLETE_NOT_SCORED: expected cell absent'
        found=[r/'cells'/cell['cell_id'] for r in a.roots if (r/'cells'/cell['cell_id']/'record.json').is_file()]
        assert len(found)==1
        paths.append((cell,found[0],record))
    sys.path.insert(0,str(Path(__file__).parent))
    from frozen_smoke import offline_network
    extra=Path(plan['paths']['extra'])
    os.environ['TORCH_HOME']=str(extra/'torch')
    sys.path.insert(0,str(extra/'python-overlay'))
    import numpy as np
    import torch
    from PIL import Image
    import lpips
    from safetensors.torch import load_file
    from torchvision import transforms as T
    from pytorch_msssim import ssim
    from skimage.metrics import peak_signal_noise_ratio
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    assert torch.__version__=='2.7.1+cu128'
    assert sha(Path(lpips.__file__).parent/'weights/v0.1/vgg.pth')==sha(extra/'metrics/lpips-v0.1-vgg.pth')
    if a.device=='cuda':
        allocation=json.loads(Path(spec['allocation']).read_text())
        assert os.environ['CUDA_VISIBLE_DEVICES'] in allocation['gpu_uuids']
        assert time.time()<allocation['deadline_unix']
    torch.set_num_threads(8)
    transform=T.ToTensor();resize224=T.Compose([T.Resize((224,224)),T.ToTensor()])
    resize299=T.Compose([T.Resize((299,299)),T.ToTensor()])
    values=[]
    with offline_network(),torch.no_grad():
        metric=lpips.LPIPS(net='vgg',verbose=False).to(a.device).eval().requires_grad_(False)
        fid=FrechetInceptionDistance(feature=2048,normalize=False).to(a.device)
        kid=KernelInceptionDistance(subset_size=50,subsets=100,normalize=False).to(a.device)
        for cell,directory,record in paths:
            gt_path=safe_relative(plan['paths']['splits'],cell['image_path'])
            assert sha(gt_path)==cell['image_sha256']
            gt=Image.open(gt_path).convert('RGB');pred=Image.open(directory/'prediction.png').convert('RGB')
            raw=load_file(directory/'prediction.safetensors',device='cpu')['prediction']
            assert raw.dtype==torch.float32 and tuple(raw.shape)==(1,3,256,256) and torch.isfinite(raw).all()
            assert __import__('hashlib').sha256(raw.contiguous().numpy().tobytes()).hexdigest()==record['prediction_payload_sha256']
            png=np.rint(((raw[0].permute(1,2,0).numpy()+1)/2).clip(0,1)*255).astype(np.uint8)
            assert np.array_equal(png,np.array(pred))
            g=transform(gt).unsqueeze(0).to(a.device);v=transform(pred).unsqueeze(0).to(a.device)
            rescore=-10*math.log10(max(float(((raw.double()-(transform(gt)*2-1).double().unsqueeze(0))/2).square().mean()),1e-20))
            assert abs(rescore-record['psnr_float_diagnostic'])<1e-4
            row=dict(cell_id=cell['cell_id'],original_id=cell['original_id'],
                psnr=float(peak_signal_noise_ratio(g.cpu().numpy()*255,v.cpu().numpy()*255,data_range=255)),
                ssim=float(ssim(g*255,v*255)),
                lpips_cli=float(metric(resize224(gt).to(a.device),resize224(pred).to(a.device)).item()),
                lpips_native256=float(metric(g*2-1,v*2-1).item()))
            assert all(math.isfinite(row[k]) for k in ['psnr','ssim','lpips_cli','lpips_native256'])
            values.append(row)
            real=(resize299(gt)*255).to(torch.uint8).unsqueeze(0).to(a.device)
            fake=(resize299(pred)*255).to(torch.uint8).unsqueeze(0).to(a.device)
            fid.update(real,real=True);fid.update(fake,real=False)
            kid.update(real,real=True);kid.update(fake,real=False)
            if a.device=='cuda':assert time.time()<allocation['deadline_unix']
        torch.manual_seed(20261102)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(20261102)
        if a.canary:
            assert torch.isfinite(fid.real_features_sum).all() and int(fid.real_features_num_samples)==16
            fid_score=km=ks=None
        else:
            fid_score=float(fid.compute());km,ks=kid.compute()
    if a.canary:
        a.out.mkdir(exist_ok=False)
        dump_new(a.out/'canary.json',dict(status='PASS_TECHNICAL_16_TUNING_OUTPUTS',full_launch_allowed=False,
            dataset_score='NOT_COMPUTED',rows=values,scorer_sha256=sha(__file__),contract_sha256=cs,
            fid_features_finite=True,seconds=time.monotonic()-start))
        print('INVERSE_SCORER_CANARY_PASS',flush=True)
        return
    assert all(math.isfinite(v) for v in [fid_score,float(km),float(ks)])
    a.out.mkdir(exist_ok=False)
    dump_new(a.out/'per-image.json',values)
    dump_new(a.out/'summary.json',dict(status='COMPLETE_1000_EXPECTED_IDS',claim='reconstructed',
        dataset=a.dataset,task=a.task,method=a.method,contract_sha256=cs,plan_sha256=sha(a.plan),
        scorer_sha256=sha(__file__),device=a.device,
        metrics={key:mean_ci([v[key] for v in values]) for key in ['psnr','ssim','lpips_cli','lpips_native256']},
        fid=fid_score,kid_mean=float(km),kid_std_subsets=float(ks),
        no_FID_CI=True,no_equivalence_claim=True,seconds=time.monotonic()-start,
        per_image_sha256=sha(a.out/'per-image.json')))
    print('INVERSE_SCORE_COMPLETE='+json.dumps(dict(directory=str(a.out),images=1000,fid=fid_score,kid=float(km))),flush=True)


if __name__=='__main__':main()
