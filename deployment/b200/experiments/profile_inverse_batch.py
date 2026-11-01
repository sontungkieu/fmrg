"""Bounded tuning-only batch throughput/gradient probe, never efficacy."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(exist_ok=False)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError('BATCH_PROBE_900S')))
    signal.alarm(900)
    plan = json.loads(a.plan.read_text())
    paths = plan['paths']
    sys.path.insert(0, paths['source'])
    sys.path.insert(0, paths['loader'])
    sys.path.insert(0, str(a.plan.parent))
    import torch
    import frozen_obs
    import frozen_smoke as common
    from paper_inverse_adapter import PaperInverseAdapter
    assert common.sha(Path(paths['source'])/'fluxfm_sampler_ip.py') == common.SAMPLER_SHA
    assert common.sha(Path(paths['loader'])/'frozen_obs.py') == common.LOADER_SHA
    assert common.sha(paths['obs_manifest']) == 'f0a6b32ea1fd1ba6a1f0ff0e7055680ee32ae39f40f4e9bf44046fd29effac7a'
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (10, 0)
    torch.set_num_threads(4)
    rows = sorted((r for r in frozen_obs.load_manifest(paths['obs_manifest'])['rows']
                   if r['dataset'] == 'AFHQ-Dog' and r['split'] == 'tuning'), key=lambda r:r['original_id'])[:16]
    measurements = []
    for r in rows:
        b = frozen_obs.load_observation(paths['obs_manifest'], paths['obs'],
            r['dataset'], r['split'], r['original_id'], 'sr4')
        assert common.sha(b['tensor_file']) == b['entry']['measurement_sha256']
        measurements.append(b['y'])
    op, restore = frozen_obs.restore_operator('sr4', str(Path(paths['obs'])/'operators'), b['entry'], 'cuda')
    assets = json.loads((Path(paths['core'])/'assets.json').read_text())
    results = []
    with common.offline_network():
        from fluxfm_sampler_ip import FluxFlowMapSampler
        sampler = FluxFlowMapSampler(model_id=assets['base_path'],lora_path=assets['lora_256'],
                                     prompt=list(plan['prompts'].values()))
        sampler.pipeline.transformer.requires_grad_(False)
        sampler.vae.requires_grad_(False)
        sampler.pipeline.transformer.enable_gradient_checkpointing()
        adapter = PaperInverseAdapter(sampler)
        grad = torch.autograd.grad
        grad_norms = []
        def checked_grad(*args, **kwargs):
            values = grad(*args, **kwargs)
            assert all(torch.isfinite(x).all() for x in values)
            grad_norms.append(float(values[0].detach().float().norm()))
            return values
        torch.autograd.grad = checked_grad
        try:
            for size in [1, 4, 8, 16]:
                y = torch.cat(measurements[:size]).cuda()
                torch.manual_seed(0); torch.cuda.manual_seed(0)
                # Same seed0 latent per image, independent of batch grouping.
                one = torch.randn((1,16,256//sampler.vae_scale_factor,256//sampler.vae_scale_factor),
                                  device='cuda',dtype=sampler.dtype)
                latent = one.expand(size,-1,-1,-1).clone()
                grad_norms.clear()
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                begin = time.monotonic()
                try:
                    with frozen_obs.ForbidRegen():
                        pred = adapter.sample(y,op,'sr4',latent=latent,method='J',num_steps=8,
                                              eta=5,nopt=1,t_stop=1)
                    torch.cuda.synchronize()
                    seconds = time.monotonic()-begin
                    assert torch.isfinite(pred).all() and max(grad_norms)>0
                    assert adapter.last_trace['expected_autograd_calls'] == 8
                    r = dict(batch=size,status='PASS',steps=8,seconds=seconds,
                        seconds_per_image_200step_estimate=seconds*25/size,
                        peak_gib=torch.cuda.max_memory_allocated()/2**30,
                        grad_min=min(grad_norms),grad_max=max(grad_norms))
                    del pred
                except torch.cuda.OutOfMemoryError:
                    r = dict(batch=size,status='OOM',steps=8)
                    results.append(r)
                    print('BATCH_PROFILE='+json.dumps(r),flush=True)
                    break
                results.append(r)
                print('BATCH_PROFILE='+json.dumps(r),flush=True)
                del latent,one,y
        finally:
            torch.autograd.grad = grad
    report = dict(status='TECHNICAL_THROUGHPUT_PROBE',full_launch_allowed=False,
        generated_for='tuning-only transient tensors; no images saved or efficacy scores',
        config='J8 exact adapter/source; 200step timing extrapolation needs full-step qualification',
        batch_semantics='sum per-item losses; per-item norms; seed0 latent repeated; GPU numerics can change with batch',
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],results=results,restore=restore,
        total_seconds=time.monotonic()-started)
    (a.out/'profile.json').write_text(json.dumps(report,indent=2)+'\n')
    print('BATCH_PROFILE_SUMMARY='+json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
