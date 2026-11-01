"""Bounded E/J end-to-end smoke with immutable reconstructed observations.

Uses the released sampler unchanged. This is not a paper-formula reproduction.
Run only on the authorized company B200 through the guarded browser bridge.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

SAMPLER_SHA = '8da781f6902d3c8839603e20fbb8ed0adfc0e5721666806999a106b45bf08cdc'
LOADER_SHA = 'cddf4e21e5c5dc0e4e9ad2ab989417a28c2891d407094a34ba66aade0c262c9f'
LORA_SHA = '1ac550f6800fb086986643475ba9ebbec7bb84bd87b9cab3573ec693332232be'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@contextlib.contextmanager
def offline_network():
    old = socket.socket.connect, socket.create_connection

    def denied(*args, **kwargs):
        raise RuntimeError('TCP_DISABLED_FOR_FROZEN_SMOKE')

    socket.socket.connect = socket.create_connection = denied
    try:
        yield
    finally:
        socket.socket.connect, socket.create_connection = old


def sample_frozen(sampler, measurement, operator, cfg, guard, torch):
    # Match the untouched sampler's RNG/shape exactly, outside the observation
    # regeneration guard. Latent noise is required; observation noise is frozen.
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    shape = (measurement.shape[0], 16, 256 // sampler.vae_scale_factor,
             256 // sampler.vae_scale_factor)
    latent = torch.randn(shape, device=sampler.device, dtype=sampler.dtype)
    with guard():
        return sampler.sample_inverse_problem(
            measurement, operator, cfg['task'], latent=latent,
            num_steps=cfg['num_steps'], guidance_scale=3.5,
            step_size=cfg['step_size'], img_shape=(256, 256),
            grad_mode=cfg['grad_mode'], sample_mode='flow_map1',
            seed=cfg['seed'], num_optim_iters=cfg['num_optim_iters'],
            loss_mode='pixel', loss_func='norm',
            normalize_grad=cfg['normalize_grad'], early_stop=0,
            enable_callback=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--phase', choices=['first', 'rest'], required=True)
    p.add_argument('--first-out', type=Path)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    assert plan['full_launch_allowed'] is False and len(plan['cells']) == 12
    assert plan['company_cost_usd'] == 0 and plan['paid_services_allowed'] is False
    deadline = time.time() + 1800
    if a.phase == 'rest':
        assert a.first_out
        first = json.loads((a.first_out / 'summary.json').read_text())
        assert first['status'] == 'COMPLETE' and first['completed'] == 1
        assert first['plan_sha256'] == sha(a.plan)
        deadline = first['deadline_unix']
    assert deadline > time.time(), 'Original bounded smoke deadline expired'
    signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError('SMOKE_DEADLINE')))
    signal.alarm(max(1, int(deadline - time.time())))
    a.out.mkdir(exist_ok=False)
    started = time.monotonic()
    rows, status, error = [], 'FAILED', None
    paths = plan['paths']
    selected = list(enumerate(plan['cells']))[:1] if a.phase == 'first' else list(enumerate(plan['cells']))[1:]
    summary = dict(protocol='fmrg-frozen-integrated-smoke-v1', full_launch_allowed=False,
                   expected=len(selected), completed=0, phase=a.phase, deadline_unix=deadline,
                   plan_sha256=sha(a.plan), runner_sha256=sha(__file__), claim=plan['claim'])
    try:
        import numpy as np
        import torch
        from PIL import Image
        assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (10, 0)
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
        matching = [r for r in gpu.splitlines() if r.split(',')[0].strip() == plan['gpu_uuid']]
        assert len(matching) == 1 and int(matching[0].split(',')[1]) < 1024, 'GPU lease is not idle'
        assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpu_uuid']
        assert sha(Path(paths['source']) / 'fluxfm_sampler_ip.py') == SAMPLER_SHA
        assert sha(Path(paths['loader']) / 'frozen_obs.py') == LOADER_SHA
        assert sha(Path(paths['core']) / 'lora/256/pytorch_lora_weights.safetensors') == LORA_SHA
        sys.path.insert(0, paths['source'])
        sys.path.insert(0, paths['loader'])
        import frozen_obs
        obs_manifest_path = Path(paths['obs_manifest'])
        assert sha(obs_manifest_path) == 'f0a6b32ea1fd1ba6a1f0ff0e7055680ee32ae39f40f4e9bf44046fd29effac7a'
        obs_manifest = frozen_obs.load_manifest(obs_manifest_path)
        index = {(r['dataset'], r['split'], r['original_id']): r for r in obs_manifest['rows']}
        assert len(index) == 2100 and obs_manifest['claim'] == 'reconstructed'
        assets = json.loads((Path(paths['core']) / 'assets.json').read_text())
        assert Path(assets['base_path']).resolve() == (Path(paths['core']) / 'flux').resolve()
        torch.set_num_threads(8)
        torch.manual_seed(0)
        counts, gradients, calls = {}, [], []
        with offline_network():
            from fluxfm_sampler_ip import FluxFlowMapSampler
            prompts = ['a photo of a dog', 'a photo of a human face']
            sampler = FluxFlowMapSampler(model_id=assets['base_path'], lora_path=assets['lora_256'], prompt=prompts)
            sampler.pipeline.transformer.requires_grad_(False)
            sampler.vae.requires_grad_(False)
            sampler.pipeline.transformer.enable_gradient_checkpointing()
            original_predict, original_decode = sampler.predict_vector, sampler.decode
            original_grad = torch.autograd.grad

            def predict(*args, **kwargs):
                counts['predict_forward'] += 1
                calls.append(dict(sigma_cur=float(args[1]), sigma_endpoint=kwargs.get('t_next')))
                return original_predict(*args, **kwargs)

            def decode(*args, **kwargs):
                counts['vae_decode'] += 1
                return original_decode(*args, **kwargs)

            def grad(*args, **kwargs):
                values = original_grad(*args, **kwargs)
                counts['autograd_calls'] += 1
                assert all(v is not None and torch.isfinite(v).all() for v in values)
                gradients.append(float(values[0].detach().float().norm()))
                return values

            def hook(name):
                def count(*args):
                    counts[name] += 1
                return count

            sampler.predict_vector, sampler.decode = predict, decode
            sampler.pipeline.transformer.register_forward_pre_hook(hook('transformer_forward'))
            for group in ['transformer_blocks', 'single_transformer_blocks']:
                for block in getattr(sampler.pipeline.transformer, group):
                    block.register_forward_pre_hook(hook('observed_block_forward_including_recompute'))
            torch.autograd.grad = grad
            # Metric cache is pre-staged; do not invoke the setup helper or
            # resize 256px predictions to224. LPIPS input must be [-1,1].
            os.environ['TORCH_HOME'] = str(Path(paths['extra']) / 'torch')
            assert (Path(os.environ['TORCH_HOME']) / 'hub/checkpoints/vgg16-397923af.pth').is_file()
            import lpips
            metric = lpips.LPIPS(net='vgg', verbose=False).cuda().eval().requires_grad_(False)
            assert sha(Path(lpips.__file__).parent / 'weights/v0.1/vgg.pth') == sha(Path(paths['extra']) / 'metrics/lpips-v0.1-vgg.pth')
            summary.update(load_seconds=time.monotonic() - started, gpu_uuid=plan['gpu_uuid'],
                           torch=torch.__version__, sampler_sha256=SAMPLER_SHA, loader_sha256=LOADER_SHA,
                           lora_sha256=LORA_SHA, metric='LPIPS VGG v0.1,256px,RGB[-1,1],no resize',
                           source='be485ba40f1d1c163b95ad61112aca6baa13ed22',
                           runtime_wrapper_sha256=sha('/opt/fmrg/python-fmrg.sh'))
            for n, cell in selected:
                assert time.time() < deadline
                row = index[(cell['dataset'], cell['split'], cell['original_id'])]
                assert cell['split'] == 'tuning'
                b = frozen_obs.load_observation(obs_manifest_path, paths['obs'], cell['dataset'], cell['split'], cell['original_id'], cell['task'])
                assert sha(b['tensor_file']) == cell['measurement_sha256'] == b['entry']['measurement_sha256']
                assert sha(b['state_file']) == cell['operator_state_sha256']
                cfg_path = Path(paths['obs']) / 'operators' / (cell['task'] + '.json')
                assert sha(cfg_path) == cell['operator_config_sha256']
                image_path = Path(paths['splits']) / row['path']
                assert sha(image_path) == row['png_sha256']
                image = Image.open(image_path).convert('RGB')
                assert image.size == (256, 256)
                target = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float().unsqueeze(0).cuda() / 127.5 - 1
                measurement = b['y'].cuda()
                op, restore = frozen_obs.restore_operator(cell['task'], str(Path(paths['obs']) / 'operators'), b['entry'], 'cuda')
                assert frozen_obs.sha_payload(measurement) == cell['tensor_payload_sha256']
                cfg = dict(plan[cell['method']], task=cell['task'], seed=cell['seed'])
                if cell['method'] == 'J':
                    cfg['step_size'] = cfg['step_size'][cell['dataset']]
                prompt = prompts[0] if cell['dataset'] == 'AFHQ-Dog' else prompts[1]
                sampler.set_prompt_embeddings(prompt)
                counts.clear()
                counts.update({k: 0 for k in ['predict_forward', 'transformer_forward', 'observed_block_forward_including_recompute', 'vae_decode', 'autograd_calls']})
                gradients.clear()
                calls.clear()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                begin = time.monotonic()
                print('FMRG_CELL_START=' + json.dumps(dict(index=n, dataset=cell['dataset'], task=cell['task'], method=cell['method'], config=cfg)), flush=True)
                result = sample_frozen(sampler, measurement, op, cfg, frozen_obs.ForbidRegen, torch)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - begin
                assert torch.isfinite(result).all()
                assert counts['predict_forward'] == cfg['num_steps']
                assert counts['autograd_calls'] == cfg['num_steps'] * cfg['num_optim_iters']
                assert gradients and max(gradients) > 0
                assert frozen_obs.sha_payload(measurement) == cell['tensor_payload_sha256']
                used_counts = counts.copy()
                sigma_grid = [float(t / 1000) for t in sampler.pipeline.scheduler.timesteps] + [0.0]
                pred = ((result.detach().float() + 1) / 2).clamp(0, 1)
                label = (target + 1) / 2
                with torch.no_grad():
                    mse = float((pred - label).square().mean())
                    lp = float(metric(pred * 2 - 1, target).mean())
                    residual = float((op.A(result.float()) - measurement).square().mean().sqrt())
                assert math.isfinite(lp) and math.isfinite(mse) and math.isfinite(residual)
                filename = f'{n:02d}-{cell["dataset"]}-{cell["task"]}-{cell["method"]}.png'
                Image.fromarray(np.rint(pred[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(a.out / filename)
                record = dict(cell, index=n, status='COMPLETE', prompt=prompt, resolved=cfg,
                              psnr=-10 * math.log10(max(mse, 1e-20)), lpips_vgg=lp, measurement_rmse=residual,
                              seconds=elapsed, counts=used_counts, sigma_grid=sigma_grid,
                              predict_calls=list(calls), gradient_norm_min=min(gradients), gradient_norm_max=max(gradients),
                              peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                              image_file=filename, image_sha256=sha(a.out / filename),
                              gt_png_sha256=row['png_sha256'], restore=restore,
                              measurement_payload_used=frozen_obs.sha_payload(measurement),
                              observation_regeneration_guard='ENABLED_DURING_SAMPLING_LATENT_PRECREATED')
                dump(a.out / f'cell-{n:02d}.json', record)
                rows.append(record)
                print('FMRG_CELL_COMPLETE=' + json.dumps({k: record[k] for k in ['index', 'dataset', 'task', 'method', 'psnr', 'lpips_vgg', 'seconds', 'counts']}), flush=True)
                del result, pred, label, target, measurement, op
            torch.autograd.grad = original_grad
        status = 'COMPLETE'
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        summary.update(status=status, completed=len(rows), error=error, total_seconds=time.monotonic() - started)
        dump(a.out / 'summary.json', summary)
        print('FMRG_PHASE_SUMMARY=' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
