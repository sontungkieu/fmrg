"""Matched 12-pair (24 output) practical-paper vs released-IP smoke."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import frozen_smoke as common
from paper_inverse_adapter import PaperInverseAdapter, VERSION, scheduler_grid, cutoff_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--phase', choices=['first', 'rest'], required=True)
    p.add_argument('--first-out', type=Path)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    assert plan['full_launch_allowed'] is False and plan['paid_services_allowed'] is False
    assert len(plan['cells']) == 12 and plan['company_cost_usd'] == 0
    deadline = time.time() + 1800
    if a.phase == 'rest':
        first = json.loads((a.first_out / 'summary.json').read_text())
        assert first['status'] == 'COMPLETE' and first['completed_pairs'] == 1
        assert first['plan_sha256'] == common.sha(a.plan)
        deadline = first['deadline_unix']
    assert time.time() < deadline
    signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError('PAIR_SMOKE_DEADLINE')))
    signal.alarm(max(1, int(deadline-time.time())))
    a.out.mkdir(exist_ok=False)
    start, completed, status, error = time.monotonic(), [], 'FAILED', None
    selected = list(enumerate(plan['cells']))[:1] if a.phase == 'first' else list(enumerate(plan['cells']))[1:]
    paths = plan['paths']
    summary = dict(protocol='fmrg-paper-practical-compare-v1', full_launch_allowed=False,
                   phase=a.phase, expected_pairs=len(selected), completed_pairs=0,
                   plan_sha256=common.sha(a.plan), deadline_unix=deadline,
                   runner_sha256=common.sha(__file__), adapter_version=VERSION,
                   adapter_sha256=common.sha(Path(__file__).parent/'paper_inverse_adapter.py'),
                   common_sha256=common.sha(common.__file__), claim=plan['claim'])
    try:
        import numpy as np
        import torch
        from PIL import Image
        from safetensors.torch import save_file
        assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (10, 0)
        assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpu_uuid']
        state = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
        matching = [r for r in state.splitlines() if r.split(',')[0].strip() == plan['gpu_uuid']]
        assert len(matching) == 1 and int(matching[0].split(',')[1]) < 1024
        assert common.sha(Path(paths['source'])/'fluxfm_sampler_ip.py') == common.SAMPLER_SHA
        assert common.sha(Path(paths['loader'])/'frozen_obs.py') == common.LOADER_SHA
        assert common.sha(Path(paths['core'])/'lora/256/pytorch_lora_weights.safetensors') == common.LORA_SHA
        assert common.sha(paths['obs_manifest']) == 'f0a6b32ea1fd1ba6a1f0ff0e7055680ee32ae39f40f4e9bf44046fd29effac7a'
        sys.path.insert(0, paths['source'])
        sys.path.insert(0, paths['loader'])
        import frozen_obs
        man = frozen_obs.load_manifest(paths['obs_manifest'])
        assert man['claim'] == 'reconstructed'
        index = {(r['dataset'],r['split'],r['original_id']):r for r in man['rows']}
        assert len(index) == 2100
        for name,digest in man['measurement_source']['files'].items():
            assert common.sha(Path(paths['source'])/name) == digest
        assets = json.loads((Path(paths['core'])/'assets.json').read_text())
        torch.set_num_threads(8)
        torch.manual_seed(0)
        counts, gradients, predict_calls = {}, [], []
        with common.offline_network():
            from fluxfm_sampler_ip import FluxFlowMapSampler
            sampler = FluxFlowMapSampler(model_id=assets['base_path'], lora_path=assets['lora_256'],
                                         prompt=list(plan['prompts'].values()))
            sampler.pipeline.transformer.requires_grad_(False)
            sampler.vae.requires_grad_(False)
            sampler.pipeline.transformer.enable_gradient_checkpointing()
            adapter = PaperInverseAdapter(sampler)
            original_predict, original_decode, original_grad = sampler.predict_vector,sampler.decode,torch.autograd.grad
            def predict(*args,**kwargs):
                counts['predict_forward'] += 1
                predict_calls.append(dict(sigma=float(args[1]),target=kwargs.get('t_next')))
                return original_predict(*args,**kwargs)
            def decode(*args,**kwargs):
                counts['vae_decode'] += 1
                return original_decode(*args,**kwargs)
            def grad(*args,**kwargs):
                values=original_grad(*args,**kwargs)
                assert all(v is not None and torch.isfinite(v).all() for v in values)
                counts['autograd_calls'] += 1
                gradients.append(float(values[0].detach().float().norm()))
                return values
            def hook(name):
                def count(*args): counts[name] += 1
                return count
            sampler.predict_vector,sampler.decode,torch.autograd.grad=predict,decode,grad
            sampler.pipeline.transformer.register_forward_pre_hook(hook('transformer_forward'))
            for group in ['transformer_blocks','single_transformer_blocks']:
                for block in getattr(sampler.pipeline.transformer,group):
                    block.register_forward_pre_hook(hook('observed_block_forward_including_recompute'))
            os.environ['TORCH_HOME']=str(Path(paths['extra'])/'torch')
            assert (Path(os.environ['TORCH_HOME'])/'hub/checkpoints/vgg16-397923af.pth').is_file()
            import lpips
            metric=lpips.LPIPS(net='vgg',verbose=False).cuda().eval().requires_grad_(False)
            assert common.sha(Path(lpips.__file__).parent/'weights/v0.1/vgg.pth') == common.sha(Path(paths['extra'])/'metrics/lpips-v0.1-vgg.pth')
            summary.update(load_seconds=time.monotonic()-start,gpu_uuid=plan['gpu_uuid'],
                           torch=torch.__version__,source_sha256=common.SAMPLER_SHA,
                           loader_sha256=common.LOADER_SHA,lora_sha256=common.LORA_SHA)
            for n,cell in selected:
                assert time.time() < deadline and cell['split']=='tuning'
                item=index[(cell['dataset'],cell['split'],cell['original_id'])]
                b=frozen_obs.load_observation(paths['obs_manifest'],paths['obs'],cell['dataset'],cell['split'],cell['original_id'],cell['task'])
                assert common.sha(b['tensor_file'])==cell['measurement_sha256']
                assert common.sha(b['state_file'])==cell['operator_state_sha256']
                assert common.sha(Path(paths['obs'])/'operators'/(cell['task']+'.json'))==cell['operator_config_sha256']
                gt_path=Path(paths['splits'])/item['path']
                assert common.sha(gt_path)==item['png_sha256']
                image=Image.open(gt_path).convert('RGB');assert image.size==(256,256)
                target=torch.from_numpy(np.array(image).copy()).permute(2,0,1).float().unsqueeze(0).cuda()/127.5-1
                y=b['y'].cuda()
                op,restore=frozen_obs.restore_operator(cell['task'],str(Path(paths['obs'])/'operators'),b['entry'],'cuda')
                assert frozen_obs.sha_payload(y)==cell['tensor_payload_sha256']
                cfg=dict(plan[cell['method']])
                if cell['method']=='J': cfg['eta']=cfg['eta'][cell['dataset']]
                sampler.set_prompt_embeddings(plan['prompts'][cell['dataset']])
                torch.manual_seed(cell['seed']);torch.cuda.manual_seed(cell['seed'])
                latent=torch.randn((1,16,256//sampler.vae_scale_factor,256//sampler.vae_scale_factor),device='cuda',dtype=sampler.dtype)
                cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
                latent_sha=hashlib.sha256(latent.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                latent_file=f'{n:02d}-initial-latent.safetensors'
                save_file({'latent':latent.cpu().contiguous()},str(a.out/latent_file))
                grid=scheduler_grid(sampler,latent,cfg['num_steps'],256,256).detach().cpu().tolist()
                guided=cutoff_grid(grid,cfg['t_stop'])
                keep=len(guided)-1 if cell['method']=='E' else 0
                track_records,predictions={},{}
                for track in ['released','paper']:
                    assert time.time() < deadline
                    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
                    counts.clear();counts.update({k:0 for k in ['predict_forward','transformer_forward','observed_block_forward_including_recompute','vae_decode','autograd_calls']})
                    gradients.clear();predict_calls.clear()
                    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                    before=latent.clone()
                    begin=time.monotonic()
                    print('FMRG_PAIR_START='+json.dumps(dict(index=n,dataset=cell['dataset'],task=cell['task'],method=cell['method'],track=track,config=cfg,released_early_stop_count=keep)),flush=True)
                    with frozen_obs.ForbidRegen():
                        if track=='released':
                            result=sampler.sample_inverse_problem(y,op,cell['task'],latent=before,
                                num_steps=cfg['num_steps'],step_size=cfg['eta'],num_optim_iters=cfg['nopt'],
                                guidance_scale=3.5,img_shape=(256,256),grad_mode='euc' if cell['method']=='E' else 'jac',
                                sample_mode='flow_map2',loss_mode='pixel',loss_func='sum',
                                normalize_grad=cell['method']=='J',early_stop=keep,enable_callback=False)
                            trace=dict(implementation='untouched_source',full_sigma_grid=grid,
                                effective_sigma_grid=grid[:keep]+[0.] if keep else grid,
                                integer_early_stop=keep,final_interval_guided=True,
                                expected_predict_calls=2*(keep or cfg['num_steps']),
                                expected_autograd_calls=(keep or cfg['num_steps'])*cfg['nopt'])
                        else:
                            result=adapter.sample(y,op,cell['task'],latent=before,method=cell['method'],
                                num_steps=cfg['num_steps'],eta=cfg['eta'],nopt=cfg['nopt'],t_stop=cfg['t_stop'])
                            trace=adapter.last_trace
                    torch.cuda.synchronize();seconds=time.monotonic()-begin
                    assert torch.isfinite(result).all()
                    actual_grid=[float(t/1000) for t in sampler.pipeline.scheduler.timesteps]+[0.]
                    assert len(actual_grid)==len(grid) and all(abs(x-y)<1e-7 for x,y in zip(actual_grid,grid))
                    assert counts['predict_forward']==trace['expected_predict_calls']
                    assert counts['transformer_forward']==counts['predict_forward']
                    assert counts['autograd_calls']==trace['expected_autograd_calls']
                    assert gradients and max(gradients)>0
                    assert frozen_obs.sha_payload(y)==cell['tensor_payload_sha256']
                    assert hashlib.sha256(before.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==latent_sha
                    used_counts=counts.copy()
                    pred=result.detach().float()
                    with torch.no_grad():
                        mse=float(((pred-target)/2).square().mean())
                        lp256=float(metric(pred,target).mean())
                        lp224=float(metric(torch.nn.functional.interpolate(pred,size=224,mode='bilinear',align_corners=False),torch.nn.functional.interpolate(target,size=224,mode='bilinear',align_corners=False)).mean())
                        residual=float((op.A(pred)-y).square().mean().sqrt())
                    assert all(math.isfinite(v) for v in [mse,lp256,lp224,residual])
                    stem=f'{n:02d}-{cell["dataset"]}-{cell["task"]}-{cell["method"]}-{track}'
                    png=stem+'.png';tensor=stem+'.safetensors'
                    Image.fromarray(np.rint(((pred[0].cpu().permute(1,2,0).numpy()+1)/2).clip(0,1)*255).astype(np.uint8)).save(a.out/png)
                    save_file({'prediction':pred.cpu().contiguous()},str(a.out/tensor))
                    record=dict(cell,index=n,track=track,status='COMPLETE',resolved=cfg,
                        prompt=plan['prompts'][cell['dataset']],initial_latent_sha256=latent_sha,
                        cpu_rng_sha256=hashlib.sha256(cpu_rng.numpy().tobytes()).hexdigest(),
                        cuda_rng_sha256=hashlib.sha256(cuda_rng.cpu().numpy().tobytes()).hexdigest(),
                        latent_file=latent_file,latent_file_sha256=common.sha(a.out/latent_file),
                        measurement_payload_used=frozen_obs.sha_payload(y),restore=restore,trace=trace,
                        psnr=-10*math.log10(max(mse,1e-20)),lpips_vgg_256=lp256,lpips_vgg_224=lp224,
                        measurement_rmse=residual,seconds=seconds,counts=used_counts,predict_calls=list(predict_calls),
                        gradient_norm_min=min(gradients),gradient_norm_max=max(gradients),
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        image_file=png,image_sha256=common.sha(a.out/png),
                        float_file=tensor,float_file_sha256=common.sha(a.out/tensor),
                        float_payload_sha256=frozen_obs.sha_payload(pred))
                    common.dump(a.out/(stem+'.json'),record)
                    track_records[track]=record;predictions[track]=pred.cpu()
                    print('FMRG_TRACK_COMPLETE='+json.dumps({k:record[k] for k in ['index','dataset','task','method','track','psnr','lpips_vgg_256','seconds','counts']}),flush=True)
                    del result,pred,before
                diff=predictions['paper']-predictions['released']
                comparison=dict(index=n,dataset=cell['dataset'],task=cell['task'],method=cell['method'],status='COMPLETE',
                    initial_latent_sha256=latent_sha,measurement_payload_sha256=cell['tensor_payload_sha256'],
                    delta_psnr=track_records['paper']['psnr']-track_records['released']['psnr'],
                    delta_lpips_vgg_256=track_records['paper']['lpips_vgg_256']-track_records['released']['lpips_vgg_256'],
                    max_abs_prediction_diff=float(diff.abs().max()),prediction_rmse=float(diff.square().mean().sqrt()),
                    bitwise_equal=bool(torch.equal(predictions['paper'],predictions['released'])),
                    scientific_equivalence_claim=False,
                    tracks={t:{k:r[k] for k in ['psnr','lpips_vgg_256','lpips_vgg_224','seconds','counts','peak_allocated_gib','image_file','image_sha256','float_file','float_file_sha256']} for t,r in track_records.items()})
                common.dump(a.out/f'pair-{n:02d}.json',comparison)
                completed.append(comparison)
                print('FMRG_PAIR_COMPLETE='+json.dumps({k:comparison[k] for k in ['index','dataset','task','method','delta_psnr','delta_lpips_vgg_256','max_abs_prediction_diff','bitwise_equal']}),flush=True)
                del target,y,op,latent,predictions,track_records,diff
            torch.autograd.grad=original_grad
        status='COMPLETE'
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
        raise
    finally:
        summary.update(status=status,completed_pairs=len(completed),completed_outputs=2*len(completed),
                       error=error,total_seconds=time.monotonic()-start)
        common.dump(a.out/'summary.json',summary)
        print('FMRG_PAIR_PHASE_SUMMARY='+json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
