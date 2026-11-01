"""One immutable, bounded and resumable inverse campaign worker on one GPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from inverse_campaign import batches, canonical, dump_new, find_complete, safe_relative, sha


def main():
    p = argparse.ArgumentParser()
    for name in ['contract','plan','cells','allocation','out']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',required=True)
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world',type=int,default=1)
    p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--qualify',action='store_true')
    p.add_argument('--qualify-task',choices=['sr4','motion61','mask64'])
    p.add_argument('--qualify-dataset',choices=['AFHQ-Dog','FFHQ'])
    p.add_argument('--max-batches',type=int)
    p.add_argument('--resume',type=Path,action='append',default=[])
    a=p.parse_args()
    c=json.loads(a.contract.read_text()); plan=json.loads(a.plan.read_text()); alloc=json.loads(a.allocation.read_text())
    contract_sha,runner_sha=sha(a.contract),sha(__file__)
    assert plan['contract_sha256']==contract_sha and c['data_claim']=='reconstructed'
    assert alloc['paid_services_allowed'] is False and alloc['cumulative_cap_usd']==10
    assert alloc['contract_sha256']==contract_sha and a.gpu in alloc['gpu_uuids']
    assert alloc['user_authorized_inverse_full'] is True
    deadline=alloc['deadline_unix']
    assert time.time()<deadline
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu
    split='tuning' if a.qualify else 'eval'
    assert a.cells.name==split+'.jsonl' and sha(a.cells)==plan[split+'_jsonl_sha256']
    cells=[json.loads(line) for line in a.cells.read_text().splitlines()]
    assert len(cells)==(600 if a.qualify else 12000)
    assert all(x['split']==('evaluation' if split=='eval' else 'tuning') for x in cells)
    if not a.qualify:
        assert alloc.get('full_launch_allowed') is True
        q=json.loads(Path(alloc['qualification_receipt']).read_text())
        assert sha(alloc['qualification_receipt'])==alloc['qualification_sha256']
        assert q['status']=='PASS_FULL_CONFIG_BATCH'
        assert q['runner_sha256']==runner_sha and q['contract_sha256']==contract_sha
        assert q['batch_size']==a.batch_size and q['full_launch_allowed'] is False
    jobs=batches(cells,a.batch_size,a.rank,a.world)
    if a.qualify:
        # Only first batch for each configured dataset/task/method group.
        seen=set();selected=[]
        for job in jobs:
            key=(job[0]['dataset'],job[0]['task'],job[0]['method'])
            if key not in seen:
                selected.append(job);seen.add(key)
        jobs=selected
        if a.qualify_task:
            jobs=[j for j in jobs if j[0]['task']==a.qualify_task]
        if a.qualify_dataset:
            jobs=[j for j in jobs if j[0]['dataset']==a.qualify_dataset]
    if a.max_batches is not None:
        assert a.max_batches>0
        jobs=jobs[:a.max_batches]
    a.out.mkdir(exist_ok=False);(a.out/'cells').mkdir()
    status='FAILED'; completed=[]; skipped=[]; error=None; timings=[]
    started=time.monotonic()
    summary=dict(protocol='fmrg-inverse-worker-v1',full_launch_allowed=not a.qualify,
        split=split,contract_sha256=contract_sha,runner_sha256=runner_sha,
        plan_sha256=sha(a.plan),cells_sha256=sha(a.cells),allocation_sha256=sha(a.allocation),
        rank=a.rank,world=a.world,batch_size=a.batch_size,gpu_uuid=a.gpu,
        planned_cells=sum(map(len,jobs)),deadline_unix=deadline,claim='reconstructed')
    def timeout(*args):
        raise TimeoutError('ALLOCATION_DEADLINE')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(deadline-time.time())))
    try:
        paths=plan['paths']
        sys.path.insert(0,paths['source']);sys.path.insert(0,paths['loader'])
        sys.path.insert(0,str(a.plan.parent))
        import torch
        import numpy as np
        from PIL import Image
        from safetensors.torch import save_file
        import frozen_obs
        import frozen_smoke as common
        from paper_inverse_adapter import PaperInverseAdapter,scheduler_grid
        assert sha(Path(paths['source'])/'fluxfm_sampler_ip.py')==c['source_sha256']
        assert sha(common.__file__)=='6db23e85ff49b8346e9590961289ad844c5a99b9302f41308bebb7e0deb6754f'
        assert sha(Path(paths['loader'])/'frozen_obs.py')==c['loader_sha256']
        assert sha(Path(__file__).parent/'paper_inverse_adapter.py')==c['adapter_sha256']
        assert sha(paths['obs_manifest'])==c['measurement_manifest_sha256']
        assert sha(Path(paths['core'])/'lora/256/pytorch_lora_weights.safetensors')==c['lora256_sha256']
        assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(10,0)
        assert torch.__version__=='2.7.1+cu128'
        torch.set_num_threads(4)
        assets=json.loads((Path(paths['core'])/'assets.json').read_text())
        with common.offline_network():
            from fluxfm_sampler_ip import FluxFlowMapSampler
            sampler=FluxFlowMapSampler(model_id=assets['base_path'],lora_path=assets['lora_256'],prompt=list(c['prompts'].values()))
            sampler.pipeline.transformer.requires_grad_(False);sampler.vae.requires_grad_(False)
            sampler.pipeline.transformer.enable_gradient_checkpointing()
            adapter=PaperInverseAdapter(sampler)
            counts={};gradient_stats=[]
            orig_predict,orig_decode,orig_grad=sampler.predict_vector,sampler.decode,torch.autograd.grad
            def predict(*args,**kwargs):
                counts['predict_forward']+=1
                return orig_predict(*args,**kwargs)
            def decode(*args,**kwargs):
                counts['vae_decode']+=1
                return orig_decode(*args,**kwargs)
            def gradient(*args,**kwargs):
                values=orig_grad(*args,**kwargs)
                assert all(torch.isfinite(v).all() for v in values)
                counts['autograd_calls']+=1
                gradient_stats.append(float(values[0].detach().float().norm()))
                return values
            def hook(name):
                def h(*args): counts[name]+=1
                return h
            sampler.predict_vector,sampler.decode,torch.autograd.grad=predict,decode,gradient
            sampler.pipeline.transformer.register_forward_pre_hook(hook('transformer_forward'))
            for group in ['transformer_blocks','single_transformer_blocks']:
                for block in getattr(sampler.pipeline.transformer,group):
                    block.register_forward_pre_hook(hook('block_forward_including_recompute'))
            operators={}
            for number,job in enumerate(jobs):
                if timings and time.time()+max(timings)*1.3+60>=deadline:
                    status='DEADLINE_PARTIAL';break
                pending=[]
                for cell in job:
                    if find_complete(a.resume,cell,contract_sha,runner_sha):
                        skipped.append(cell['cell_id'])
                    else: pending.append(cell)
                if not pending: continue
                job=pending;sample=job[0];task=sample['task'];method=sample['method'];cfg=sample['resolved']
                values=[];gt=[]
                for cell in job:
                    entry=cell['observation']
                    tensor_file=safe_relative(paths['obs'],entry['path'])
                    assert sha(tensor_file)==entry['measurement_sha256']
                    state_file=Path(paths['obs'])/'operators'/(task+'.safetensors')
                    config_file=state_file.with_suffix('.json')
                    assert sha(state_file)==entry['operator_state_sha256']
                    assert sha(config_file)==entry['operator_config_sha256']
                    frozen_obs.verify_state_file(state_file,entry)
                    values.append(frozen_obs.verify_tensor_file(tensor_file,entry))
                    image=safe_relative(paths['splits'],cell['image_path'])
                    assert sha(image)==cell['image_sha256']
                    pixels=np.array(Image.open(image).convert('RGB')).copy()
                    assert pixels.shape==(256,256,3)
                    gt.append(torch.from_numpy(pixels).permute(2,0,1).float()/127.5-1)
                if task not in operators:
                    operators[task]=frozen_obs.restore_operator(task,str(state_file.parent),entry,'cuda')
                operator,restore=operators[task]
                y=torch.cat(values).cuda();target=torch.stack(gt).cuda()
                for i,cell in enumerate(job):
                    assert frozen_obs.sha_payload(y[i:i+1])==cell['observation']['tensor_payload_sha256']
                sampler.set_prompt_embeddings(sample['prompt'])
                torch.manual_seed(0);torch.cuda.manual_seed(0)
                one=torch.randn((1,16,256//sampler.vae_scale_factor,256//sampler.vae_scale_factor),device='cuda',dtype=sampler.dtype)
                latent=one.expand(len(job),-1,-1,-1).clone()
                latent_sha=hashlib.sha256(one.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                grid=scheduler_grid(sampler,latent,cfg['num_steps'],256,256).cpu().tolist()
                assert len(grid)==len(plan['reference_grids'][method])
                assert all(abs(x-z)<1e-7 for x,z in zip(grid,plan['reference_grids'][method]))
                cpu_rng_sha=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
                cuda_rng_sha=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
                counts.clear();counts.update({k:0 for k in ['predict_forward','transformer_forward','block_forward_including_recompute','vae_decode','autograd_calls']})
                gradient_stats.clear();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                begin=time.monotonic()
                print('INVERSE_BATCH_START='+canonical(dict(rank=a.rank,batch=number,images=len(job),dataset=sample['dataset'],task=task,method=method)),flush=True)
                with frozen_obs.ForbidRegen():
                    result=adapter.sample(y,operator,task,latent=latent,method=method,
                        num_steps=cfg['num_steps'],eta=cfg['eta'],nopt=cfg['nopt'],t_stop=cfg['t_stop'])
                torch.cuda.synchronize();seconds=time.monotonic()-begin
                assert torch.isfinite(result).all() and gradient_stats and max(gradient_stats)>0
                trace=adapter.last_trace
                assert counts['predict_forward']==trace['expected_predict_calls']==counts['transformer_forward']
                assert counts['autograd_calls']==trace['expected_autograd_calls']
                assert counts['predict_forward']==(78 if method=='E' else 400)
                assert counts['autograd_calls']==(190 if method=='E' else 200)
                pred=result.detach().float().cpu()
                for i,cell in enumerate(job):
                    directory=a.out/'cells'/cell['cell_id'];directory.mkdir(exist_ok=False)
                    assert frozen_obs.sha_payload(y[i:i+1])==cell['observation']['tensor_payload_sha256']
                    assert hashlib.sha256(latent[i:i+1].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==latent_sha
                    save_file({'prediction':pred[i:i+1].contiguous()},str(directory/'prediction.safetensors'))
                    save_file({'latent':one.cpu().contiguous()},str(directory/'latent.safetensors'))
                    png=np.rint(((pred[i].permute(1,2,0).numpy()+1)/2).clip(0,1)*255).astype(np.uint8)
                    Image.fromarray(png).save(directory/'prediction.png')
                    mse=float(((pred[i].double()-target[i].cpu().double())/2).square().mean())
                    rec=dict(status='COMPLETE',cell=cell,contract_sha256=contract_sha,runner_sha256=runner_sha,
                        allocation_sha256=sha(a.allocation),plan_sha256=sha(a.plan),
                        initial_latent_sha256=latent_sha,cpu_rng_sha256=cpu_rng_sha,cuda_rng_sha256=cuda_rng_sha,
                        trace=trace,restore=restore,batch_size=len(job),rank=a.rank,batch=number,
                        batch_sampling_seconds=seconds,per_image_amortized_seconds=seconds/len(job),
                        batch_invocation_counts=dict(counts),effective_invocations_per_image=counts['predict_forward'],
                        gradient_norm_min=min(gradient_stats),gradient_norm_max=max(gradient_stats),
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        psnr_float_diagnostic=-10*math.log10(max(mse,1e-20)),
                        prediction_payload_sha256=hashlib.sha256(pred[i:i+1].contiguous().numpy().tobytes()).hexdigest(),
                        files_sha256={name:sha(directory/name) for name in ['prediction.safetensors','prediction.png','latent.safetensors']})
                    dump_new(directory/'record.json',rec)
                    completed.append(cell['cell_id'])
                timings.append(seconds)
                print('INVERSE_BATCH_COMPLETE='+canonical(dict(rank=a.rank,batch=number,images=len(job),method=method,seconds=seconds,
                    seconds_per_image=seconds/len(job),peak_gib=rec['peak_allocated_gib'],completed=len(completed))),flush=True)
                del y,target,latent,one,result,pred,values,gt
            else:
                status='COMPLETE'
            torch.autograd.grad=orig_grad
    except TimeoutError as exc:
        status='DEADLINE_PARTIAL';error=dict(type=type(exc).__name__,message=str(exc))
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        summary.update(status=status,completed=len(completed),skipped_verified=len(skipped),
                       completed_ids=completed,skipped_ids=skipped,error=error,seconds=time.monotonic()-started)
        dump_new(a.out/'summary.json',summary)
        print('INVERSE_WORKER_SUMMARY='+canonical(summary),flush=True)


if __name__=='__main__':
    main()
