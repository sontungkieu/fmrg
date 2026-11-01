"""Independently verify 24 PNG/float outputs and twelve matched pairs."""
import argparse
import base64
import hashlib
import io
import importlib.metadata
import json
import math
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload(t):
    return hashlib.sha256(t.cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def collect(plan_path, first, rest, out):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from safetensors.torch import load_file
    from torchvision import transforms
    from skimage.metrics import peak_signal_noise_ratio
    from frozen_smoke import offline_network
    plan=json.loads(plan_path.read_text())
    phases=[json.loads((d/'summary.json').read_text()) for d in [first,rest]]
    assert [s['completed_pairs'] for s in phases]==[1,11]
    assert all(s['status']=='COMPLETE' and s['full_launch_allowed'] is False for s in phases)
    assert len({s['adapter_sha256'] for s in phases})==len({s['runner_sha256'] for s in phases})==1
    assert all(s['plan_sha256']==sha(plan_path) for s in phases)
    paths=plan['paths']
    metric_source=Path(paths['source'])/'scripts/solve_inverse_problem.py'
    assert sha(metric_source)=='924288546c5c3e027d3a86a9d4a17a65b3bff4a1e25b4590b4806889ba5b8c1b'
    torch.set_num_threads(8)
    os.environ['TORCH_HOME']=str(Path(paths['extra'])/'torch')
    assert (Path(os.environ['TORCH_HOME'])/'hub/checkpoints/vgg16-397923af.pth').is_file()
    import lpips
    with offline_network():
        cli_metric=lpips.LPIPS(net='vgg',verbose=False).cpu().eval().requires_grad_(False)
    cli_transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    manifest=json.loads(Path(paths['obs_manifest']).read_text())
    assert sha(paths['obs_manifest'])=='f0a6b32ea1fd1ba6a1f0ff0e7055680ee32ae39f40f4e9bf44046fd29effac7a'
    index={(r['dataset'],r['split'],r['original_id']):r for r in manifest['rows']}
    for name,digest in manifest['measurement_source']['files'].items():
        assert sha(Path(paths['source'])/name)==digest
    pairs,all_records,pictures=[],[],[]
    for n,cell in enumerate(plan['cells']):
        root=first if n==0 else rest
        pair=json.loads((root/f'pair-{n:02d}.json').read_text())
        pair['pair_record_sha256']=sha(root/f'pair-{n:02d}.json')
        assert pair['index']==n and pair['status']=='COMPLETE'
        assert pair['measurement_payload_sha256']==cell['tensor_payload_sha256']
        target_row=index[(cell['dataset'],cell['split'],cell['original_id'])]
        gt_path=Path(paths['splits'])/target_row['path']
        assert sha(gt_path)==target_row['png_sha256']
        target=torch.from_numpy(np.array(Image.open(gt_path).convert('RGB')).copy()).permute(2,0,1).float().unsqueeze(0)/127.5-1
        gt_pil=Image.open(gt_path).convert('RGB')
        records,values={},{ }
        for track in ['released','paper']:
            stem=f'{n:02d}-{cell["dataset"]}-{cell["task"]}-{cell["method"]}-{track}'
            record_file=root/(stem+'.json')
            r=json.loads(record_file.read_text())
            assert r['status']=='COMPLETE' and r['index']==n and r['track']==track
            for key,value in cell.items(): assert r[key]==value,(n,key)
            cfg=dict(plan[cell['method']])
            if cell['method']=='J': cfg['eta']=cfg['eta'][cell['dataset']]
            assert r['resolved']==cfg and r['prompt']==plan['prompts'][cell['dataset']]
            assert r['measurement_payload_used']==cell['tensor_payload_sha256']
            assert r['initial_latent_sha256']==pair['initial_latent_sha256']
            assert sha(root/r['latent_file'])==r['latent_file_sha256']
            latent=load_file(root/r['latent_file'],device='cpu')['latent']
            assert payload(latent)==r['initial_latent_sha256']
            assert sha(root/r['float_file'])==r['float_file_sha256']
            pred=load_file(root/r['float_file'],device='cpu')['prediction']
            assert pred.dtype==torch.float32 and list(pred.shape)==[1,3,256,256] and torch.isfinite(pred).all()
            assert payload(pred)==r['float_payload_sha256']
            assert sha(root/r['image_file'])==r['image_sha256']
            png=np.array(Image.open(root/r['image_file']).convert('RGB'))
            expected=np.rint(((pred[0].permute(1,2,0).numpy()+1)/2).clip(0,1)*255).astype(np.uint8)
            assert np.array_equal(png,expected),'PNG must exactly quantize saved float prediction'
            # Literal current released CLI metric recipe: PNG/PIL224,ToTensor
            # in [0,1],no Normalize. Keep distinct from float LPIPS[-1,1].
            pred_pil=Image.open(root/r['image_file']).convert('RGB')
            with offline_network(),torch.no_grad():
                cli_lpips=float(cli_metric(cli_transform(gt_pil),cli_transform(pred_pil)).item())
            cli_psnr=float(peak_signal_noise_ratio(
                transforms.ToTensor()(gt_pil).numpy()*255,
                transforms.ToTensor()(pred_pil).numpy()*255,data_range=255))
            pair['tracks'][track]['lpips_released_cli_png224_01_cpu']=cli_lpips
            pair['tracks'][track]['psnr_released_cli_png_cpu']=cli_psnr
            mse=float(((pred.double()-target.double())/2).square().mean())
            rescored=-10*math.log10(max(mse,1e-20))
            assert abs(rescored-r['psnr'])<1e-4,'Independent CPU float64 PSNR check failed'
            assert all(math.isfinite(r[k]) for k in ['psnr','lpips_vgg_256','lpips_vgg_224','seconds','measurement_rmse'])
            assert r['counts']['predict_forward']==r['trace']['expected_predict_calls']
            assert r['counts']['transformer_forward']==r['counts']['predict_forward']
            assert r['counts']['autograd_calls']==r['trace']['expected_autograd_calls']
            if track=='paper' and cell['method']=='E':
                trace=r['trace'];events=trace['events']
                assert abs(trace['effective_data_time_stop']-.67)<1e-7
                assert events[-1]['kind']=='unguided_completion' and events[-1]['nopt']==0
                assert events[-1]['gain_per_inner']==0 and abs(events[-1]['sigma']-.33)<1e-7
                assert events[0]['gain_per_inner']==0 and events[0]['nopt']==0
                for e in events[:-1]:
                    assert e['data_time_next']<=.67+1e-7
                    assert abs(e['gain_per_inner']-cfg['eta']*(1-e['sigma'])*e['sigma_next'])<1e-12
            if track=='released' and cell['method']=='E':
                assert r['trace']['final_interval_guided'] is True
                assert r['trace']['effective_sigma_grid'][-1]==0
            if track=='paper' and cell['method']=='J':
                assert r['trace']['implementation']=='delegate_unchanged_inverse_J'
            records[track]=r;values[track]=pred
            all_records.append(dict(index=n,track=track,record_sha256=sha(record_file),
                float_payload_sha256=r['float_payload_sha256'],psnr_rescore_error=abs(rescored-r['psnr'])))
        assert records['released']['cpu_rng_sha256']==records['paper']['cpu_rng_sha256']
        assert records['released']['cuda_rng_sha256']==records['paper']['cuda_rng_sha256']
        diff=values['paper']-values['released']
        assert abs(float(diff.abs().max())-pair['max_abs_prediction_diff'])<1e-7
        assert abs(float(diff.square().mean().sqrt())-pair['prediction_rmse'])<1e-7
        assert bool(torch.equal(values['paper'],values['released']))==pair['bitwise_equal']
        assert pair['delta_psnr']==records['paper']['psnr']-records['released']['psnr']
        assert pair['delta_lpips_vgg_256']==records['paper']['lpips_vgg_256']-records['released']['lpips_vgg_256']
        pair['value_equal']=pair.pop('bitwise_equal')
        pair['bitwise_equal']=records['paper']['float_payload_sha256']==records['released']['float_payload_sha256']
        pair['clamped_fraction']={t:float((v.abs()>=1).float().mean()) for t,v in values.items()}
        pair['guidance_stop']={t:(records[t]['trace'].get('effective_data_time_stop',1.)) for t in records}
        pair['delta_lpips_released_cli_png224_01_cpu']=pair['tracks']['paper']['lpips_released_cli_png224_01_cpu']-pair['tracks']['released']['lpips_released_cli_png224_01_cpu']
        pairs.append(pair)
        pictures.append((cell,root/records['released']['image_file'],root/records['paper']['image_file']))
    assert len(pairs)==12 and len(all_records)==24
    assert len({(p['dataset'],p['task'],p['method']) for p in pairs})==12
    counters={t:{k:sum(p['tracks'][t]['counts'][k] for p in pairs) for k in pairs[0]['tracks'][t]['counts']} for t in ['released','paper']}
    times={t:sum(p['tracks'][t]['seconds'] for p in pairs) for t in ['released','paper']}
    controls=[p for p in pairs if p['method']=='J']
    report=dict(protocol='fmrg-paper-practical-comparison-report-v1',status='COMPLETE_12_PAIRS_24_OUTPUTS',
        full_launch_allowed=False,expected_pairs=12,completed_pairs=12,completed_outputs=24,failed=[],missing=[],
        plan_sha256=sha(plan_path),adapter_sha256=phases[0]['adapter_sha256'],runner_sha256=phases[0]['runner_sha256'],
        claim=plan['claim'],interpretation=plan['interpretation'],baseline=plan['baseline'],
        settings=plan['settings'],prompts=plan['prompts'],E=plan['E'],J=plan['J'],
        phases=phases,pairs=pairs,artifacts=all_records,counters=counters,sampling_seconds=times,
        phase_total_seconds=sum(p['total_seconds'] for p in phases),
        max_peak_allocated_gib=max(p['tracks'][t]['peak_allocated_gib'] for p in pairs for t in ['released','paper']),
        J_control=dict(pairs=6,bitwise_equal=sum(p['bitwise_equal'] for p in controls),
                       max_abs_prediction_diff=max(p['max_abs_prediction_diff'] for p in controls),
                       max_abs_delta_psnr=max(abs(p['delta_psnr']) for p in controls),
                       equivalence_claim=False),
        validation='All24 float/file/payload/PNG identities; shared latent+RNG+input; independent PSNR <=1e-4dB; E gain/stop trace; actual calls verified. GPU float LPIPS not re-evaluated; separate literal CLI PNG/PIL protocol independently rescored on CPU.',
        metric_source_sha256=sha(metric_source),cpu_metric_versions={k:importlib.metadata.version(k) for k in ['torch','torchvision','lpips','pillow']},
        metric_protocols=dict(gpu_float='LPIPS VGG[-1,1] at256 and float bilinear224,align_cornersFalse,no antialias; diagnostics,not exact CLI preprocessing.',
            released_cli_rescore='Independent CPU VGG on quantized PNG/PIL Resize224/ToTensor[0,1],no Normalize; mirrors current scripts/solve_inverse_problem.py. CPU backend,not a numerical-equivalence claim.',
            psnr='GPU float score independently checked within1e-4dB; quantized PNG source recipe also rescored on CPU.'),
        cpu_tests=plan['cpu_tests'],source_commit=plan['source_commit'],paper=plan['paper'],
        paid_services_used=False,modal_launched=False,company_cost_usd=0,cumulative_cap_usd=10,
        limitations=['Two tuning images and one seed; no efficacy or statistical equivalence claim',
         'Practical E realization is explicit; generic Algorithms5/6 dt/nopt/order and author grid are not recovered',
         'Matched baseline differs from prior norm/pixel flow_map1 smoke; not a causal ablation',
         'Author-exact data,style,tuning,baselines,reward/VLM pipelines and full allocation remain separate'])
    out.mkdir(exist_ok=False)
    (out/'report.json').write_text(json.dumps(report,separators=(',',':'),allow_nan=False)+'\n')
    lines=['# Inverse practical-paper formula: matched smoke','',
        '**12 pairs /24 outputs COMPLETE.** Two reconstructed tuning images,one seed; no full launch or efficacy claim.','',
        '| Dataset | Task | Method | PSNR released | PSNR adapter | Delta | LPIPS256 released | LPIPS256 adapter | CLI LPIPS released | CLI LPIPS adapter |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in pairs:
        x,y=row['tracks']['released'],row['tracks']['paper']
        lines.append(f'| {row["dataset"]} | {row["task"]} | {row["method"]} | {x["psnr"]:.6f} | {y["psnr"]:.6f} | {row["delta_psnr"]:+.6f} | {x["lpips_vgg_256"]:.6f} | {y["lpips_vgg_256"]:.6f} | {x["lpips_released_cli_png224_01_cpu"]:.6f} | {y["lpips_released_cli_png224_01_cpu"]:.6f} |')
    lines+=['', 'J is an A/A control of the unchanged normalized inverse-J source: '+json.dumps(report['J_control'])+'.',
        '', 'Sampling seconds: '+json.dumps(times)+f'; phase execution total {report["phase_total_seconds"]:.3f}s.',
        '', 'Measured calls: '+json.dumps(counters)+'.','',
        'Both tracks use squared/pixel loss,flow_map2,paper class prompts and shared latent/RNG/input bytes. E uses50 nominal grid intervals,eta6,nopt5; J200/eta5(AFHQ) or3(FFHQ)/nopt1.',
        '', 'The reconstructed cutoff policy preserves the shifted full grid and inserts data-time0.67. Released E matches the integer guided count but still guides its jump to data; adapter E finishes with a separate unguided map.',
        '', report['interpretation'],'',report['validation'],'',
        'GPU LPIPS256/float224 use[-1,1]. Additional CLI LPIPS is independently rescored on CPU with the actual released PNG/PIL224/ToTensor[0,1] recipe,no Normalize. These protocols must not be mixed. No parameter was retuned after viewing the smoke.',
        '', 'Full reproduction remains unqualified; see the interpretation contract and report limitations.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    sheet=Image.new('RGB',(512,12*286),'white');draw=ImageDraw.Draw(sheet)
    for n,(cell,left,right) in enumerate(pictures):
        for col,path in enumerate([left,right]):
            x,y=256*col,286*n
            sheet.paste(Image.open(path).convert('RGB'),(x,y+28))
            draw.text((x+3,y+2),f'{cell["dataset"]} {cell["task"]} {cell["method"]}',fill='black')
            draw.text((x+3,y+14),'released' if col==0 else 'practical adapter',fill='black')
    sheet.save(out/'contact-sheet.png')
    # Preview contains all six E pairs,chosen before seeing results; J controls
    # remain in the full sheet and report. This keeps the bridge transfer small.
    esheet=Image.new('RGB',(512,6*286),'white');j=0
    for n,(cell,*_) in enumerate(pictures):
        if cell['method']=='E':
            esheet.paste(sheet.crop((0,n*286,512,(n+1)*286)),(0,j*286));j+=1
    encoded=''
    for width in [256,192]:
        preview=esheet.resize((width,round(esheet.height*width/512)),Image.Resampling.LANCZOS)
        for quality in [70,50,30,15]:
            buffer=io.BytesIO();preview.save(buffer,format='JPEG',quality=quality,optimize=True)
            encoded=base64.b64encode(buffer.getvalue()).decode()
            if len(encoded)<30000: break
        if len(encoded)<30000: break
    assert len(encoded)<30000
    (out/'preview.json').write_text(json.dumps(dict(mime='image/jpeg',sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),data=encoded)))
    print('FMRG_COMPARE_REPORT='+json.dumps(dict(status=report['status'],pairs=12,outputs=24,J_control=report['J_control'],sampling_seconds=times,counters=counters,peak_gib=report['max_peak_allocated_gib'])),flush=True)
    print('FMRG_COMPARE_REPORT_FILES='+json.dumps(dict(directory=str(out),report_sha256=sha(out/'report.json'),report_bytes=(out/'report.json').stat().st_size,preview_bytes=len(encoded))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['plan','first','rest','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();collect(a.plan,a.first,a.rest,a.out)
