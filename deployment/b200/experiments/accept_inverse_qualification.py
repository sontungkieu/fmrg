"""Independently verify all128 full-config tuning outputs before launch."""
import argparse
import hashlib
import json
from pathlib import Path

from inverse_campaign import dump_new, find_complete, sha


def main():
    p=argparse.ArgumentParser()
    for name in ['root','spec','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scorer-canary',type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.spec.read_text())
    qpath=a.root/'qualification.json';q=json.loads(qpath.read_text())
    assert q['status']=='PASS_FULL_CONFIG_BATCH' and q['images']==128
    canary=json.loads(a.scorer_canary.read_text())
    assert canary['status']=='PASS_TECHNICAL_16_TUNING_OUTPUTS' and canary['dataset_score']=='NOT_COMPUTED'
    plan=json.loads(Path(spec['plan']).read_text())
    path=Path(spec['plan']).parent/'tuning.jsonl'
    assert sha(path)==plan['tuning_jsonl_sha256']
    cells=[json.loads(x) for x in path.read_text().splitlines()]
    roots=[a.root/f'worker-{rank}' for rank in range(4)]
    selected=[]
    for dataset,task in [('AFHQ-Dog','sr4'),('AFHQ-Dog','motion61'),('AFHQ-Dog','mask64'),('FFHQ','sr4')]:
        for method in ['E','J']:
            selected.extend([c for c in cells if (c['dataset'],c['task'],c['method'])==(dataset,task,method)][:16])
    assert len(selected)==128
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    latent_digests=set();times={};peaks=[]
    for cell in selected:
        r=find_complete(roots,cell,spec['contract_sha256'],spec['code_sha256']['run_inverse_campaign.py'])
        assert r is not None and r['batch_size']==16
        folder=next(root/'cells'/cell['cell_id'] for root in roots if (root/'cells'/cell['cell_id']/'record.json').is_file())
        pred=load_file(folder/'prediction.safetensors',device='cpu')['prediction']
        assert pred.dtype==torch.float32 and tuple(pred.shape)==(1,3,256,256) and torch.isfinite(pred).all()
        assert hashlib.sha256(pred.contiguous().numpy().tobytes()).hexdigest()==r['prediction_payload_sha256']
        png=np.rint(((pred[0].permute(1,2,0).numpy()+1)/2).clip(0,1)*255).astype(np.uint8)
        assert np.array_equal(np.array(Image.open(folder/'prediction.png')),png)
        latent=load_file(folder/'latent.safetensors',device='cpu')['latent']
        assert latent.dtype==torch.bfloat16
        digest=hashlib.sha256(latent.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        assert digest==r['initial_latent_sha256'];latent_digests.add(digest)
        assert r['trace']['full_sigma_grid']==plan['reference_grids'][cell['method']]
        counts=r['batch_invocation_counts']
        assert counts['predict_forward']==counts['transformer_forward']==(78 if cell['method']=='E' else 400)
        assert counts['autograd_calls']==(190 if cell['method']=='E' else 200)
        assert r['gradient_norm_max']>0
        key='/'.join([cell['dataset'],cell['task'],cell['method']])
        times[key]=r['per_image_amortized_seconds'];peaks.append(r['peak_allocated_gib'])
    assert len(latent_digests)==1
    result=dict(status='PASS_FULL_CONFIG_BATCH',full_launch_allowed=False,images=128,batch_size=16,
        contract_sha256=spec['contract_sha256'],runner_sha256=spec['code_sha256']['run_inverse_campaign.py'],
        qualification_sha256=sha(qpath),scorer_canary_sha256=sha(a.scorer_canary),
        verifier_sha256=sha(__file__),all_output_files_payloads_and_PNG_verified=True,
        one_seed0_latent_digest=next(iter(latent_digests)),seconds_per_image=times,peak_gib=max(peaks),
        actual_E_forwards=78,actual_J_forwards=400,
        no_efficacy_or_bitwise_equivalence_claim=True)
    dump_new(a.out,result)
    print('INVERSE_QUALIFICATION_ACCEPTED='+json.dumps(result,separators=(',',':')),flush=True)


if __name__=='__main__':main()
