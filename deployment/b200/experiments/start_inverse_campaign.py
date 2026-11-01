"""Start a durable supervisor; never replay an unknown launch request."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from inverse_campaign import canonical, dump_new, sha


def worker_args(spec, out, rank, qualify):
    root=Path(__file__).parent
    result=['/opt/fmrg/python-fmrg.sh','-B',str(root/'run_inverse_campaign.py'),
        '--contract',spec['contract'],'--plan',spec['plan'],
        '--cells',str(Path(spec['plan']).parent/('tuning.jsonl' if qualify else 'eval.jsonl')),
        '--allocation',spec['allocation'],'--out',str(out/f'worker-{rank}'),
        '--gpu',spec['gpu_uuids'][rank],'--batch-size',str(spec['batch_size'])]
    if qualify:
        dataset,task=[('AFHQ-Dog','sr4'),('AFHQ-Dog','motion61'),('AFHQ-Dog','mask64'),('FFHQ','sr4')][rank]
        result+=['--qualify','--qualify-dataset',dataset,'--qualify-task',task]
    else:
        result+=['--rank',str(rank),'--world','4']
        for prior in spec.get('resume',[]):result+=['--resume',prior]
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--supervise',action='store_true')
    a=p.parse_args()
    spec=json.loads(a.spec.read_text());qualify=spec['mode']=='qualify'
    assert spec['mode'] in ['qualify','full']
    assert len(spec['gpu_uuids'])==len(set(spec['gpu_uuids']))==4
    assert spec['batch_size']==16
    for name,digest in spec['code_sha256'].items():
        assert sha(Path(__file__).parent/name)==digest
    assert sha(spec['contract'])==spec['contract_sha256']
    assert sha(spec['plan'])==spec['plan_sha256']
    assert sha(spec['allocation'])==spec['allocation_sha256']
    allocation=json.loads(Path(spec['allocation']).read_text())
    assert allocation['gpu_uuids']==spec['gpu_uuids'] and allocation['user_authorized_inverse_full'] is True
    assert time.time()<allocation['deadline_unix']
    if not a.supervise:
        assert shutil.disk_usage(a.out.parent).free>40*2**30,'Need40GiB free before campaign'
        current=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
        memory={line.split(',')[0].strip():int(line.split(',')[1]) for line in current.splitlines()}
        assert all(g in memory and memory[g]<1024 for g in spec['gpu_uuids']), 'Authorized GPU not idle'
        a.out.mkdir(exist_ok=False)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        with (a.out/'supervisor.log').open('xb') as log:
            child=subprocess.Popen(['/opt/fmrg/python-fmrg.sh','-B',str(Path(__file__)),
                '--spec',str(a.spec),'--out',str(a.out),'--supervise'],stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        dump_new(a.out/'launch.json',dict(pid=child.pid,mode=spec['mode'],spec_sha256=sha(a.spec),
            deadline_unix=allocation['deadline_unix'],full_launch_allowed=not qualify,
            expected_outputs=128 if qualify else 12000))
        print('INVERSE_SUPERVISOR_SUBMITTED='+canonical(dict(pid=child.pid,directory=str(a.out),mode=spec['mode'])),flush=True)
        return
    children=[];codes={};summaries=[]
    try:
        for rank,gpu in enumerate(spec['gpu_uuids']):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',
                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_XET='1')
            log=(a.out/f'worker-{rank}.log').open('xb')
            child=subprocess.Popen(worker_args(spec,a.out,rank,qualify),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            log.close();children.append((rank,child))
        dump_new(a.out/'running.json',dict(supervisor_pid=os.getpid(),workers=[dict(rank=r,pid=p.pid) for r,p in children]))
        while children:
            alive=[]
            for rank,child in children:
                code=child.poll()
                if code is None:alive.append((rank,child))
                else:codes[str(rank)]=code
            children=alive
            if children:time.sleep(5)
        for rank in range(4):
            path=a.out/f'worker-{rank}/summary.json'
            summaries.append(json.loads(path.read_text()) if path.is_file() else dict(status='MISSING_SUMMARY',rank=rank))
        good=all(code==0 for code in codes.values()) and all(s['status']=='COMPLETE' for s in summaries)
        ids=[x for s in summaries for x in s.get('completed_ids',[])+s.get('skipped_ids',[])]
        assert len(ids)==len(set(ids)), 'Duplicate cells across shards'
        if qualify:
            good=good and len(ids)==128
            receipt=dict(status='PASS_FULL_CONFIG_BATCH' if good else 'FAIL',full_launch_allowed=False,
                batch_size=16,contract_sha256=spec['contract_sha256'],runner_sha256=spec['code_sha256']['run_inverse_campaign.py'],
                images=len(ids),workers=4,all_three_tasks=True,both_prompts=True,
                exact_E50_J200=True,codes=codes,summaries=summaries)
            dump_new(a.out/'qualification.json',receipt)
        else:
            good=good and len(ids)==12000
        dump_new(a.out/'completion.json',dict(status=('QUALIFIED' if qualify else 'GENERATION_COMPLETE_METRICS_PENDING') if good else 'PARTIAL_OR_FAILED',
            full_launch_allowed=not qualify,codes=codes,completed_or_verified=len(ids),summaries=summaries))
    except BaseException as exc:
        dump_new(a.out/'supervisor-error.json',dict(type=type(exc).__name__,message=str(exc),codes=codes))
        raise


if __name__=='__main__':main()
