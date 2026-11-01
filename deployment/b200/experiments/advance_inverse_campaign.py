"""Fail-closed qualification -> scorer canary -> verified E-stage launch.

Watches the exact existing qualification; never retries or replays that job.
Runs entirely on the authorized instance, including when bridge disconnects.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from inverse_campaign import canonical, dump_new, sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--qualification-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--generation-out',type=Path,required=True)
    p.add_argument('--detach',action='store_true')
    a=p.parse_args();code=Path(__file__).parent
    spec=json.loads(a.spec.read_text());assert spec['mode']=='qualify'
    for name,digest in spec['code_sha256'].items():assert sha(code/name)==digest
    assert sha(spec['plan'])==spec['plan_sha256']
    assert sha(spec['contract'])==spec['contract_sha256']
    base=json.loads(Path(spec['allocation']).read_text())
    assert sha(spec['allocation'])==spec['allocation_sha256']
    assert base['paid_services_allowed'] is False and base['user_authorized_inverse_full'] is True
    assert time.time()<base['deadline_unix']
    if a.detach:
        a.out.mkdir(exist_ok=False)
        with (a.out/'workflow.log').open('xb') as log:
            process=subprocess.Popen(['/opt/fmrg/python-fmrg.sh','-B',str(Path(__file__)),
                '--spec',str(a.spec),'--qualification-root',str(a.qualification_root),
                '--out',str(a.out),'--generation-out',str(a.generation_out)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
        dump_new(a.out/'watcher.json',dict(pid=process.pid,watching_existing_qualification=str(a.qualification_root),
            generation_out=str(a.generation_out),deadline_unix=base['deadline_unix'],
            requested_stage='E6000',driver_sha256=sha(__file__),full_launch_allowed=False))
        print('INVERSE_ADVANCE_WATCHER='+canonical(dict(pid=process.pid,out=str(a.out))),flush=True)
        return
    def event(stage,**extra):
        value=dict(stage=stage,utc_unix=time.time(),**extra)
        with (a.out/'events.jsonl').open('a') as f:f.write(canonical(value)+'\n')
        print('INVERSE_ADVANCE='+canonical(value),flush=True)
    def run(script,args,logname,timeout):
        assert time.time()<base['deadline_unix']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        with (a.out/logname).open('xb') as log:
            subprocess.run(['/opt/fmrg/python-fmrg.sh','-B',str(code/script),*map(str,args)],
                env=env,stdout=log,stderr=subprocess.STDOUT,check=True,
                timeout=min(timeout,max(1,int(base['deadline_unix']-time.time()))))
    try:
        event('WAITING_FOR_EXISTING_QUALIFICATION')
        complete=a.qualification_root/'completion.json'
        while not complete.is_file():
            assert time.time()<base['deadline_unix'],'Allocation expired before qualification'
            error=a.qualification_root/'supervisor-error.json'
            assert not error.is_file(),'Qualification supervisor failed; do not retry'
            time.sleep(5)
        q=json.loads((a.qualification_root/'qualification.json').read_text())
        assert q['status']=='PASS_FULL_CONFIG_BATCH' and q['images']==128
        event('FULL_CONFIG_QUALIFICATION_PASS',images=128)
        roots=[a.qualification_root/f'worker-{r}' for r in range(4)]
        scorer_out=a.out/'scorer-canary'
        run('score_inverse_campaign.py',['--contract',spec['contract'],'--plan',spec['plan'],
            '--launch-spec',a.spec,'--out',scorer_out,'--roots',*roots,
            '--dataset','AFHQ-Dog','--task','sr4','--method','E','--device','cpu','--canary'],
            'scorer-canary.log',600)
        event('SCORER_CANARY_PASS')
        accepted=a.out/'qualification-accepted.json'
        run('accept_inverse_qualification.py',['--root',a.qualification_root,'--spec',a.spec,
            '--scorer-canary',scorer_out/'canary.json','--out',accepted],
            'accept-qualification.log',180)
        receipt=json.loads(accepted.read_text())
        assert receipt['status']=='PASS_FULL_CONFIG_BATCH' and receipt['all_output_files_payloads_and_PNG_verified'] is True
        event('ALL_128_ARTIFACTS_INDEPENDENTLY_VERIFIED')
        alloc=dict(base,full_launch_allowed=True,allowed_methods=['E'],
            qualification_receipt=str(accepted),qualification_sha256=sha(accepted),
            authorization='User authorized inverse E/J; existing4GPU group starts E-only stage, J awaits separate route',
            worker_source_git_commit='1f937c5af730725708db899cd8f78e0c0de3ac66',driver_sha256=sha(__file__))
        allocation_path=a.out/'allocation-E.json';dump_new(allocation_path,alloc)
        full=dict(spec,mode='full',methods=['E'],full_launch_allowed=True,
            allocation=str(allocation_path),allocation_sha256=sha(allocation_path))
        spec_path=a.out/'launch-E.json';dump_new(spec_path,full)
        run('start_inverse_campaign.py',['--spec',spec_path,'--out',a.generation_out],'launch-E.log',120)
        launch=json.loads((a.generation_out/'launch.json').read_text())
        assert launch['expected_outputs']==6000 and launch['full_launch_allowed'] is True
        result=dict(status='FULL_E_STAGE_SUBMITTED_NOT_COMPLETE',full_launch_allowed=True,
            methods=['E'],expected_outputs=6000,generation_out=str(a.generation_out),
            supervisor_pid=launch['pid'],contract_sha256=spec['contract_sha256'],
            plan_sha256=spec['plan_sha256'],allocation_sha256=sha(allocation_path),
            qualification_accepted_sha256=sha(accepted),launch_spec_sha256=sha(spec_path),
            deadline_unix=base['deadline_unix'],J_stage='WAITING_FOR_OTHER_CLUSTER_ROUTE',
            paid_services_used=False,modal_launched=False,cumulative_cap_usd=10)
        dump_new(a.out/'result.json',result);event('FULL_E_STAGE_SUBMITTED',expected_outputs=6000,pid=launch['pid'])
    except BaseException as exc:
        dump_new(a.out/'blocked.json',dict(status='BLOCKED_NO_AUTOMATIC_RETRY',
            type=type(exc).__name__,message=str(exc),full_launch_allowed=False))
        raise


if __name__=='__main__':main()
