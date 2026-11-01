"""Read-only bounded supervisor/worker status; never relaunch or interrupt."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def alive(pid):
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root
    result=dict(directory=str(root),workers=[])
    for name in ['launch','running','completion','supervisor-error','qualification']:
        path=root/(name+'.json')
        if path.is_file():
            value=json.loads(path.read_text())
            if name=='qualification':
                value={k:v for k,v in value.items() if k!='summaries'}
            if name=='completion':
                value={k:v for k,v in value.items() if k!='summaries'}
            result[name]=value
    for entry in result.get('running',{}).get('workers',[]):
        rank=entry['rank'];r=dict(entry,alive=alive(entry['pid']))
        path=root/f'worker-{rank}.log'
        if path.is_file():
            with path.open('rb') as f:
                f.seek(max(0,path.stat().st_size-7000))
                tail=f.read().decode('utf-8',errors='replace')
            events=[x for x in tail.splitlines() if x.startswith('INVERSE_')]
            r['last_events']=events[-2:]
            if 'Traceback' in tail:r['error_tail']=tail[-2200:]
        r['complete_record_files']=sum(1 for _ in (root/f'worker-{rank}/cells').glob('*/record.json'))
        result['workers'].append(r)
    result['gpu_state']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
    print(json.dumps(result,separators=(',',':')),flush=True)


if __name__=='__main__':main()
