"""Content-addressed inverse campaign preparation and artifact validation.

All functions in this module use the Python standard library. Neural execution
is separate so plan/completeness/resume guards are testable without Torch.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump_new(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(data, indent=2, allow_nan=False)+'\n')


def safe_relative(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe relative artifact path')
    root = Path(root).resolve()
    target = (root / str(p)).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Artifact escapes declared root')
    return target


def make_cells(man, contract, split):
    assert man['claim'] == 'reconstructed'
    assert split in ['tuning', 'eval']
    expected = contract['tuning_images_per_dataset'] if split == 'tuning' else contract['eval_images_per_dataset']
    manifest_split = 'evaluation' if split == 'eval' else 'tuning'
    rows = [r for r in man['rows'] if r['split'] == manifest_split]
    keys = [(r['dataset'], r['original_id']) for r in rows]
    assert len(set(keys)) == len(keys), 'Duplicate image identity'
    assert Counter(r['dataset'] for r in rows) == {d:expected for d in contract['datasets']}
    other = {(r['dataset'], r['original_id']) for r in man['rows'] if r['split'] != manifest_split}
    assert not (set(keys) & other), 'Tuning/eval leakage'
    cells = []
    # Homogeneous contiguous groups make deterministic batching/sharding simple.
    for dataset in contract['datasets']:
        selected = sorted((r for r in rows if r['dataset']==dataset), key=lambda r:r['original_id'])
        for task in contract['tasks']:
            for method in ['E', 'J']:
                for row in selected:
                    obs = row['observations'][task]
                    for key in ['measurement_sha256','tensor_payload_sha256','operator_state_sha256','operator_config_sha256']:
                        assert len(obs[key]) == 64
                    identity = dict(dataset=dataset,split=manifest_split,original_id=row['original_id'],task=task,method=method,seed=0)
                    cell_id = hashlib.sha256(canonical(identity).encode()).hexdigest()[:24]
                    cfg = dict(contract['methods'][method]); cfg['eta'] = cfg['eta'][dataset]
                    cells.append(dict(identity,cell_id=cell_id,resolved=cfg,prompt=contract['prompts'][dataset],
                        image_path=row['path'],image_sha256=row['png_sha256'],observation=dict(obs)))
    assert len({c['cell_id'] for c in cells}) == len(cells)
    return cells


def batches(cells, batch_size, rank=0, world=1):
    assert isinstance(batch_size,int) and batch_size > 0 and 0 <= rank < world
    groups = {}
    for c in cells:
        groups.setdefault((c['dataset'],c['task'],c['method']),[]).append(c)
    jobs = []
    for key, rows in groups.items():
        for n in range(0,len(rows),batch_size):
            jobs.append(rows[n:n+batch_size])
    # Round-robin by batch: same input plan and size gives disjoint stable shards.
    return jobs[rank::world]


def verify_cell(directory, cell, contract_sha, runner_sha):
    directory = Path(directory)
    record = json.loads((directory/'record.json').read_text())
    assert record['status']=='COMPLETE'
    assert record['contract_sha256']==contract_sha and record['runner_sha256']==runner_sha
    assert record['cell']==cell, 'Cell configuration/measurement identity mismatch'
    for name, digest in record['files_sha256'].items():
        assert sha(safe_relative(directory,name))==digest, 'Artifact hash mismatch'
    assert set(record['files_sha256']) == {'prediction.safetensors','prediction.png','latent.safetensors'}
    return record


def find_complete(roots, cell, contract_sha, runner_sha):
    matches = []
    for root in roots:
        d = Path(root)/'cells'/cell['cell_id']
        if d.exists():
            if not (d/'record.json').is_file():
                raise ValueError('Incomplete prior cell; explicit recovery required')
            matches.append(verify_cell(d,cell,contract_sha,runner_sha))
    if len(matches)>1:
        raise ValueError('Multiple complete attempts for one cell; selection must be explicit')
    return matches[0] if matches else None


def prepare(contract_path, base_plan, out):
    contract = json.loads(contract_path.read_text())
    paths = json.loads(base_plan.read_text())['paths']
    assert sha(paths['obs_manifest'])==contract['measurement_manifest_sha256']
    man = json.loads(Path(paths['obs_manifest']).read_text())
    eval_cells, tuning_cells = make_cells(man,contract,'eval'),make_cells(man,contract,'tuning')
    assert len(eval_cells)==contract['expected_eval_outputs']
    assert len(tuning_cells)==600
    # Source and operator identities are pinned before any evaluation outputs exist.
    for name,digest in man['measurement_source']['files'].items():
        assert sha(Path(paths['source'])/name)==digest
    for task in contract['tasks']:
        entries = [c['observation'] for c in eval_cells if c['task']==task]
        assert len({x['operator_state_sha256'] for x in entries})==1
        assert len({x['operator_config_sha256'] for x in entries})==1
        for suffix,key in [('.json','operator_config_sha256'),('.safetensors','operator_state_sha256')]:
            assert sha(Path(paths['obs'])/'operators'/(task+suffix))==entries[0][key]
    reference_root = Path(paths['obs']).parent
    report_path = reference_root/'fmrg-paper-compare-20261101-v1-report/report.json'
    assert sha(report_path)=='3c0721426c854b196b0354be39e3f6cfbf3f59bec3f418b450975280c85f3cae'
    reference = json.loads(report_path.read_text())
    grids = {}
    for method,index,phase in [('E',0,'first'),('J',1,'rest')]:
        path = reference_root/('fmrg-paper-compare-20261101-v1-'+phase)/f'{index:02d}-AFHQ-Dog-sr4-{method}-paper.json'
        identity = next(a for a in reference['artifacts'] if a['index']==index and a['track']=='paper')
        assert sha(path)==identity['record_sha256']
        grids[method] = json.loads(path.read_text())['trace']['full_sigma_grid']
    out.mkdir(exist_ok=False)
    for split, cells in [('eval',eval_cells),('tuning',tuning_cells)]:
        with (out/(split+'.jsonl')).open('x',encoding='utf-8') as f:
            for c in cells:
                f.write(canonical(c)+'\n')
    dump_new(out/'plan.json', dict(protocol='fmrg-inverse-campaign-plan-v1',full_launch_allowed=False,
        contract_sha256=sha(contract_path),paths=paths,eval_cells=12000,tuning_cells=600,
        eval_jsonl_sha256=sha(out/'eval.jsonl'),tuning_jsonl_sha256=sha(out/'tuning.jsonl'),
        reference_grids=grids,claim='reconstructed',note='Preparation alone does not authorize inference'))
    print('INVERSE_PLAN='+canonical(dict(directory=str(out),plan_sha256=sha(out/'plan.json'),
        eval_sha256=sha(out/'eval.jsonl'),expected_eval=12000,expected_tuning=600)),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--base-plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args(); prepare(a.contract,a.base_plan,a.out)
