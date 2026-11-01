"""Validate all twelve saved smoke cells and build a small inspectable report."""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import struct


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(plan_path, first, rest, out):
    plan = json.loads(plan_path.read_text())
    observation_manifest = json.loads(Path(plan['paths']['obs_manifest']).read_text())
    operator_source_hashes = observation_manifest['measurement_source']['files']
    for name, digest in operator_source_hashes.items():
        assert sha(Path(plan['paths']['source']) / name) == digest, name
    summaries = [json.loads((root / 'summary.json').read_text()) for root in [first, rest]]
    assert [s['completed'] for s in summaries] == [1, 11]
    assert all(s['status'] == 'COMPLETE' and s['full_launch_allowed'] is False for s in summaries)
    assert all(s['plan_sha256'] == sha(plan_path) for s in summaries)
    assert len({s['runner_sha256'] for s in summaries}) == 1
    rows, pictures = [], []
    for n, cell in enumerate(plan['cells']):
        root = first if n == 0 else rest
        file = root / f'cell-{n:02d}.json'
        row = json.loads(file.read_text())
        assert row['index'] == n and row['status'] == 'COMPLETE'
        for key, value in cell.items():
            assert row[key] == value, (n, key)
        assert row['measurement_payload_used'] == cell['tensor_payload_sha256']
        assert row['observation_regeneration_guard'] == 'ENABLED_DURING_SAMPLING_LATENT_PRECREATED'
        image = root / row['image_file']
        assert sha(image) == row['image_sha256']
        header = image.read_bytes()[:24]
        assert header[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II', header[16:24]) == (256, 256)
        expected = plan[cell['method']]
        resolved = dict(expected, task=cell['task'], seed=cell['seed'])
        if cell['method'] == 'J':
            resolved['step_size'] = resolved['step_size'][cell['dataset']]
        assert row['resolved'] == resolved
        assert row['prompt'] == plan['settings']['prompts'][cell['dataset']]
        assert row['counts']['predict_forward'] == expected['num_steps']
        assert row['counts']['transformer_forward'] == expected['num_steps']
        assert row['counts']['autograd_calls'] == expected['num_steps'] * expected['num_optim_iters']
        assert all(math.isfinite(row[k]) for k in ['psnr', 'lpips_vgg', 'seconds', 'measurement_rmse'])
        rows.append(dict(index=n, dataset=row['dataset'], original_id=row['original_id'],
                         task=row['task'], method=row['method'], seed=row['seed'], status=row['status'],
                         psnr=row['psnr'], lpips_vgg=row['lpips_vgg'], measurement_rmse=row['measurement_rmse'],
                         seconds=row['seconds'], peak_allocated_gib=row['peak_allocated_gib'],
                         counts=row['counts'], image_file=row['image_file'], image_sha256=row['image_sha256'],
                         record_sha256=sha(file), measurement_payload_sha256=row['measurement_payload_used'],
                         operator_state_sha256=row['operator_state_sha256'], restore=row['restore']))
        pictures.append(image)
    assert len(rows) == 12 and len({(r['dataset'], r['task'], r['method']) for r in rows}) == 12
    totals = {key: sum(r['counts'][key] for r in rows) for key in rows[0]['counts']}
    report = dict(protocol='fmrg-bounded-integrated-smoke-report-v1', status='COMPLETE_12_OF_12',
                  full_launch_allowed=False, expected=12, completed=12, failed=[], missing=[],
                  claim=plan['claim'], deviations=plan['deviations'], plan_sha256=sha(plan_path),
                  runner_sha256=summaries[0]['runner_sha256'], phases=summaries, rows=rows,
                  operator_source_hashes=operator_source_hashes,
                  counts=totals, sampling_seconds=sum(r['seconds'] for r in rows),
                  phase_total_seconds=sum(s['total_seconds'] for s in summaries),
                  max_peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),
                  metrics='PSNR RGB[0,1] and LPIPS VGG v0.1,256px[-1,1]; measured on float predictions before PNG quantization.',
                  numerical_parity_claim=False, efficacy_claim=False,
                  costs=dict(company_usd=0,modal_launched=False,cumulative_cap_usd=10),
                  scope='Two tuning images,one per dataset; one seed; no selection/tuning/full matrix.',
                  remaining=['Paper-formula sampler adapter and its separate qualification',
                             'flow_map2/time-grid/early-stop contract for paper track',
                             'Tuning objective,selection policy,baselines and full compute budget',
                             'Full allocation/deadline/launch authorization',
                             'Author-exact inputs/style refs/prompts and VLM renoise constant remain unresolved'])
    out.mkdir(exist_ok=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    text = ['# Frozen-input E/J integrated smoke', '', '**12/12 COMPLETE.** Reconstructed tuning-only inputs; not paper-formula reproduction or efficacy.', '',
            '| Dataset | Task | Method | PSNR | LPIPS VGG | Sampling seconds |', '|---|---|---|---:|---:|---:|']
    for r in rows:
        text.append(f'| {r["dataset"]} | {r["task"]} | {r["method"]} | {r["psnr"]:.6f} | {r["lpips_vgg"]:.6f} | {r["seconds"]:.3f} |')
    text += ['', f'Total sampling: {report["sampling_seconds"]:.3f}s; total phase execution including both model loads: {report["phase_total_seconds"]:.3f}s.',
             '', 'Counts: ' + json.dumps(totals) + '.', '', report['metrics'], '',
             'Profile: untouched released sampler, flow_map1, norm/pixel, no early stop. E50steps/5inner iterations/eta6; J200steps/1inner iteration/normalized gradient/eta5(AFHQ) or3(FFHQ).',
             '', 'All twelve image/record hashes, input payload identities, per-cell counters and phase identities verified. Guard remained active during sampling; only seeded initial latent was created before it.',
             '', 'Full launch remains disabled. Missing/failed outputs: none. No Modal or paid services used for this run.']
    (out / 'REPORT.md').write_text('\n'.join(text) + '\n')
    # A contact sheet is a preview; metrics above use full-resolution floats.
    from PIL import Image, ImageDraw
    sheet = Image.new('RGB', (512, 6 * 286), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (r, path) in enumerate(zip(rows, pictures)):
        x, y = (i % 2) * 256, (i // 2) * 286
        sheet.paste(Image.open(path).convert('RGB'), (x, y + 28))
        draw.text((x + 3, y + 2), f'{r["dataset"]} {r["task"]} {r["method"]}', fill='black')
        draw.text((x + 3, y + 14), f'PSNR {r["psnr"]:.2f} LPIPS {r["lpips_vgg"]:.3f}', fill='black')
    sheet.save(out / 'contact-sheet.png')
    preview = sheet.resize((256, 858), Image.Resampling.LANCZOS)
    for quality in [70, 55, 40, 25, 15]:
        buffer = io.BytesIO()
        preview.save(buffer, format='JPEG', quality=quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        if len(encoded) < 30000:
            break
    assert len(encoded) < 30000
    (out / 'preview.json').write_text(json.dumps(dict(mime='image/jpeg', sha256=hashlib.sha256(buffer.getvalue()).hexdigest(), data=encoded)))
    print('FMRG_REPORT=' + json.dumps({k: report[k] for k in ['status', 'completed', 'sampling_seconds', 'phase_total_seconds', 'max_peak_allocated_gib', 'counts']}), flush=True)
    print('FMRG_REPORT_FILES=' + json.dumps(dict(directory=str(out),report_sha256=sha(out/'report.json'),preview_bytes=len(encoded),contact_sheet_sha256=sha(out/'contact-sheet.png'))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['plan', 'first', 'rest', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    collect(a.plan, a.first, a.rest, a.out)
