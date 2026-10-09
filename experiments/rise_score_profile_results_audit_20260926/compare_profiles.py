"""Four fixed-checkpoint rows; never pool users or claim three training seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ('RankMixer', 20262028, 'experiments/rise_discussion_validation_20260925/results_651654'),
    ('RankMixer', 20262029, 'experiments/rise_score_profile_repeats_20260926/rankmixer_kuairand_seed20262029/results'),
    ('HiFormer', 20262028, 'experiments/rise_discussion_validation_hiformer_20260925/results_651691'),
    ('HiFormer', 20262029, 'experiments/rise_score_profile_repeats_20260926/hiformer_taac25_seed20262029/results'),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def loss(y, p):
    assert np.isin(y, [0, 1]).all() and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['RankMixer', 'HiFormer'])
    args = parser.parse_args()
    scheduler = subprocess.check_output(['sacct', '-j', '652762,652763', '--format=JobID,State,ExitCode', '-P', '-n'], text=True)
    status = {parts[0]: parts[1:3] for line in scheduler.splitlines() if (parts := line.split('|')) and len(parts) >= 3}
    needed_jobs = {'RankMixer': '652762', 'HiFormer': '652763'}
    assert all(status.get(job) == ['COMPLETED', '0:0'] for model, job in needed_jobs.items() if args.model in (None, model)), status
    rows, evidence, baseline_inputs = [], {}, {}
    for model, seed, relative in CASES:
        if args.model is not None and model != args.model:
            continue
        source = ROOT / relative
        complete = read(source / 'complete.json')
        assert complete['completed'] and complete['all_gates_passed']
        checked = {}
        for name in ('predictions.csv', 'meta.csv', 'plan.json', 'gates.json', 'manifest.json'):
            actual = sha(source / name)
            assert actual == complete['files_sha256'][name], (model, seed, name)
            checked[name] = actual
        frame = pd.read_csv(source / 'predictions.csv', float_precision='round_trip', low_memory=False)
        meta = pd.read_csv(source / 'meta.csv', low_memory=False)
        assert len(frame) == len(meta) == 131072
        assert np.array_equal(frame.row_index, np.arange(len(frame)))
        fields = [c for c in ('row_index', 'part_id', 'row_offset', 'user_id', 'item_id', 'seq_len', 'effective_length', 'is_click', 'eligible') if c in meta]
        assert all(np.array_equal(frame[c], meta[c]) for c in fields)
        if model in baseline_inputs:
            assert meta[fields].equals(baseline_inputs[model]), (model, 'seed windows or eligibility changed')
        else:
            baseline_inputs[model] = meta[fields].copy()
        selected = frame.loc[frame.eligible].copy() if 'eligible' in frame else frame
        y = selected.is_click.to_numpy()
        losses = {route: loss(y, selected[f'is_click_{route}'].to_numpy(dtype=np.float64)) for route in ('full', 'local_off', 'template_raw', 'template_rms')}
        metrics = pd.read_csv(source / 'analysis/metrics.csv')
        for route, values in losses.items():
            reported = metrics.loc[(metrics.route == route) & (metrics.label == 'is_click'), 'mean_bce'].iloc[0]
            assert abs(values.mean() - reported) < 1e-12
        paired = losses['full'] - losses['template_rms']
        grouped = pd.DataFrame({'user_id': selected.user_id.to_numpy(), 'delta': paired}).groupby('user_id', sort=True).delta.agg(['sum', 'count'])
        saved = pd.read_csv(source / 'analysis/cluster_loss_sums.csv')
        saved = saved.loc[(saved.label == 'is_click') & saved.route.isin(['full', 'template_rms'])]
        pivot = saved.pivot(index='user_id', columns='route', values='loss_sum').sort_index()
        assert np.array_equal(grouped.index.to_numpy(), pivot.index.to_numpy())
        assert np.allclose(grouped['sum'], pivot['full'] - pivot['template_rms'], rtol=0, atol=1e-10)
        counts, sums = grouped['count'].to_numpy(dtype=np.float64), grouped['sum'].to_numpy()
        n_users = len(counts)
        draws = np.random.default_rng(20260925).multinomial(n_users, np.full(n_users, 1 / n_users), size=2000)
        distribution = draws @ sums / (draws @ counts)
        balanced = draws @ (sums / counts) / n_users
        ci = np.quantile(distribution, [.025, .975])
        bci = np.quantile(balanced, [.025, .975])
        summary = read(source / 'analysis/summary.json')
        primary = summary['primary_endpoint']
        expected = [primary['bce_difference'], primary['ci_low'], primary['ci_high'], primary['user_balanced_difference'], primary['user_balanced_ci_low'], primary['user_balanced_ci_high']]
        actual = [float(paired.mean()), *ci.tolist(), float(np.mean(sums / counts)), *bci.tolist()]
        assert np.allclose(actual, expected, rtol=0, atol=1e-12), (model, seed, actual, expected)
        rows.append({'model': model, 'checkpoint_seed': seed, 'rows': len(selected), 'users': n_users,
            'full_click_bce': float(losses['full'].mean()), 'shared_raw_click_bce': float(losses['template_raw'].mean()),
            'shared_rms_click_bce': float(losses['template_rms'].mean()), 'off_click_bce': float(losses['local_off'].mean()),
            'full_minus_rms': actual[0], 'ci_low': actual[1], 'ci_high': actual[2],
            'user_balanced_full_minus_rms': actual[3], 'user_balanced_ci_low': actual[4], 'user_balanced_ci_high': actual[5]})
        evidence[f'{model}_{seed}'] = {'remote_results': str(source), 'checked_input_sha256': checked,
            'summary_sha256': sha(source / 'analysis/summary.json'), 'raw_cluster_recomputation_max_error': float(np.max(np.abs(np.array(actual) - expected))),
            'metadata_equal_across_available_checkpoints': True}
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(args.output / 'comparison.csv', index=False)
    result = {'status': 'independently_verified', 'rows': rows, 'evidence': evidence,
        'scheduler': scheduler, 'script_sha256': sha(__file__),
        'direction': 'BCE(full) minus BCE(shared RMS); positive means shared RMS has lower loss.',
        'scope': 'Two available checkpoints per model evaluated on the same fixed windows; each checkpoint gets its own recalculated profiles and conditional 2000-draw user-cluster interval. No pooling of users across seeds, no three-seed claim, no training-randomness interval.'}
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['| Model | Seed | Full BCE | Shared raw BCE | Shared RMS BCE | Full - RMS [95% CI] |', '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['checkpoint_seed']} | {r['full_click_bce']:.9f} | {r['shared_raw_click_bce']:.9f} | {r['shared_rms_click_bce']:.9f} | {r['full_minus_rms']:+.9f} [{r['ci_low']:+.9f}, {r['ci_high']:+.9f}] |")
    (args.output / 'comparison.md').write_text('\n'.join(lines) + '\n\n' + result['scope'] + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
