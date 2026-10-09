"""Separate, post hoc AUC interval study; raw predictions only, no inference/training."""
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from auc_core import prepare_auc, summarize_interval, draw_user_weights

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCES = ('auc_core.py', 'run_auc.py', 'test_auc.py', 'submit_auc.sbatch')
LABELS = {'RankMixer': ['is_click', 'is_follow', 'is_like', 'is_comment', 'is_forward', 'long_view'],
          'HiFormer': ['is_click', 'is_conversion']}
CASES = [
    ('RankMixer', 20262028, 'experiments/rise_discussion_validation_20260925/results_651654'),
    ('RankMixer', 20262029, 'experiments/rise_score_profile_repeats_20260926/rankmixer_kuairand_seed20262029/results'),
    ('HiFormer', 20262028, 'experiments/rise_discussion_validation_hiformer_20260925/results_651691'),
    ('HiFormer', 20262029, 'experiments/rise_score_profile_repeats_20260926/hiformer_taac25_seed20262029/results'),
]
B = 10000
SEED = 20260926
BATCH = 128


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def environment():
    return {'python': sys.version, 'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn')},
            'threads': {name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}


def source_hashes():
    return {name: sha(HERE / name) for name in SOURCES}


def input_hashes():
    result = {}
    for model, seed, relative in CASES:
        source = ROOT / relative
        complete = read(source / 'complete.json')
        if not complete.get('completed') or not complete.get('all_gates_passed'):
            raise ValueError(f'Incomplete original probe: {model} {seed}')
        for name in ('complete.json', 'plan.json', 'gates.json', 'predictions.csv', 'meta.csv', 'analysis/metrics.csv', 'analysis/summary.json'):
            path = source / name
            actual = sha(path)
            if name in complete['files_sha256'] and actual != complete['files_sha256'][name]:
                raise ValueError(f'Original artifact hash changed: {path}')
            result[str(path)] = actual
    return result


def load_model_cases(model):
    labels = LABELS[model]
    loaded = []
    reference = None
    for actual_model, seed, relative in CASES:
        if actual_model != model:
            continue
        source = ROOT / relative
        original_plan = read(source / 'plan.json')
        original_parameters = original_plan['training_params']
        original_labels = [item['name'] for item in original_parameters['label_col']]
        if original_plan.get('model') != model or original_parameters.get('model') != model or original_parameters.get('seed') != seed or original_labels != labels:
            raise ValueError(f'Original model/checkpoint seed/label-order provenance mismatch: {model} {seed}')
        identity_fields = ['row_index', 'part_id', 'row_offset', 'user_id', 'item_id', 'seq_len', 'effective_length', 'eligible']
        wanted = set(identity_fields + labels + [f'{label}_{route}' for label in labels for route in ('full', 'template_rms')])
        frame = pd.read_csv(source / 'predictions.csv', usecols=lambda name: name in wanted,
                            float_precision='round_trip', low_memory=False)
        meta = pd.read_csv(source / 'meta.csv', usecols=lambda name: name in set(identity_fields + labels), low_memory=False)
        if len(frame) != 131072 or not np.array_equal(frame.row_index, np.arange(131072)):
            raise ValueError('Original fixed windows changed')
        for name in meta:
            if name not in frame or not np.array_equal(frame[name], meta[name]):
                raise ValueError(f'Input/meta mismatch: {model} {seed} {name}')
        if 'eligible' in frame:
            if frame.eligible.dtype != bool:
                raise ValueError('Eligibility is not Boolean')
            frame = frame.loc[frame.eligible].copy()
            meta = meta.loc[meta.eligible].copy()
        if reference is None:
            reference = meta.copy()
        elif not reference.equals(meta):
            raise ValueError('Expected identical retained inputs/labels across checkpoint seeds')
        user_ids, user_index = np.unique(frame.user_id.to_numpy(), return_inverse=True)
        groups, points, inventories, support = {}, {}, {}, {}
        metrics = pd.read_csv(source / 'analysis/metrics.csv')
        for label in labels:
            y = frame[label].to_numpy()
            support[label] = {'positive_rows': int((y == 1).sum()), 'negative_rows': int((y == 0).sum()),
                              'users_with_positive_rows': int(np.unique(user_index[y == 1]).size),
                              'users_with_negative_rows': int(np.unique(user_index[y == 0]).size)}
        for route in ('full', 'template_rms'):
            groups[route], points[route], inventories[route] = [], [], []
            for label in labels:
                y = frame[label].to_numpy()
                p = frame[f'{label}_{route}'].to_numpy(dtype=np.float32)
                grouped = prepare_auc(y, p, user_index, len(user_ids))
                point = float(grouped.evaluate(np.ones((len(user_ids), 1)))[0][0])
                direct = float(roc_auc_score(y, p))
                reported = float(metrics.loc[(metrics.route == route) & (metrics.label == label), 'diagnostic_auc'].iloc[0])
                if not np.isfinite(point) or abs(point - direct) > 1e-12 or abs(point - reported) > 1e-12:
                    raise ValueError(f'Tie-canonicalized point AUC mismatch: {model} {seed} {route} {label}')
                groups[route].append(grouped)
                points[route].append(point)
                inventories[route].append({'label': label, 'score_groups': grouped.n_groups,
                    'representation': grouped.representation, 'point_auc': point, 'point_error_vs_stored': point - reported})
        cohort_hash = hashlib.sha256(meta.to_csv(index=False).encode()).hexdigest()
        loaded.append({'model': model, 'checkpoint_seed': seed, 'rows': len(frame), 'users': len(user_ids),
                       'user_ids': user_ids, 'user_index': user_index, 'groups': groups, 'point_auc': points,
                       'inventory': inventories, 'class_support': support, 'cohort_hash': cohort_hash})
    if len(loaded) != 2 or loaded[0]['cohort_hash'] != loaded[1]['cohort_hash'] or not np.array_equal(loaded[0]['user_ids'], loaded[1]['user_ids']):
        raise ValueError('Shared draw cohort mapping failed')
    return loaded


def plan_summary(case):
    return {key: case[key] for key in ('model', 'checkpoint_seed', 'rows', 'users', 'point_auc', 'inventory', 'class_support', 'cohort_hash')}


def prepare(path):
    if path.exists():
        raise FileExistsError(path)
    inputs = input_hashes()
    cases = [plan_summary(case) for model in LABELS for case in load_model_cases(model)]
    plan = {'schema_version': 1, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'study': 'Exploratory paired user-cluster AUC intervals for fixed Full and shared RMS checkpoints',
        'post_hoc_focal_case': 'HiFormer seed 20262029 was selected after inspecting its AUC/BCE point estimates; no confirmatory or prespecified-study claim',
        'family': [{'model': case['model'], 'checkpoint_seed': case['checkpoint_seed'], 'endpoint': endpoint} for case in cases for endpoint in ('macro_auc', 'click_auc')],
        'primary_endpoints': 'Four macro AUC differences; click AUC is secondary but all eight endpoints included in the correction family',
        'difference': 'AUC(Full) minus AUC(shared RMS); positive favors Full',
        'labels': LABELS, 'bootstrap_draws': B, 'bootstrap_seed': SEED, 'batch_size': BATCH,
        'bootstrap': 'Each model resets default_rng(20260926), draws U users with replacement as multinomial counts; same multiplicities across all routes, labels and both identical-cohort checkpoint seeds within model',
        'macro': 'Equal mean of all six/two task AUCs computed inside each draw; no pooled-label AUC, no averaging of label CI bounds, no nanmean',
        'auc': 'Canonical float32 retained probabilities, ascending exact tie groups, weighted Mann-Whitney numerator with 0.5 credit for ties; original point AUC verified against sklearn and stored metrics',
        'ci': {'method': 'percentile, numpy quantile method=linear', 'nominal': [.025, .975], 'bonferroni8': [.003125, .996875], 'adjusted_level': .99375},
        'decision': 'Full advantage only if adjusted lower bound > 0; RMS advantage only if adjusted upper bound < 0; otherwise unresolved, not equivalence',
        'p_values': 'None; no centered-bootstrap p or exchangeability-based permutation test',
        'invalid_draw_policy': 'Keep all 10000 attempted draws. Undefined class AUC is NaN; any missing label makes macro NaN. Record positive/negative class-drop counts. If any endpoint draw undefined, withhold its intervals and statistical-direction classification; never resample or condition on finite draws.',
        'uncertainty': 'Conditional on these trained checkpoints, chosen windows and estimated shared templates; no training-seed replication uncertainty, population generalization or template re-estimation inside bootstrap',
        'monte_carlo_note': '10000 draws give about31 observations per adjusted tail; reported bootstrap endpoints have Monte Carlo uncertainty and are not exact coverage guarantees',
        'cases': cases, 'environment': environment(), 'input_sha256': inputs, 'source_sha256': source_hashes(),
        'no_training': True, 'no_new_inference': True, 'no_gpu': True}
    plan['plan_hash'] = json_sha({k: v for k, v in plan.items() if k != 'prepared_at_utc'})
    write(path, plan)
    print(json.dumps({'prepared': str(path), 'plan_hash': plan['plan_hash'], 'cases': cases}, indent=2))


def run(path, output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run the full bootstrap in a CPU Slurm allocation')
    if output.exists() or not output.resolve().is_relative_to(HERE):
        raise ValueError('Use a new result directory inside this separate study')
    plan = read(path)
    if json_sha({k: v for k, v in plan.items() if k not in ('plan_hash', 'prepared_at_utc')}) != plan['plan_hash']:
        raise ValueError('Plan content hash mismatch')
    if source_hashes() != plan['source_sha256'] or input_hashes() != plan['input_sha256']:
        raise ValueError('Frozen inputs or source drift')
    current_environment = environment()
    if current_environment['python'] != plan['environment']['python'] or current_environment['packages'] != plan['environment']['packages']:
        raise ValueError('Frozen software environment drift')
    output.mkdir()
    write(output / 'plan.json', plan)
    start = time.perf_counter()
    results, array_records, invalid_records, model_records = [], {}, {}, {}
    for model in LABELS:
        cases = load_model_cases(model)
        planned = [v for v in plan['cases'] if v['model'] == model]
        if [plan_summary(v) for v in cases] != planned:
            raise ValueError('Loaded cohort, class supports or points differ from plan')
        n_labels, n_users = len(LABELS[model]), cases[0]['users']
        values = np.full((B, 2, 2, n_labels), np.nan, dtype=np.float64)
        no_positive = np.zeros((B, n_labels), dtype=bool)
        no_negative = np.zeros((B, n_labels), dtype=bool)
        draw_hash = hashlib.sha256()
        model_start = time.perf_counter()
        for begin, weights in draw_user_weights(n_users, B, SEED, BATCH):
            end = begin + len(weights)
            draw_hash.update(np.ascontiguousarray(weights, dtype=np.int64).tobytes())
            wt = np.ascontiguousarray(weights.T, dtype=np.float64)
            reference_counts = {}
            for checkpoint_index, case in enumerate(cases):
                for route_index, route in enumerate(('full', 'template_rms')):
                    for label_index, grouped in enumerate(case['groups'][route]):
                        auc, pos, neg = grouped.evaluate(wt)
                        values[begin:end, checkpoint_index, route_index, label_index] = auc
                        if label_index not in reference_counts:
                            reference_counts[label_index] = (pos, neg)
                            no_positive[begin:end, label_index] = pos == 0
                            no_negative[begin:end, label_index] = neg == 0
                        elif not np.array_equal(pos, reference_counts[label_index][0]) or not np.array_equal(neg, reference_counts[label_index][1]):
                            raise ValueError('Class support differs across paired routes or identical-cohort seeds')
            if begin == 0 or end == B or end % 1024 == 0:
                write(output / 'progress.json', {'model': model, 'completed_draws': end, 'total_draws': B,
                    'seconds': time.perf_counter() - model_start, 'no_interval_outcomes_in_progress': True})
        np.savez_compressed(output / f'{model.lower()}_draw_auc.npz', auc=values, no_positive=no_positive, no_negative=no_negative)
        array_records[model] = {'path': f'{model.lower()}_draw_auc.npz', 'sha256': sha(output / f'{model.lower()}_draw_auc.npz'),
            'axis_order': 'draw, checkpoint_seed[20262028,20262029], route[full,template_rms], original_label_order',
            'user_draw_stream_sha256_int64': draw_hash.hexdigest()}
        invalid_records[model] = {label: {'no_positive_draws': int(no_positive[:, i].sum()), 'no_negative_draws': int(no_negative[:, i].sum()),
            'undefined_draws': int((no_positive[:, i] | no_negative[:, i]).sum())} for i, label in enumerate(LABELS[model])}
        for checkpoint_index, case in enumerate(cases):
            for endpoint in ('macro_auc', 'click_auc'):
                if endpoint == 'macro_auc':
                    # Ordinary mean deliberately propagates any missing task.
                    full, rms = values[:, checkpoint_index, 0].mean(axis=1), values[:, checkpoint_index, 1].mean(axis=1)
                    point_full, point_rms = np.mean(case['point_auc']['full']), np.mean(case['point_auc']['template_rms'])
                else:
                    full, rms = values[:, checkpoint_index, 0, 0], values[:, checkpoint_index, 1, 0]
                    point_full, point_rms = case['point_auc']['full'][0], case['point_auc']['template_rms'][0]
                results.append({'model': model, 'checkpoint_seed': case['checkpoint_seed'], 'endpoint': endpoint,
                    'rows': case['rows'], 'users': n_users, 'full_point_auc': float(point_full), 'rms_point_auc': float(point_rms),
                    **summarize_interval(full-rms, point_full-point_rms)})
        model_records[model] = {'seconds': time.perf_counter() - model_start, 'identical_cohort_draws_shared_across_checkpoint_seeds': True}
        del cases, values
    if len(results) != 8:
        raise ValueError('Missing prespecified endpoints')
    if source_hashes() != plan['source_sha256'] or input_hashes() != plan['input_sha256']:
        raise ValueError('Frozen inputs or source changed during bootstrap; results cannot be certified')
    result = {'status': 'complete', 'plan_hash': plan['plan_hash'], 'endpoints': results,
        'focal_HiFormer_seed20262029': [r for r in results if r['model'] == 'HiFormer' and r['checkpoint_seed'] == 20262029],
        'focal_selection': plan['post_hoc_focal_case'], 'invalid_class_draws': invalid_records,
        'saved_bootstrap_arrays': array_records, 'model_timings': model_records,
        'total_seconds': time.perf_counter() - start, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'environment': current_environment,
        'source_and_input_hashes_unchanged_at_end': True,
        'uncertainty_scope': plan['uncertainty'], 'monte_carlo_note': plan['monte_carlo_note'],
        'no_training': True, 'no_new_inference': True, 'no_gpu': True}
    write(output / 'auc_intervals.json', result)
    lines = ['This is a post hoc exploratory comparison of fixed predictions; positive AUC(Full) - AUC(RMS) favors Full.', '',
        '| Model | Seed | Endpoint | Full AUC | RMS AUC | Difference | 95% interval | 99.375% Bonferroni-8 interval | Adjusted direction |',
        '|---|---:|---|---:|---:|---:|---|---|---|']
    for r in results:
        fmt = lambda interval: 'withheld (undefined draw)' if interval is None else f'[{interval[0]:+.9f}, {interval[1]:+.9f}]'
        lines.append(f"| {r['model']} | {r['checkpoint_seed']} | {r['endpoint']} | {r['full_point_auc']:.9f} | {r['rms_point_auc']:.9f} | {r['point_difference_full_minus_rms']:+.9f} | {fmt(r['ci95'])} | {fmt(r['ci99_375_bonferroni8'])} | {r['adjusted_direction']} |")
    lines += ['', plan['post_hoc_focal_case'], '', plan['uncertainty'], '', plan['monte_carlo_note'], '',
        'No p-values are reported. An interval crossing zero means unresolved, not equivalent. No labels or missing-class draws are silently omitted.']
    (output / 'auc_intervals.md').write_text('\n'.join(lines) + '\n')
    write(output / 'complete.json', {'completed': True, 'plan_hash': plan['plan_hash'],
        'files_sha256': {str(p.relative_to(output)): sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'completed': True, 'output': str(output), 'total_seconds': result['total_seconds'], 'endpoints': results}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--plan', type=Path, default=HERE / 'plan.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.plan)
    elif args.output is None:
        parser.error('--output is required for --run')
    else:
        run(args.plan, args.output)
