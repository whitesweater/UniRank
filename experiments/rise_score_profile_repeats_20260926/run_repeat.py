"""Repeat the original score-profile probe with one newly frozen checkpoint."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
CASES = {'rankmixer_kuairand': 'RankMixer', 'hiformer_taac25': 'HiFormer'}
ROOT = Path(__file__).resolve().parents[2]
ORIGINALS = {'rankmixer_kuairand': ROOT / 'experiments/rise_discussion_validation_20260925',
             'hiformer_taac25': ROOT / 'experiments/rise_discussion_validation_hiformer_20260925'}
ALLOWED_CONFIG_DIFFERENCES = {'seed', 'dataloader_seed', 'sisa_parameter_seed', 'model_id'}


def load_original(old_dir):
    if old_dir.is_relative_to(HERE) or 'probe_math' in sys.modules:
        raise ValueError('Load the original probe once, from its original experiment')
    sys.path.insert(0, str(old_dir))
    spec = importlib.util.spec_from_file_location('original_score_probe', old_dir / 'run_probe.py')
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    os.chdir(old.ROOT)
    return old


def config_differences(original, new):
    changes = {key: {'original': original.get(key), 'new': new.get(key)}
               for key in original.keys() | new.keys()
               if key not in original or key not in new or original[key] != new[key]}
    if set(changes) - ALLOWED_CONFIG_DIFFERENCES:
        raise ValueError(f'Non-seed forward/protocol configuration differences: {changes}')
    return changes


def prepare(case, checkpoint, training_log):
    old_dir = ORIGINALS[case]
    old = load_original(old_dir)
    baseline_path = old_dir / 'plan.json'
    plan = json.loads(baseline_path.read_text())
    old.engine.DATASET, old.engine.LABELS = plan['dataset'], tuple(old.LABELS)
    params, historical_metrics = old.engine.parse_log(training_log)
    changes = config_differences(plan['training_params'], params)
    checkpoint, training_log = checkpoint.resolve(), training_log.resolve()
    if params['model'] != CASES[case] or params['dataset_id'] != plan['dataset']:
        raise ValueError('Training log model/dataset mismatch')
    if checkpoint.stem != params['model_id'] or training_log.stem != checkpoint.stem:
        raise ValueError('Checkpoint, log and model_id must identify the same training run')
    if params['seed'] == plan['training_params']['seed']:
        raise ValueError('This wrapper is only for an additional checkpoint seed')
    old.verify_watch(plan['frozen_files_sha256'])
    target = HERE / f"{case}_seed{params['seed']}"
    target.mkdir(exist_ok=False)
    config_path = target / 'training_params.json'
    old.engine.atomic_json(config_path, params)
    watched = dict(plan['frozen_files_sha256'])
    for path in [checkpoint, training_log, config_path, baseline_path, old_dir / 'frozen_meta.csv',
                 old_dir / 'run_probe.py', old_dir / 'probe_math.py', old_dir / 'analyze_probe.py',
                 Path(__file__).resolve(), HERE / 'submit_repeat.sbatch']:
        watched[str(path)] = old.engine.sha256(path)
    plan.update({'case': case, 'old_source_dir': str(old_dir), 'config_path': str(config_path),
                 'output_dir': str(target / 'results'), 'training_params': params,
                 'checkpoint': dict(old.engine.file_identity(checkpoint), path=str(checkpoint)),
                 'checkpoint_sha256': watched[str(checkpoint)], 'training_log': str(training_log),
                 'historical_training_log_metrics': historical_metrics,
                 'frozen_files_sha256': watched, 'configuration_differences': changes,
                 'reference_plan_path': str(baseline_path),
                 'replication_selection': 'Additional checkpoint seed, fixed before repeat outcomes',
                 'gates': ['Fresh plain/identity predictions bitwise equal on all original windows',
                           'All routes preserve native QKV, masks, scale and native scores',
                           'Actual bias equals intended once-cast BF16 replacement',
                           'Same cached cohorts; new checkpoint and all frozen inputs verified before and after']})
    old.engine.atomic_json(target / 'plan.json', plan)
    print(json.dumps({'plan': str(target / 'plan.json'), 'configuration_differences': changes}, indent=2))


def run(plan_path):
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text())
    case = plan['case']
    if case not in CASES or plan['model'] != CASES[case]:
        raise ValueError('Unknown case or mismatched model')
    old_dir, output = Path(plan['old_source_dir']).resolve(), Path(plan['output_dir']).resolve()
    if old_dir != ORIGINALS[case] or not plan_path.is_relative_to(HERE) or not output.is_relative_to(HERE) or output.exists():
        raise ValueError('Use the original source and a new output inside this experiment')
    old = load_original(old_dir)
    config_path = Path(plan['config_path']).resolve()
    required = [Path(plan['checkpoint']['path']).resolve(), config_path,
                old_dir / 'run_probe.py', old_dir / 'probe_math.py', old_dir / 'analyze_probe.py',
                old.SOURCE / 'sample_batches.pt', old_dir / 'frozen_meta.csv',
                Path(__file__).resolve(), HERE / 'submit_repeat.sbatch']
    watched = plan['frozen_files_sha256']
    if any(str(path) not in watched for path in required):
        raise ValueError('Plan must freeze checkpoint, config, samples, old source and new runner')
    old.verify_watch(watched)
    if watched[str(required[0])] != plan['checkpoint_sha256']:
        raise ValueError('Checkpoint hash fields disagree')
    if json.loads(config_path.read_text()) != plan['training_params']:
        raise ValueError('Frozen training configuration differs from the plan')
    reference_plan = json.loads((old_dir / 'plan.json').read_text())
    if config_differences(reference_plan['training_params'], plan['training_params']) != plan['configuration_differences']:
        raise ValueError('Recorded configuration differences changed')
    if str(old.SOURCE) != plan['source_dir']:
        raise ValueError('Original cached sample source changed')
    if old.engine.sha256(old_dir / 'frozen_meta.csv') != plan['frozen_meta_csv_sha256']:
        raise ValueError('Original metadata CSV changed')
    is_hi = case == 'hiformer_taac25'
    if is_hi:
        old.historical_diagnostic.wrapper.configure_engine(old.DATASET)
    old.engine.DATASET = old.DATASET if is_hi else old.semantic.DATASET
    old.engine.LABELS = old.LABELS
    samples, meta = old.load_samples()
    meta_hash = (hashlib.sha256(meta.to_csv(index=False).encode()).hexdigest()
                 if is_hi else old.digest_array(meta.to_numpy()))
    if meta_hash != plan['meta_sha256'] or len(meta) != 131072:
        raise ValueError('Cached cohort or metadata changed')
    eligible = meta.eligible.to_numpy(dtype=bool) if is_hi else np.ones(len(meta), dtype=bool)
    if int(eligible.sum()) != (131070 if is_hi else 131072):
        raise ValueError('Original eligible cohort changed')
    prior = json.loads((old.SOURCE / 'manifest.json').read_text())
    for name, identity in prior['data_files'].items():
        if old.engine.file_identity(Path(name)) != identity:
            raise ValueError(f'Original data identity changed: {name}')
    if (not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1 or int(os.environ.get('WORLD_SIZE', 1)) != 1):
        raise RuntimeError('Requires a Slurm allocation, one visible GPU and one process')
    if 'H100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Repeat the original H100 inference protocol')
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('high')
    output.mkdir(parents=True)
    shutil.copyfile(plan_path, output / 'plan.json')
    shutil.copyfile(old_dir / 'frozen_meta.csv', output / 'meta.csv')
    frozen = output / 'frozen_sources'
    frozen.mkdir()
    for source in [old_dir / name for name in ('run_probe.py', 'probe_math.py', 'analyze_probe.py')]:
        shutil.copyfile(source, frozen / source.name)
    shutil.copyfile(Path(__file__), frozen / 'run_repeat.py')
    manifest = {'plan_sha256': old.engine.sha256(plan_path), 'device': old.semantic.device_metadata(),
                'started_git_status': subprocess.check_output(['git', 'status', '--short'], text=True),
                'route_info': {}, 'historical_prediction_gates_replaced':
                'New checkpoint: fresh plain/identity bitwise equivalence; no old predictions reused'}
    gates = {}
    plain, _, _ = old.run_route(plan, samples, 'plain', None, output)
    full, full_raw, full_info = old.run_route(plan, samples, 'identity_hook', None, output)
    gates['identity_vs_plain'] = old.equal_predictions(full, plain)
    gates['identity_vs_plain_bitwise_hash'] = old.prediction_hashes(full) == old.prediction_hashes(plain)
    gates['eight_identity_windows'] = len(full_info) == 8
    cached_mask = np.concatenate([sample['batch'][2].numpy().astype(bool) for sample in samples])
    gates['native_mask_equals_cached_mask'] = bool(np.array_equal(full_raw['native_mask'][:, 0], cached_mask))
    old.engine.atomic_json(output / 'gates.json', gates)
    if not old.passed(gates['identity_vs_plain']) or not all(gates[k] for k in
            ('identity_vs_plain_bitwise_hash', 'eight_identity_windows', 'native_mask_equals_cached_mask')):
        raise RuntimeError('Fresh checkpoint identity or original mask gate failed')
    manifest['fresh_plain_allwindow_prediction_sha256'] = old.prediction_hashes(plain)
    manifest['route_info']['identity_hook'] = full_info
    np.savez_compressed(output / 'attention_full.npz', **full_raw)
    preds = meta.copy()
    for label in old.LABELS:
        preds[f'{label}_full'] = full[label]
    preds.to_csv(output / 'predictions.csv', index=False)
    delta, weights = full_raw['delta'][:, 0], full_raw['weight_after'][:, 0]
    mask, parts = full_raw['native_mask'][:, 0], meta.part_id.to_numpy()
    raw, rms, receipts = old.crosspart_templates(delta, parts, mask)
    replacements = {'local_off': np.zeros_like(delta), 'template_raw': raw.astype(np.float32),
                    'template_rms': rms.astype(np.float32)}
    donor_maps = {}
    for seed in old.DONOR_SEEDS:
        mapping = (old.crosspart_donor_indices(parts, mask, seed) if is_hi
                   else old.crosspart_donor_indices(parts, seed))
        donor_maps[str(seed)] = mapping
        replacement = (old.match_moments(delta[mapping], delta, mask, eligible) if is_hi
                       else old.match_moments(delta[mapping], delta))
        replacements[f'donor_rms_{seed}'] = replacement.astype(np.float32)
    np.savez_compressed(output / 'donor_maps.npz', **donor_maps)
    old.engine.atomic_json(output / 'template_receipts.json', receipts)
    fidelity, diagnostics = [], []
    for route in old.ROUTES[2:]:
        replacement = replacements[route]
        route_pred, route_raw, route_info = old.run_route(plan, samples, route, replacement, output)
        same_native = len(route_info) == len(full_info) == 8 and all(
            item['qkv'] == reference['qkv'] and item['site'] == reference['site']
            for item, reference in zip(route_info, full_info))
        actual = route_raw['delta'][:, 0]
        expected = torch.as_tensor(replacement).to(torch.bfloat16).float().numpy()
        gates[route] = {'all131072_qkv_mask_scale_equal': same_native,
                        'native_scores_equal': bool(np.array_equal(route_raw['native_scores'], full_raw['native_scores'])),
                        'native_mask_equal': bool(np.array_equal(route_raw['native_mask'], full_raw['native_mask'])),
                        'actual_bias_equals_intended_bf16': bool(np.array_equal(actual, expected))}
        if is_hi and route != 'local_off':
            gates[route]['masked_and_ineligible_unchanged'] = bool(
                np.array_equal(actual[~mask], delta[~mask]) and
                np.array_equal(actual[~eligible], delta[~eligible]))
        old.engine.atomic_json(output / 'gates.json', gates)
        if not all(gates[route].values()):
            raise RuntimeError(f'{route}: native-path or intended-bias gate failed')
        for label in old.LABELS:
            preds[f'{label}_{route}'] = route_pred[label]
        preds.to_csv(output / 'predictions.csv', index=False)
        moment_args = (mask[eligible],) if is_hi else ()
        before_mean, before_rms = old.moments(replacement[eligible], *moment_args)
        after_mean, after_rms = old.moments(actual[eligible], *moment_args)
        source_mean, source_rms = old.moments(delta[eligible], *moment_args)
        fidelity.append({'route': route,
            'beforecast_mean_abs_error_to_full': float(np.abs(before_mean - source_mean).max()),
            'beforecast_rms_abs_error_to_full': float(np.abs(before_rms - source_rms).max()),
            'postcast_mean_abs_error_to_intended': float(np.abs(after_mean - before_mean).max()),
            'postcast_rms_abs_error_to_intended': float(np.abs(after_rms - before_rms).max()),
            'postcast_rms_relative_error_p99': float(np.quantile(np.abs(after_rms - before_rms) / np.maximum(before_rms, 1e-12), .99)),
            'actual_bias_sha256': old.digest_array(actual), 'intended_fp32_bias_sha256': old.digest_array(replacement)})
        diag_args = (mask, eligible) if is_hi else ()
        diag = pd.DataFrame(old.row_diagnostics(delta, actual, weights, route_raw['weight_after'][:, 0], *diag_args))
        diag.insert(0, 'row_index', meta.row_index)
        diag.insert(1, 'route', route)
        if is_hi:
            diag.insert(2, 'eligible', eligible)
        diagnostics.append(diag)
        pd.concat(diagnostics, ignore_index=True).to_csv(output / 'row_diagnostics.csv', index=False)
        np.savez_compressed(output / f'attention_{route}.npz', delta=route_raw['delta'], weight=route_raw['weight_after'])
        manifest['route_info'][route] = route_info
        old.engine.atomic_json(output / 'fidelity.json', fidelity)
        old.engine.atomic_json(output / 'manifest.json', manifest)
        del route_raw
        gc.collect()
    old.verify_watch(watched)
    if old.engine.sha256(plan_path) != manifest['plan_sha256']:
        raise ValueError('Plan changed during inference')
    manifest['finished_git_status'] = subprocess.check_output(['git', 'status', '--short'], text=True)
    old.engine.atomic_json(output / 'manifest.json', manifest)
    old.engine.atomic_json(output / 'complete.json', {'completed': True, 'all_gates_passed': True,
        'rows': len(meta), 'eligible_rows': int(eligible.sum()),
        'files_sha256': {str(path.relative_to(output)): old.engine.sha256(path)
                         for path in sorted(output.rglob('*')) if path.is_file()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--training-log', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.prepare and args.case and args.checkpoint and args.training_log and not args.plan:
        prepare(args.case, args.checkpoint, args.training_log)
    elif args.plan and not args.prepare:
        run(args.plan)
    else:
        parser.error('Use --prepare --case --checkpoint --training-log, or --plan')
