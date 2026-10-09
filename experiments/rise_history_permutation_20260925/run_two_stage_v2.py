"""Faithful native-score capture then original-input replay; no new inside hooks."""
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_permutation import (ROOT, CASES, ROUTES, prior_module, plan_path, load_model,
                             permutations, native_numerical_check)
from permutation_math import joint_raw_batch, digest
import fidelity_v2

OWN_FILES = ('run_two_stage_v2.py', 'fidelity_v2.py', 'submit_fidelity_v2.sbatch',
             'submit_two_stage_v2.sbatch', 'README_v2.md')


def new_plan_path(case):
    return HERE / f'plan_two_stage_v2_{case}.json'


def prepare(case):
    old = prior_module(case)
    prior = json.loads(plan_path(case).read_text())
    old.verify_watch(prior['frozen_files_sha256'])
    if not fidelity_v2.fidelity_plan(case).exists():
        fidelity_v2.prepare(case)
    watched = dict(prior['frozen_files_sha256'])
    for path in [plan_path(case), fidelity_v2.fidelity_plan(case)] + [HERE / name for name in OWN_FILES]:
        watched[str(path)] = old.engine.sha256(path)
    plan = dict(prior)
    plan.update(schema_version=2, protocol_revision='native_capture_original_input_replay_v2',
        failed_v1='Identity fidelity failed before any permutation outcomes; preserved under failure_archive/array_652246',
        first_window_preflight='Run fidelity_v2 original/capture/replay/zero on first16384 rows; abort unless all strict gates pass',
        route_definition='StageA jointly permutes raw history item attributes and masks outside compiled model, using only historical audited dot Observer to capture native BF16 Delta. Inverse-align captured key-axis Delta. StageB replays it on original raw records with the same audited Observer replacement tensor; Q/K/V/mask/scale unchanged.',
        identity_definition='Capture original native Delta and replay it on original records with audited Observer; must match historical/full original predictions and bias bitwise',
        off_definition='Replay zeros_like on original records with audited Observer; bitwise match historical local_off on all rows',
        native_content_alignment='During StageA, inverse-align actual captured native K/V outside forward and require original hash equality; candidate query and mask also exact',
        inside_model_hooks='Only unchanged old run_probe.Observer at dot_attention; NO SISA or pooling hooks in main routes',
        secondary_endpoints=['all-row click BCE', 'all other labels', 'user-balanced click sensitivity',
                             'centered correction change, attention TV and prediction differences'],
        pooling_measurement='No main-route pooling capture; separate64-row eager native numerical check only',
        frozen_files_sha256=watched)
    path = new_plan_path(case)
    if path.exists():
        raise FileExistsError(path)
    old.engine.atomic_json(path, plan)
    print(json.dumps({'case': case, 'plan': str(path), 'sha256': old.engine.sha256(path)}), flush=True)


def capture_native_permuted(old, case, plan, samples, mappings, original_info, output, route):
    """Only old Observer hook; raw permutation and content audit are outside forward."""
    model = load_model(old, case, plan)
    observer = old.Observer(model, 'identity_hook')
    torch._dynamo.reset()
    deltas, receipts = [], []
    try:
        for window, sample in enumerate(samples):
            mapping = torch.as_tensor(mappings[window], dtype=torch.long)
            changed = joint_raw_batch(sample['batch'], mapping)
            with torch.no_grad(), model._get_amp_context():
                model.forward(old.semantic.tree_map(changed, lambda tensor: tensor.clone()))
            if len(observer.entries) != 1:
                raise AssertionError('Expected one original Observer capture')
            args, kwargs = observer.entries[0]
            inverse = mapping.argsort(1).to(args[1].device)
            aligned = {'query': old.tensor_digest(args[0])}
            for name, tensor in zip(('key', 'value'), args[1:]):
                indices = inverse[:, None, :, None].expand_as(tensor)
                aligned[name] = old.tensor_digest(tensor.gather(-2, indices))
            if aligned != original_info[window]['qkv']:
                raise AssertionError(f'{route} changed content Q/K/V after inverse alignment')
            if old.tensor_digest(kwargs['mask']) != original_info[window]['site']['mask']:
                raise AssertionError('Raw permutation changed fixed valid positions')
            raw, info = observer.consume(sample['selected_indices'])
            delta = raw['delta'][:, 0]
            aligned_delta = np.take_along_axis(delta, np.argsort(mappings[window], axis=1), axis=1)
            deltas.append(aligned_delta)
            receipts.append({'original_order_qkv_recovered_bitwise': True,
                             'aligned_qkv': aligned, 'captured_native': info,
                             'captured_delta_sha256': digest(delta), 'aligned_delta_sha256': digest(aligned_delta)})
            old.engine.atomic_json(output / 'progress.json', {'stage': 'native_raw_permutation_capture',
                'route': route, 'windows_completed': window + 1})
            del changed, raw, args, kwargs
    finally:
        observer.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return np.concatenate(deltas), receipts


def same_native(info, reference):
    return len(info) == len(reference) and all(a['qkv'] == b['qkv'] and a['site'] == b['site']
                                             for a, b in zip(info, reference))


def run(case, output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Requires one allocated H100 GPU')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('One inference process required')
    allowed = ROOT / 'artifacts/rise_history_permutation_20260925'
    if output.exists() or not output.resolve().is_relative_to(allowed):
        raise ValueError('New isolated version2 output directory required')
    output.mkdir(parents=True)
    old = prior_module(case)
    plan = json.loads(new_plan_path(case).read_text())
    old.verify_watch(plan['frozen_files_sha256'])
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('high')
    old.engine.DATASET, old.engine.LABELS = plan['dataset'], tuple(plan['labels'])
    started = time.perf_counter()
    fidelity_v2.run(case, output / 'first_window_fidelity')
    preflight = json.loads((output / 'first_window_fidelity/complete.json').read_text())
    if not preflight.get('all_gates_passed'):
        raise AssertionError('First-window fidelity did not pass')
    samples, meta = old.load_samples()
    meta['history_permutation_eligible'] = meta.effective_length >= 2
    if hashlib.sha256(meta.to_csv(index=False).encode()).hexdigest() != plan['meta_csv_sha256']:
        raise AssertionError('Fixed window metadata changed')
    maps = permutations(samples)
    if {route: [digest(value) for value in windows] for route, windows in maps.items()} != plan['permutation_hashes']:
        raise AssertionError('Permutation definitions changed')
    meta.to_csv(output / 'meta.csv', index=False)
    shutil.copyfile(new_plan_path(case), output / 'plan.json')
    (output / 'frozen_sources').mkdir()
    for name in OWN_FILES:
        shutil.copyfile(HERE / name, output / 'frozen_sources' / name)
    np.savez_compressed(output / 'permutations.npz', **{route: np.concatenate(maps[route]) for route in ('reverse', 'random')})
    previous = pd.read_csv(Path(plan['prior_probe_results']) / 'predictions.csv', float_precision='round_trip', low_memory=False)
    gates = {'first_window_fidelity_passed': True}
    def save_gates():
        old.engine.atomic_json(output / 'gates.json', gates)
    plain, _, _ = old.run_route(plan, samples, 'plain', None, output)
    reference, reference_raw, reference_info = old.run_route(plan, samples, 'identity_hook', None, output)
    gates['native_capture_vs_fresh_plain_allrows'] = old.equal_predictions(reference, plain)
    gates['native_capture_vs_prior_probe_full_allrows'] = old.equal_predictions(reference,
        {label: previous[label + '_full'].to_numpy(dtype=np.float32) for label in old.LABELS})
    save_gates()
    if not old.passed(gates['native_capture_vs_prior_probe_full_allrows']) or not old.passed(gates['native_capture_vs_fresh_plain_allrows']):
        raise AssertionError('Audited full-window capture no longer matches historical predictions')
    selected = np.concatenate([sample['selected_indices'] + 16384 * window for window, sample in enumerate(samples)])
    historical_name = 'predictions.csv' if case == 'rankmixer_kuairand' else 'prediction_check.csv'
    historical = pd.read_csv(old.SOURCE / historical_name, float_precision='round_trip')
    gates['plain_vs_historical_attention512'] = old.equal_predictions(
        {label: value[selected] for label, value in plain.items()},
        {label: historical[label + '_plain'].to_numpy(dtype=np.float32) for label in old.LABELS})
    source_raw = np.load(old.SOURCE / 'raw_attention.npz')
    gates['historical_attention512_raw_bitwise'] = all(np.array_equal(reference_raw[key][selected], source_raw[key], equal_nan=True)
        for key in reference_raw if key in source_raw.files and key != 'native_mask')
    if not gates['historical_attention512_raw_bitwise'] or not old.passed(gates['plain_vs_historical_attention512']):
        save_gates()
        raise AssertionError('Historical native attention capture changed')
    replacement = reference_raw['delta'][:, 0]
    values, raw, info = old.run_route(plan, samples, 'order_identity_replay', replacement, output)
    gates['identity_vs_plain_allrows'] = old.equal_predictions(values, plain)
    gates['identity_vs_prior_probe_full_allrows'] = old.equal_predictions(values,
        {label: previous[label + '_full'].to_numpy(dtype=np.float32) for label in old.LABELS})
    gates['identity_replay_actual_bias_exact'] = bool(np.array_equal(raw['delta'], reference_raw['delta']))
    if not old.passed(gates['identity_vs_plain_allrows']) or not gates['identity_replay_actual_bias_exact'] or not same_native(info, reference_info):
        save_gates()
        raise AssertionError('Full-window identity replay failed')
    manifest = {'case': case, 'protocol_revision': plan['protocol_revision'], 'no_training': True,
        'device': old.semantic.device_metadata(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'plan_sha256': old.engine.sha256(new_plan_path(case)), 'route_info': {'identity': info}, 'capture_alignment': {}}
    preds = meta.copy()
    deltas = {'identity': raw['delta'][:, 0]}
    diagnostics = []
    def record(route, prediction, captured):
        for label in old.LABELS:
            preds[label + '_' + route] = prediction[label]
        preds.to_csv(output / 'predictions.csv', index=False)
        delta = captured['delta'][:, 0]
        deltas[route] = delta
        mask = captured['native_mask'][:, 0].astype(bool)
        count = np.maximum(mask.sum(1), 1)
        centered = delta - (delta * mask).sum(1, keepdims=True) / count[:, None]
        baseline = reference_raw['delta'][:, 0]
        centered_baseline = baseline - (baseline * mask).sum(1, keepdims=True) / count[:, None]
        diagnostics.append(pd.DataFrame({'row_index': meta.row_index, 'route': route,
            'centered_delta_difference_rms': np.sqrt((((centered - centered_baseline) ** 2) * mask).sum(1) / count),
            'attention_tv': .5 * np.abs(captured['weight_after'][:, 0] - reference_raw['weight_after'][:, 0]).sum(1)}))
    record('identity', values, raw)
    # Full original/off gates complete BEFORE any nonidentity permutation runs.
    values, raw, info = old.run_route(plan, samples, 'order_off_replay', np.zeros_like(replacement), output)
    gates['off_vs_prior_probe_local_off_allrows'] = old.equal_predictions(values,
        {label: previous[label + '_local_off'].to_numpy(dtype=np.float32) for label in old.LABELS})
    gates['off'] = {'native_qkv_mask_scale_bitwise': same_native(info, reference_info),
                    'native_scores_bitwise': bool(np.array_equal(raw['native_scores'], reference_raw['native_scores'])),
                    'actual_bias_equals_intended': bool(np.count_nonzero(raw['delta']) == 0)}
    save_gates()
    if not old.passed(gates['off_vs_prior_probe_local_off_allrows']) or not all(gates['off'].values()):
        raise AssertionError('Full-window off replay failed')
    manifest['route_info']['off'] = info
    record('off', values, raw)
    for route in ('reverse', 'random'):
        aligned, alignment = capture_native_permuted(old, case, plan, samples, maps[route], reference_info, output, route)
        manifest['capture_alignment'][route] = alignment
        values, raw, info = old.run_route(plan, samples, f'order_{route}_replay', aligned, output)
        intended = torch.as_tensor(aligned).to(torch.bfloat16).float().numpy()
        gates[route] = {'native_qkv_mask_scale_bitwise': same_native(info, reference_info),
            'native_scores_bitwise': bool(np.array_equal(raw['native_scores'], reference_raw['native_scores'])),
            'actual_bias_equals_intended': bool(np.array_equal(raw['delta'][:, 0], intended)),
            'captured_native_content_inverse_alignment': all(item['original_order_qkv_recovered_bitwise'] for item in alignment)}
        save_gates()
        if not all(gates[route].values()):
            raise AssertionError(f'{route} replay or raw content alignment failed')
        manifest['route_info'][route] = info
        record(route, values, raw)
        old.engine.atomic_json(output / 'manifest.json', manifest)
    np.savez_compressed(output / 'score_bias_by_route.npz', **deltas)
    pd.concat(diagnostics, ignore_index=True).to_csv(output / 'row_diagnostics.csv', index=False)
    gates['native_joint_permutation_numerical_floor'] = native_numerical_check(old, case, plan, samples[0])
    save_gates()
    old.verify_watch(plan['frozen_files_sha256'])
    manifest['elapsed_seconds'] = time.perf_counter() - started
    old.engine.atomic_json(output / 'manifest.json', manifest)
    hashes = {str(path.relative_to(output)): old.engine.sha256(path) for path in output.rglob('*') if path.is_file()}
    old.engine.atomic_json(output / 'complete.json', {'completed': True, 'all_gates_passed': True, 'no_training': True,
        'protocol_revision': plan['protocol_revision'], 'rows': len(meta),
        'primary_eligible_rows': int(meta.history_permutation_eligible.sum()), 'files_sha256': hashes})
    print(json.dumps({'completed': True, 'output': str(output), 'all_gates_passed': True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.prepare:
        prepare(args.case)
    elif args.output_dir is not None:
        run(args.case, args.output_dir)
    else:
        parser.error('--output-dir required')


if __name__ == '__main__':
    main()
