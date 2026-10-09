"""No-training history-order probe reusing previously audited checkpoint windows."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import gc
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT)]
from permutation_math import (PERMUTATION_SEED, make_permutation, gather_history,
                              combined_mapping, align_bias, joint_raw_batch, eager_children, digest)

CASES = {
    'rankmixer_kuairand': ('rise_discussion_validation_20260925', 'results_651654'),
    'hiformer_taac25': ('rise_discussion_validation_hiformer_20260925', 'results_651691'),
}
ROUTES = ('identity', 'off', 'reverse', 'random')
OWN_FILES = ('run_permutation.py', 'permutation_math.py', 'test_permutation.py', 'submit_permutation.sbatch', 'README.md')


def prior_module(case):
    directory = ROOT / 'experiments' / CASES[case][0]
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('audited_prior_probe', directory / 'run_probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan_path(case):
    return HERE / f'plan_{case}.json'


def load_model(old, case, plan):
    old.engine.seed_everything(plan['training_params']['seed'])
    loader = old.semantic.load_model if case == 'rankmixer_kuairand' else old.historical_diagnostic.wrapper.load_model
    return loader(plan['training_params'], Path(plan['checkpoint']['path']), 0)


def pooling_module(model):
    from unirank.pytorch.layers.attentions.target_attention import MultiHeadTargetAttention
    matches = [module for name, module in model.named_modules()
               if '.'.join(part for part in name.split('.') if part != '_orig_mod') == 'attention_layers'
               and isinstance(module, MultiHeadTargetAttention)]
    if len(matches) != 1:
        raise AssertionError('Expected exactly one inspected native history pooling module')
    return matches[0]


def permutations(samples):
    return {route: [make_permutation(sample['batch'][2].numpy(), route, PERMUTATION_SEED + window)
                    for window, sample in enumerate(samples)] for route in ROUTES}


def prepare(case):
    old = prior_module(case)
    path = plan_path(case)
    if path.exists():
        raise FileExistsError(path)
    prior = json.loads((old.HERE / 'plan.json').read_text())
    old.verify_watch(prior['frozen_files_sha256'])
    if old.engine.sha256(old.HERE / 'frozen_meta.csv') != prior['frozen_meta_csv_sha256']:
        raise AssertionError('Prior frozen metadata changed')
    samples, meta = old.load_samples()
    meta['history_permutation_eligible'] = meta.effective_length >= 2
    maps = permutations(samples)
    watched = dict(prior['frozen_files_sha256'])
    for file in OWN_FILES:
        watched[str(HERE / file)] = old.engine.sha256(HERE / file)
    watched[str(old.HERE / 'plan.json')] = old.engine.sha256(old.HERE / 'plan.json')
    previous_results = old.HERE / CASES[case][1]
    for name in ('predictions.csv', 'meta.csv', 'complete.json', 'gates.json'):
        if not (previous_results / name).exists():
            raise FileNotFoundError(previous_results / name)
        watched[str(previous_results / name)] = old.engine.sha256(previous_results / name)
    plan = {
        'schema_version': 1, 'case': case, 'model': prior['model'], 'dataset': prior['dataset'],
        'checkpoint': prior['checkpoint'], 'checkpoint_sha256': prior['checkpoint_sha256'],
        'training_params': prior['training_params'], 'prior_probe_plan': str(old.HERE / 'plan.json'),
        'prior_probe_results': str(previous_results), 'historical_attention_source': str(old.SOURCE),
        'labels': list(old.LABELS), 'rows': len(meta), 'primary_eligible_rows': int(meta.history_permutation_eligible.sum()),
        'primary_click_label': 'is_click', 'routes': list(ROUTES),
        'primary_endpoint': 'Mean paired click BCE(permuted correction)-BCE(identity), length>=2; reverse and one fixed shuffle separately',
        'secondary_endpoints': ['all-row click BCE', 'all other labels', 'user-balanced click sensitivity',
                                'per-row centered correction change, attention TV, pooling and prediction differences'],
        'sampling': 'Exactly the same eight preselected historical 16384-row windows; no new sampling or outcome selection',
        'route_definition': 'Permute complete valid historical event embeddings entering target RISE, keep candidate last and padding fixed, compute native score module once, inverse-align key scores to original record identity; original Q/K/V/mask/scale unchanged',
        'identity_definition': 'Same pre/post hook and identity gather; must match unmodified checkpoint predictions and earlier captured bias bitwise',
        'off_definition': 'Same hook path, replace only inspected target-site score correction by zeros_like; all other RISE sites remain active',
        'native_order_wording': 'data-provided/native history order; chronology is not presumed',
        'no_matching': 'No rowmean, RMS, moment, or sign normalization of recomputed correction',
        'permutation_seed': PERMUTATION_SEED, 'random_seed_rule': 'default_rng(base_seed + window_index), valid-index permutation independently per row',
        'permutation_hashes': {route: [digest(value) for value in windows] for route, windows in maps.items()},
        'native_numerical_check': {'window': 0, 'rows': 64, 'fp32_math_pool_atol': 1e-5, 'fp32_math_pool_rtol': 1e-4,
                                   'raw_records': 'jointly permute every item_dict history attribute and mask/multi_masks; candidate/nonhistory fields/labels fixed',
                                   'bf16': 'measure eager BF16 automatic-SDPA precision residual; compiler path is not the production main-route path; no bitwise invariance assumption'},
        'uncertainty': 'Independent analysis: 2000 paired user-cluster bootstrap draws, seed20260925; this is one checkpoint, not training replication',
        'no_training': True, 'runtime_hardware': 'one NVIDIA H100 GPU; abort on other models',
        'frozen_files_sha256': watched,
        'meta_csv_sha256': __import__('hashlib').sha256(meta.to_csv(index=False).encode()).hexdigest(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
    }
    old.engine.atomic_json(path, plan)
    print(json.dumps({'case': case, 'plan': str(path), 'sha256': old.engine.sha256(path),
                      'rows': plan['rows'], 'primary_eligible_rows': plan['primary_eligible_rows']}), flush=True)


class OrderHook:
    """Apply P before native RISE and P^-1 on the output key axis, once."""
    def __init__(self, model, route):
        self.route = route
        self.mapping = None
        self.call_markers = []
        score = pooling_module(model).sisa_score_bias
        self.handles = [score.register_forward_pre_hook(self.before, with_kwargs=True),
                        score.register_forward_hook(self.after, with_kwargs=True)]

    def before(self, module, args, kwargs):
        hidden, valid, head_dim = args
        if kwargs.get('query_slice') != slice(-1, None) or kwargs.get('key_slice') != slice(0, -1):
            raise AssertionError('Target query/key slices changed')
        full = combined_mapping(self.mapping)
        if tuple(hidden.shape[:2]) != tuple(full.shape):
            raise AssertionError('Unexpected concatenated history/candidate layout')
        return (gather_history(hidden, full), gather_history(valid, full), head_dim), kwargs

    def after(self, module, args, kwargs, output):
        self.call_markers.append(True)
        aligned = align_bias(output, self.mapping)
        if aligned.dtype != output.dtype or aligned.shape != output.shape:
            raise AssertionError('Bias shape or dtype changed')
        return torch.zeros_like(aligned) if self.route == 'off' else aligned

    def close(self):
        for handle in self.handles:
            handle.remove()


def run_route(old, case, plan, samples, route, maps, output):
    model = load_model(old, case, plan)
    observer = None if route == 'plain' else old.Observer(model, 'identity_hook')
    order = None if route == 'plain' else OrderHook(model, route)
    pools = []
    pool_entries = []
    order_calls = 0
    pool_handle = None if route == 'plain' else pooling_module(model).register_forward_hook(
        lambda module, args, values: pool_entries.append(values))
    if observer:
        torch._dynamo.reset()
    predictions = {label: [] for label in old.LABELS}
    raw_arrays, receipts = [], []
    try:
        for window, sample in enumerate(samples):
            if order:
                order.mapping = torch.as_tensor(maps[route][window], dtype=torch.long, device='cuda')
            with torch.no_grad(), model._get_amp_context():
                values = model(old.semantic.tree_map(sample['batch'], lambda value: value.clone()))
            for label in old.LABELS:
                prediction = values[label + '_pred'].detach().float().cpu().numpy().reshape(-1)
                if len(prediction) != 16384 or not np.isfinite(prediction).all():
                    raise AssertionError('Invalid frozen-window prediction')
                predictions[label].append(prediction)
            if observer:
                arrays, receipt = observer.consume(sample['selected_indices'])
                raw_arrays.append(arrays)
                receipts.append(receipt)
                if len(order.call_markers) != 1 or len(pool_entries) != 1:
                    raise AssertionError('Expected exactly one score and pooling call per window')
                order.call_markers.pop()
                order_calls += 1
                pools.append(pool_entries.pop().detach().float().cpu().numpy())
            old.engine.atomic_json(output / 'progress.json', {'route': route, 'windows_completed': window + 1})
            logging.info('%s window %s/%s', route, window + 1, len(samples))
        if order and order_calls != len(samples):
            raise AssertionError('Inspected score module executed an unexpected number of times')
    finally:
        if observer:
            observer.close()
        if order:
            order.close()
        if pool_handle:
            pool_handle.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return ({label: np.concatenate(values) for label, values in predictions.items()},
            {key: np.concatenate([item[key] for item in raw_arrays]) for key in raw_arrays[0]} if raw_arrays else None,
            receipts, np.concatenate(pools) if pools else None)


def difference(actual, reference):
    delta = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    return {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()), 'bitwise_equal': bool(np.array_equal(actual, reference))}


def native_numerical_check(old, case, plan, sample):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    model = load_model(old, case, plan)
    small = old.semantic.tree_map(sample['batch'], lambda value: value[:plan['native_numerical_check']['rows']].clone())
    modes = {mode: torch.as_tensor(make_permutation(small[2].numpy(), mode, PERMUTATION_SEED), dtype=torch.long)
             for mode in ('identity', 'reverse', 'random')}
    records = {}
    captured = []
    with eager_children(model):
        pool_handle = pooling_module(model).register_forward_hook(
            lambda module, args, output: captured.append(output.detach().float().cpu().numpy()))
        off_handle = old.semantic.site_module(model).register_forward_pre_hook(
            lambda module, args, kwargs: (args, dict(kwargs, score_bias=torch.zeros_like(kwargs['score_bias']))),
            with_kwargs=True)
        try:
            for precision in ('fp32_math', 'bf16_eager_auto_sdpa'):
                torch.set_float32_matmul_precision('highest' if precision == 'fp32_math' else 'high')
                reference = None
                for mode, mapping in modes.items():
                    changed = joint_raw_batch(small, mapping)
                    with torch.no_grad(), (sdpa_kernel(SDPBackend.MATH) if precision == 'fp32_math' else nullcontext()), \
                            torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16_eager_auto_sdpa'):
                        values = model(changed)
                    if len(captured) != 1:
                        raise AssertionError('Expected one native pooling capture')
                    pooled = captured.pop()
                    prediction = {label: values[label + '_pred'].detach().float().cpu().numpy() for label in old.LABELS}
                    if not np.isfinite(pooled).all() or not all(np.isfinite(value).all() for value in prediction.values()):
                        raise AssertionError('Non-finite numerical-check pooling or prediction')
                    if reference is None:
                        reference = (pooled, prediction)
                    if precision == 'fp32_math':
                        np.testing.assert_allclose(pooled, reference[0], atol=plan['native_numerical_check']['fp32_math_pool_atol'],
                                                   rtol=plan['native_numerical_check']['fp32_math_pool_rtol'])
                    records[f'{precision}_{mode}'] = {
                        'pooling': difference(pooled, reference[0]),
                        'prediction': {label: difference(prediction[label], reference[1][label]) for label in old.LABELS}}
        finally:
            off_handle.remove()
            pool_handle.remove()
    torch.set_float32_matmul_precision('high')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {'passed_fp32_pooling_gate': True, 'rows': len(small[2]), 'checks': records}


def run(case, output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Requires one Slurm GPU process; never training')
    if 'H100' not in torch.cuda.get_device_name(0):
        raise RuntimeError(f'Frozen inference protocol requires H100; got {torch.cuda.get_device_name(0)}')
    allowed = ROOT / 'artifacts/rise_history_permutation_20260925'
    if not output.resolve().is_relative_to(allowed) or output.exists():
        raise ValueError('Use a new output directory under the isolated runtime artifacts root')
    output.mkdir(parents=True)
    old = prior_module(case)
    plan = json.loads(plan_path(case).read_text())
    old.verify_watch(plan['frozen_files_sha256'])
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('high')
    old.engine.DATASET, old.engine.LABELS = plan['dataset'], tuple(plan['labels'])
    samples, meta = old.load_samples()
    meta['history_permutation_eligible'] = meta.effective_length >= 2
    if __import__('hashlib').sha256(meta.to_csv(index=False).encode()).hexdigest() != plan['meta_csv_sha256']:
        raise AssertionError('Preselected row metadata changed')
    maps = permutations(samples)
    if {route: [digest(value) for value in windows] for route, windows in maps.items()} != plan['permutation_hashes']:
        raise AssertionError('Prespecified permutation changed')
    meta.to_csv(output / 'meta.csv', index=False)
    shutil.copyfile(plan_path(case), output / 'plan.json')
    (output / 'frozen_sources').mkdir()
    for filename in OWN_FILES:
        shutil.copyfile(HERE / filename, output / 'frozen_sources' / filename)
    np.savez_compressed(output / 'permutations.npz', **{route: np.concatenate(values) for route, values in maps.items() if route in ('reverse', 'random')})
    started = time.perf_counter()
    gates = {}
    manifest = {'case': case, 'plan_sha256': old.engine.sha256(plan_path(case)),
                'device': old.semantic.device_metadata(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                'no_training': True, 'route_info': {}}
    plain, _, _, _ = run_route(old, case, plan, samples, 'plain', maps, output)
    selected = np.concatenate([sample['selected_indices'] + 16384 * window for window, sample in enumerate(samples)])
    historical_name = 'predictions.csv' if case == 'rankmixer_kuairand' else 'prediction_check.csv'
    historical = pd.read_csv(old.SOURCE / historical_name, float_precision='round_trip')
    gates['plain_vs_historical_attention512'] = old.equal_predictions(
        {label: values[selected] for label, values in plain.items()},
        {label: historical[label + '_plain'].to_numpy(dtype=np.float32) for label in old.LABELS})
    if not old.passed(gates['plain_vs_historical_attention512']):
        old.engine.atomic_json(output / 'gates.json', gates)
        raise AssertionError('Unmodified checkpoint differs from historical capture')
    previous = pd.read_csv(Path(plan['prior_probe_results']) / 'predictions.csv', float_precision='round_trip')
    if not np.array_equal(meta.row_index.to_numpy(), previous.row_index.to_numpy()):
        raise AssertionError('Prior probe row alignment changed')
    preds = meta.copy()
    reference_raw = reference_info = reference_pool = None
    diagnostics = []
    deltas = {}
    for route in ROUTES:
        values, raw, info, pooled = run_route(old, case, plan, samples, route, maps, output)
        if route == 'identity':
            gates['identity_vs_plain_allrows'] = old.equal_predictions(values, plain)
            gates['identity_vs_prior_probe_full_allrows'] = old.equal_predictions(values,
                {label: previous[label + '_full'].to_numpy(dtype=np.float32) for label in old.LABELS})
            source_raw = np.load(old.SOURCE / 'raw_attention.npz')
            gates['historical_attention512_raw_bitwise'] = all(
                np.array_equal(raw[key][selected], source_raw[key], equal_nan=True)
                for key in raw if key in source_raw.files and key != 'native_mask')
            if not old.passed(gates['identity_vs_plain_allrows']) or not old.passed(gates['identity_vs_prior_probe_full_allrows']) or not gates['historical_attention512_raw_bitwise']:
                old.engine.atomic_json(output / 'gates.json', gates)
                raise AssertionError('New identity hook failed bitwise fidelity')
            reference_raw, reference_info, reference_pool = raw, info, pooled
        else:
            same_inputs = all(a['qkv'] == b['qkv'] and a['site'] == b['site'] for a, b in zip(info, reference_info))
            same_scores = np.array_equal(raw['native_scores'], reference_raw['native_scores'])
            gates[route] = {'native_qkv_mask_scale_bitwise': same_inputs, 'native_scores_bitwise': same_scores}
            if not same_inputs or not same_scores:
                raise AssertionError('Order treatment changed original native attention path')
        if route == 'off':
            gates['off_vs_prior_probe_local_off_allrows'] = old.equal_predictions(values,
                {label: previous[label + '_local_off'].to_numpy(dtype=np.float32) for label in old.LABELS})
            if not old.passed(gates['off_vs_prior_probe_local_off_allrows']):
                raise AssertionError('Local off differs from earlier matched site intervention')
        for label in old.LABELS:
            preds[label + '_' + route] = values[label]
        preds.to_csv(output / 'predictions.csv', index=False)
        delta = raw['delta'][:, 0]
        deltas[route] = delta
        mask = raw['native_mask'][:, 0].astype(bool)
        count = np.maximum(mask.sum(1), 1)
        reference_delta = reference_raw['delta'][:, 0]
        centered = delta - (delta * mask).sum(1, keepdims=True) / count[:, None]
        centered_ref = reference_delta - (reference_delta * mask).sum(1, keepdims=True) / count[:, None]
        pool_difference = np.abs(pooled.astype(np.float64) - reference_pool)
        diagnostic = pd.DataFrame({'row_index': meta.row_index, 'route': route,
            'centered_delta_difference_rms': np.sqrt((((centered - centered_ref) ** 2) * mask).sum(1) / count),
            'attention_tv': .5 * np.abs(raw['weight_after'][:, 0] - reference_raw['weight_after'][:, 0]).sum(1),
            'pooling_max_abs_difference': pool_difference.max(1), 'pooling_mean_abs_difference': pool_difference.mean(1)})
        diagnostics.append(diagnostic)
        manifest['route_info'][route] = info
        old.engine.atomic_json(output / 'gates.json', gates)
        old.engine.atomic_json(output / 'manifest.json', manifest)
    np.savez_compressed(output / 'score_bias_by_route.npz', **deltas)
    pd.concat(diagnostics, ignore_index=True).to_csv(output / 'row_diagnostics.csv', index=False)
    gates['native_joint_permutation_numerical_floor'] = native_numerical_check(old, case, plan, samples[0])
    old.engine.atomic_json(output / 'gates.json', gates)
    old.verify_watch(plan['frozen_files_sha256'])
    manifest['elapsed_seconds'] = time.perf_counter() - started
    old.engine.atomic_json(output / 'manifest.json', manifest)
    files = {str(path.relative_to(output)): old.engine.sha256(path) for path in output.rglob('*') if path.is_file()}
    old.engine.atomic_json(output / 'complete.json', {'completed': True, 'all_gates_passed': True, 'no_training': True,
        'rows': len(meta), 'primary_eligible_rows': int(meta.history_permutation_eligible.sum()), 'files_sha256': files})
    print(json.dumps({'completed': True, 'output': str(output), 'no_training': True}), flush=True)


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
    else:
        if args.output_dir is None:
            parser.error('--output-dir required for no-training GPU inference')
        run(args.case, args.output_dir)


if __name__ == '__main__':
    main()
