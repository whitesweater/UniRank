"""First-window engineering fidelity only; uses the old audited dot Observer.

No history permutation, new loss analysis, training, SISA hook, or pooling hook is
executed here. Original tensor capture and replay must first be bitwise faithful.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_permutation import ROOT, CASES, prior_module, plan_path

OWN_FILES = ('fidelity_v2.py', 'submit_fidelity_v2.sbatch')


def fidelity_plan(case):
    return HERE / f'plan_fidelity_v2_{case}.json'


def prepare(case):
    old = prior_module(case)
    prior = json.loads(plan_path(case).read_text())
    old.verify_watch(prior['frozen_files_sha256'])
    watched = dict(prior['frozen_files_sha256'])
    watched[str(plan_path(case))] = old.engine.sha256(plan_path(case))
    for name in OWN_FILES:
        watched[str(HERE / name)] = old.engine.sha256(HERE / name)
    plan = dict(prior)
    plan.update(fidelity_only=True, full_history_study=False, source_protocol_failed_identity=True,
                scope='First original 16384-row window only; original/capture/replay/zero fidelity; no reverse/random treatment evaluated',
                repair='Remove all new SISA and pooling hooks; use only previous audited run_probe.run_route and Observer',
                primary_endpoint='Bitwise identity and off fidelity, not prediction quality',
                frozen_files_sha256=watched, routes=['plain', 'identity_hook', 'replay_identity', 'replay_zero'])
    path = fidelity_plan(case)
    if path.exists():
        raise FileExistsError(path)
    old.engine.atomic_json(path, plan)
    print(json.dumps({'case': case, 'plan_sha256': old.engine.sha256(path), 'fidelity_only': True}), flush=True)


def run(case, output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Requires one allocated H100 GPU')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('One inference process only')
    allowed = ROOT / 'artifacts/rise_history_permutation_20260925'
    if output.exists() or not output.resolve().is_relative_to(allowed):
        raise ValueError('Use a new isolated fidelity result directory')
    output.mkdir(parents=True)
    old = prior_module(case)
    plan = json.loads(fidelity_plan(case).read_text())
    old.verify_watch(plan['frozen_files_sha256'])
    old.engine.DATASET, old.engine.LABELS = plan['dataset'], tuple(plan['labels'])
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('high')
    samples, meta = old.load_samples()
    samples = samples[:1]
    meta = meta.iloc[:16384].copy()
    prior_predictions = pd.read_csv(Path(plan['prior_probe_results']) / 'predictions.csv',
                                    float_precision='round_trip', low_memory=False).iloc[:16384]
    meta.to_csv(output / 'meta.csv', index=False)
    shutil.copyfile(fidelity_plan(case), output / 'plan.json')
    (output / 'frozen_sources').mkdir()
    for name in OWN_FILES:
        shutil.copyfile(HERE / name, output / 'frozen_sources' / name)
    gates = {}
    def save():
        old.engine.atomic_json(output / 'gates.json', gates)
    reference = {label: prior_predictions[label + '_full'].to_numpy(dtype=np.float32) for label in old.LABELS}
    plain, _, _ = old.run_route(plan, samples, 'plain', None, output)
    gates['plain_vs_prior_full'] = old.equal_predictions(plain, reference)
    save()
    if not old.passed(gates['plain_vs_prior_full']):
        raise AssertionError('Unmodified source-window drift')
    capture, capture_raw, capture_info = old.run_route(plan, samples, 'identity_hook', None, output)
    gates['capture_vs_plain'] = old.equal_predictions(capture, plain)
    save()
    if not old.passed(gates['capture_vs_plain']):
        raise AssertionError('Previously audited Observer is no longer faithful')
    replacement = capture_raw['delta'][:, 0]
    replay, replay_raw, replay_info = old.run_route(plan, samples, 'replay_identity', replacement, output)
    gates['replay_vs_plain'] = old.equal_predictions(replay, plain)
    gates['replay_bias_exact'] = bool(np.array_equal(replay_raw['delta'], capture_raw['delta']))
    gates['replay_qkv_mask_scale_exact'] = replay_info == capture_info
    save()
    if not old.passed(gates['replay_vs_plain']) or not gates['replay_bias_exact'] or not gates['replay_qkv_mask_scale_exact']:
        raise AssertionError('Captured BF16 correction cannot be replayed faithfully')
    zero, zero_raw, zero_info = old.run_route(plan, samples, 'replay_zero', np.zeros_like(replacement), output)
    zero_reference = {label: prior_predictions[label + '_local_off'].to_numpy(dtype=np.float32) for label in old.LABELS}
    gates['zero_vs_prior_local_off'] = old.equal_predictions(zero, zero_reference)
    gates['zero_qkv_mask_scale_exact'] = zero_info == capture_info
    gates['zero_bias_exact'] = bool(np.count_nonzero(zero_raw['delta']) == 0)
    save()
    if not old.passed(gates['zero_vs_prior_local_off']) or not gates['zero_qkv_mask_scale_exact'] or not gates['zero_bias_exact']:
        raise AssertionError('Zero replay is not faithful to earlier local-off')
    arrays = {f'{label}_{route}': values[label] for route, values in
              [('plain', plain), ('capture', capture), ('replay', replay), ('off', zero)] for label in old.LABELS}
    np.savez_compressed(output / 'fidelity_predictions.npz', **arrays)
    old.engine.atomic_json(output / 'manifest.json', {'case': case, 'device': old.semantic.device_metadata(),
        'fidelity_only': True, 'no_permutation_outcome_inspected': True, 'no_training': True,
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'plan_sha256': old.engine.sha256(fidelity_plan(case)),
        'capture_info': capture_info, 'replay_info': replay_info, 'zero_info': zero_info})
    old.verify_watch(plan['frozen_files_sha256'])
    hashes = {str(path.relative_to(output)): old.engine.sha256(path) for path in output.rglob('*') if path.is_file()}
    old.engine.atomic_json(output / 'complete.json', {'completed': True, 'all_gates_passed': True,
        'fidelity_only': True, 'no_training': True, 'rows': 16384, 'files_sha256': hashes})
    print(json.dumps({'completed': True, 'all_gates_passed': True, 'fidelity_only': True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.prepare:
        prepare(args.case)
    elif args.output_dir is not None:
        run(args.case, args.output_dir)
    else:
        parser.error('--output-dir required')


if __name__ == '__main__':
    main()
