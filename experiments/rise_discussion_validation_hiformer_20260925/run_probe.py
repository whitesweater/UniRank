"""Independent HiFormer full-site replication on frozen original windows."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'artifacts/rise_attention_redistribution/tencent_646303'
sys.path.insert(0, str(ROOT / 'scripts'))
import export_rise_semantic_attention as semantic
import export_rise_attention_redistribution as historical_diagnostic
from probe_math import DONOR_SEEDS, eligibility, crosspart_templates, crosspart_donor_indices, match_moments, moments, row_diagnostics

engine = semantic.engine
LABELS = ('is_click', 'is_conversion')
DATASET = 'TencentGR_10M_Action'
ROUTES = ['plain', 'identity_hook', 'local_off', 'template_raw', 'template_rms'] + [f'donor_rms_{seed}' for seed in DONOR_SEEDS]
OWN_FILES = ['run_probe.py', 'probe_math.py', 'test_probe_math.py', 'submit_probe.sbatch', 'README.md']


def digest_array(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def prediction_hashes(predictions):
    return {label: digest_array(predictions[label]) for label in LABELS}


def equal_predictions(actual, expected):
    return {label: {'equal': bool(np.array_equal(actual[label], expected[label])),
                    'max_abs_error': float(np.max(np.abs(actual[label] - expected[label])))} for label in LABELS}


def passed(result):
    return all(record['equal'] for record in result.values())


def load_samples():
    cached = torch.load(SOURCE / 'sample_batches.pt', map_location='cpu', weights_only=True, mmap=True)
    samples, frames = [], []
    for window, item in enumerate(cached):
        batch = item['batch']
        mask = batch[2].numpy()
        part_values = torch.unique(item['meta']['part_id']).tolist()
        if mask.shape != (16384, 100) or len(part_values) != 1 or not np.isin(mask,[0,1]).all():
            raise ValueError('Frozen full-window layout or masks changed')
        part = int(part_values[0])
        data = {'row_index': np.arange(window * 16384, (window + 1) * 16384),
                'part_id': np.full(16384, part), 'row_offset': np.arange(16384),
                'user_id': batch[0]['user_id'].numpy().astype(np.int64),
                'item_id': batch[1]['item_id'][:, -1].numpy().astype(np.int64),
                'seq_len': batch[0]['seq_len'].numpy().astype(np.int64),
                'effective_length': mask.sum(1).astype(np.int64)}
        for label in LABELS:
            truth = batch[0][label].numpy()
            if not np.isin(truth, [0, 1]).all():
                raise ValueError('Nonbinary cached labels')
            data[label] = truth.astype(np.uint8)
        selected = item['selected_indices'].numpy()
        for key, value in item['meta'].items():
            if key in data and not np.array_equal(data[key][selected], value.numpy()):
                raise ValueError(f'Original selected512 metadata drift: {key}')
        samples.append({'batch': batch, 'selected_indices': selected, 'part_id': part})
        frames.append(pd.DataFrame(data))
    if len(samples) != 8 or len(set(s['part_id'] for s in samples)) != 8:
        raise ValueError('Expected exactly eight distinct physical parts')
    meta = pd.concat(frames, ignore_index=True)
    masks = np.concatenate([sample['batch'][2].numpy().astype(bool) for sample in samples])
    eligible,peers = eligibility(meta.part_id.to_numpy(),masks)
    meta['eligible'] = eligible
    meta['crosspart_same_mask_peers'] = peers
    meta['exclusion_reason'] = np.where(eligible,'',np.where(meta.effective_length==0,'empty_native_history',
        np.where(meta.effective_length<2,'fewer_than_two_allowed_keys','no_crosspart_same_mask_peer')))
    return samples, meta


def source_watch(prior):
    complete = json.loads((SOURCE / 'complete.json').read_text())
    if not complete.get('completed'):
        raise ValueError('Incomplete historical source')
    watched = {str(SOURCE / name): digest for name, digest in complete['files_sha256'].items()}
    watched.update({str(ROOT / name): digest for name, digest in prior['source_sha256'].items()})
    watched[str(SOURCE / 'complete.json')] = engine.sha256(SOURCE / 'complete.json')
    watched[prior['checkpoint']['path']] = prior['checkpoint_sha256']
    watched[prior['checkpoint_log']['path']] = prior['checkpoint_log_sha256']
    watched[prior['metadata_file']['path']] = prior['metadata_sha256']
    watched[str(Path(semantic.__file__).resolve())] = engine.sha256(Path(semantic.__file__).resolve())
    for filename in OWN_FILES:
        watched[str(HERE / filename)] = engine.sha256(HERE / filename)
    return watched


def verify_watch(watched):
    for name, expected in watched.items():
        if engine.sha256(Path(name)) != expected:
            raise ValueError(f'Frozen source/input hash mismatch: {name}')


def prepare():
    if (HERE / 'plan.json').exists():
        raise FileExistsError('Plan already frozen; preserve it and create a versioned experiment if needed')
    prior = json.loads((SOURCE / 'manifest.json').read_text())
    watched = source_watch(prior)
    verify_watch(watched)
    samples, meta = load_samples()
    counts = meta[meta.eligible].groupby('user_id').size()
    plan = {'schema_version': 1, 'model': 'HiFormer', 'dataset': DATASET,
        'source_dir': str(SOURCE), 'checkpoint': prior['checkpoint'],
        'checkpoint_sha256': prior['checkpoint_sha256'], 'training_params': prior['training_params'],
        'frozen_files_sha256': watched, 'meta_sha256': hashlib.sha256(meta.to_csv(index=False).encode()).hexdigest(),
        'rows': len(meta), 'eligible_rows': int(meta.eligible.sum()), 'labels': list(LABELS),
        'windows': 8, 'native_batch_size': 16384, 'keys': 100,
        'coverage': {'all_rows_eligible': False, 'native_mask_groups':100,
                     'template_excludes_recipient_physical_part': True,
                     'peer_count_quantiles': np.quantile(meta.crosspart_same_mask_peers,[0,.25,.5,.75,1]).tolist(),
                     'excluded_rows':meta.loc[~meta.eligible,['row_index','part_id','row_offset','user_id','item_id','effective_length','exclusion_reason']].to_dict('records')},
        'sampling': 'All rows of the eight historical first-retained16384-row windows; no outcome selection; no population representativeness claim',
        'unique_users': int(meta.user_id.nunique()), 'eligible_unique_users': len(counts), 'eligible_users':len(counts), 'user_cluster_size': counts.describe().to_dict(),
        'positive_counts': {label: int(meta[label].sum()) for label in LABELS},
        'eligible_positive_counts': {label:int(meta.loc[meta.eligible,label].sum()) for label in LABELS},
        'replication_selection': 'Chosen from preflight coverage (6563users) before inspecting primary RankMixer results; historical full-site checkpoint seed20262028, not the current history-only variant',
        'routes': ROUTES, 'donor_seeds': list(DONOR_SEEDS),
        'route_definitions': {
            'plain': 'Unmodified frozen checkpoint inference',
            'identity_hook': 'Score-bias tensor replaced by the bitwise identical native BF16 tensor; record full-window QKV hashes, attention and predictions',
            'local_off': 'Zero score-bias tensor only at target attention, preserving upstream QKV, mask, scale and all other sites',
            'template_raw': 'Mean native Delta from same-mask rows in seven other physical parts; center over allowedkeys; retain its cohort centeredRMS and add recipient original allowedkey rowmean; masked coordinates and ineligible rows remain full',
            'template_rms': 'Same centered cross-part template, rescaled to recipient original centeredRMS and shifted to recipient original rowmean',
            'donor_rms': 'Three fixedseed choices with replacement from same-mask rows outside recipient physicalpart; donor allowedkey centeredDelta rescaled to recipient centeredRMS and shifted to recipient rowmean; masked coordinates and ineligible rows remain full'},
        'gauge_and_cast': 'All nonzero controls share recipient original rowmean before single BF16 cast; log actual postcast RMS/mean mismatch',
        'primary_endpoint': 'Interaction-weighted mean click BCE(full)-BCE(template_rms), commoneligible131070rows',
        'secondary_endpoints': ['Conversion BCE explicitly secondary',
            'Fixed equal-two-label macro BCE', 'User-balanced click sensitivity',
            'Signed contrasts off-minus-raw, raw-minus-rms, rms-minus-full, full-minus-off',
            'Donor3 mean per-row loss then usercluster bootstrap; individualseedrange separately',
            'CenteredDelta cosine and full-vs-counterfactual attentionTV, entropy, topkeychange, actualRMS'],
        'uncertainty': '2000 paired user-cluster bootstrap resamples with seed20260925, interaction-weighted ratio of cluster loss sums to cluster rowcounts; userbalanced sensitivity averages clustermeans',
        'relative_fractions': 'Only descriptive if off-minus-full positive and materially nonzero; absolute signed contrasts are primary',
        'gates': ['Source/checkpoint/input SHA256 beforeandafter', 'plain512 predictions equal historical512',
            'identity all131072 predictions bitwiseequal freshplain; source512raw equal',
            'allcounterfactual all131072QKV hashes and native masks/scales exactlyequal identity',
            'Actual BF16 bias equals intended onecast bias', 'source rows/masks fixed',
            'No historical allwindow or localoff equivalence gate is claimed because source did not retain those outputs'],
        'non_claims': ['No retraining comparison', 'No isolated AM/PA attribution', 'No full-test benchmark estimate', 'No causal semantic relevance claim'],
        'git': {'branch': subprocess.check_output(['git','branch','--show-current'], text=True).strip(),
                'commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                'status': subprocess.check_output(['git','status','--short'], text=True)}}
    meta.to_csv(HERE / 'frozen_meta.csv', index=False)
    plan['frozen_meta_csv_sha256'] = engine.sha256(HERE / 'frozen_meta.csv')
    engine.atomic_json(HERE / 'plan.json', plan)
    if len(meta)!=131072 or int(meta.eligible.sum())!=131070 or meta.user_id.nunique()!=6563:
        raise ValueError('Preflight eligibility/user count drift')
    print(json.dumps({k: plan[k] for k in ['rows','eligible_rows','unique_users','eligible_unique_users','positive_counts','eligible_positive_counts','coverage']}, indent=2))


def tensor_digest(tensor):
    # No full Q/K/V array is retained on CPU or saved. Hash fixed row chunks.
    h = hashlib.sha256()
    for start in range(0, len(tensor), 1024):
        chunk = tensor[start:start + 1024].detach().contiguous().view(torch.uint8).cpu().numpy()
        h.update(chunk.tobytes())
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'sha256': h.hexdigest()}


class Observer:
    def __init__(self, model, route):
        self.route, self.replacement, self.entries = route, None, []
        self.handle = semantic.site_module(model).register_forward_pre_hook(self.hook, with_kwargs=True)

    def hook(self, module, args, kwargs):
        bias = kwargs.get('score_bias')
        if len(args) != 3 or bias is None or tuple(bias.shape) != (16384, 1, 1, 100):
            raise ValueError('Unexpected target-site structure')
        modified = bias if self.route == 'identity_hook' else self.replacement.to(bias.dtype)
        if modified is None or modified.shape != bias.shape:
            raise ValueError('Missing replacement')
        changed = dict(kwargs, score_bias=modified)
        self.entries.append((args, changed))
        return args, changed

    def consume(self, selected):
        if len(self.entries) != 1:
            raise ValueError(f'Expected one capture, got {len(self.entries)}')
        args, kwargs = self.entries.pop()
        qkv = {name: tensor_digest(tensor) for name, tensor in zip(('query','key','value'), args)}
        indices = torch.as_tensor(selected, device=args[0].device)
        qkv_selected = {name: tensor_digest(tensor.index_select(0, indices)) for name, tensor in zip(('query','key','value'), args)}
        mask = kwargs['mask']
        site = {'scale': kwargs.get('scale'), 'mask': tensor_digest(mask),
                'bias_dtype': str(kwargs['score_bias'].dtype), 'mask_dtype': str(mask.dtype)}
        raw = semantic.reconstruct_attention(args[0], args[1], kwargs['score_bias'], mask, kwargs.get('scale'))
        raw['native_mask'] = mask.squeeze(-2).detach().bool().cpu().numpy()
        return raw, {'qkv': qkv, 'selected_qkv': qkv_selected, 'site': site}

    def close(self):
        self.handle.remove()
        self.entries.clear()


def run_route(plan, samples, route, replacement, output):
    engine.seed_everything(plan['training_params']['seed'])
    model = historical_diagnostic.wrapper.load_model(plan['training_params'], Path(plan['checkpoint']['path']), 0)
    observer = None if route == 'plain' else Observer(model, route)
    if observer:
        if route != 'identity_hook':
            observer.replacement = torch.zeros((16384,1,1,100), dtype=torch.float32, device='cuda')
        torch._dynamo.reset()
    predictions = {label: [] for label in LABELS}
    arrays, info = [], []
    try:
        for window, sample in enumerate(samples):
            if observer and route != 'identity_hook':
                target = replacement[window * 16384:(window + 1) * 16384]
                observer.replacement.copy_(torch.as_tensor(target, dtype=torch.float32, device='cuda').reshape(16384,1,1,100))
            with torch.no_grad(), model._get_amp_context():
                values = model.forward(semantic.tree_map(sample['batch'], lambda t: t.clone()))
            for label in LABELS:
                values_np = values[f'{label}_pred'].detach().float().cpu().numpy().reshape(-1)
                if len(values_np) != 16384 or not np.isfinite(values_np).all():
                    raise ValueError('Invalid full-window predictions')
                predictions[label].append(values_np)
            if observer:
                raw, receipt = observer.consume(sample['selected_indices'])
                arrays.append(raw); info.append(receipt)
            logging.info('%s: %s/8 original windows', route, window + 1)
            engine.atomic_json(output / 'progress.json', {'route': route, 'windows_complete': window + 1})
    finally:
        if observer: observer.close()
        del model
        gc.collect(); torch.cuda.empty_cache()
    return {label: np.concatenate(v) for label,v in predictions.items()}, {
        key: np.concatenate([a[key] for a in arrays]) for key in arrays[0]} if arrays else None, info


def run(output):
    if not torch.cuda.is_available() or int(os.environ.get('WORLD_SIZE', 1)) != 1:
        raise RuntimeError('Requires one allocated GPU and one process')
    if not output.resolve().is_relative_to(HERE) or output.exists():
        raise ValueError('Use a new result directory inside this experiment')
    output.mkdir(parents=True)
    torch.cuda.set_device(0); torch.set_float32_matmul_precision('high')
    plan = json.loads((HERE / 'plan.json').read_text())
    verify_watch(plan['frozen_files_sha256'])
    if engine.sha256(HERE / 'frozen_meta.csv') != plan['frozen_meta_csv_sha256']:
        raise ValueError('Frozen metadata hash drift')
    prior = json.loads((SOURCE / 'manifest.json').read_text())
    for name, identity in prior['data_files'].items():
        if engine.file_identity(Path(name)) != identity:
            raise ValueError(f'Original data identity changed: {name}')
    historical_diagnostic.wrapper.configure_engine(DATASET)
    engine.DATASET, engine.LABELS = DATASET, LABELS
    samples, meta = load_samples()
    if hashlib.sha256(meta.to_csv(index=False).encode()).hexdigest() != plan['meta_sha256']:
        raise ValueError('Cached metadata drift')
    selected = np.concatenate([s['selected_indices'] + i * 16384 for i,s in enumerate(samples)])
    historical = pd.read_csv(SOURCE / 'prediction_check.csv', float_precision='round_trip')
    raw_source = np.load(SOURCE / 'raw_attention.npz')
    full_source_raw = {key: raw_source[key] for key in raw_source.files}
    manifest = {'plan_sha256': engine.sha256(HERE / 'plan.json'), 'device': semantic.device_metadata(),
                'started_git_status': subprocess.check_output(['git','status','--short'],text=True), 'route_info': {}}
    shutil.copyfile(HERE / 'plan.json', output / 'plan.json')
    shutil.copyfile(HERE / 'frozen_meta.csv', output / 'meta.csv')
    for filename in OWN_FILES:
        (output / 'frozen_sources').mkdir(exist_ok=True)
        shutil.copyfile(HERE / filename, output / 'frozen_sources' / filename)
    gates = {}
    plain, _, _ = run_route(plan, samples, 'plain', None, output)
    reference = {label: historical[f'{label}_plain'].to_numpy(dtype=np.float32) for label in LABELS}
    gates['plain_vs_source512'] = equal_predictions({label:v[selected] for label,v in plain.items()}, reference)
    manifest['fresh_plain_allwindow_prediction_sha256'] = prediction_hashes(plain)
    if not passed(gates['plain_vs_source512']):
        engine.atomic_json(output / 'gates.json', gates)
        raise RuntimeError('Plain predictions drifted from historical source')
    full, full_raw, full_info = run_route(plan, samples, 'identity_hook', None, output)
    gates['identity_vs_plain'] = equal_predictions(full, plain)
    gates['source512_raw_equal'] = all(np.array_equal(full_raw[key][selected], full_source_raw[key], equal_nan=True) for key in full_raw if key!='native_mask')
    gates['source512_mask_equal'] = np.array_equal(full_raw['native_mask'][selected,0],full_source_raw['valid_mask'])
    if not passed(gates['identity_vs_plain']) or not gates['source512_raw_equal'] or not gates['source512_mask_equal']:
        engine.atomic_json(output / 'gates.json', gates)
        raise RuntimeError('Identity hook failed historical or full-window equivalence')
    manifest['route_info']['identity_hook'] = full_info
    np.savez_compressed(output / 'attention_full.npz', **full_raw)
    preds = meta.copy()
    for label in LABELS: preds[f'{label}_full'] = full[label]
    preds.to_csv(output / 'predictions.csv', index=False)
    delta, weights = full_raw['delta'][:,0], full_raw['weight_after'][:,0]
    mask = full_raw['native_mask'][:,0]
    eligible = meta.eligible.to_numpy(dtype=bool)
    raw, rms, template_receipts = crosspart_templates(delta, meta.part_id.to_numpy(), mask)
    replacements = {'local_off': np.zeros_like(delta), 'template_raw': raw.astype(np.float32), 'template_rms': rms.astype(np.float32)}
    donor_maps = {}
    for seed in DONOR_SEEDS:
        mapping = crosspart_donor_indices(meta.part_id.to_numpy(), mask, seed)
        donor_maps[str(seed)] = mapping
        replacements[f'donor_rms_{seed}'] = match_moments(delta[mapping], delta, mask, eligible).astype(np.float32)
    np.savez_compressed(output / 'donor_maps.npz', **donor_maps)
    engine.atomic_json(output / 'template_receipts.json', template_receipts)
    fidelity, diagnostics = [], []
    for route in ROUTES[2:]:
        replacement = replacements[route]
        route_pred, route_raw, route_info = run_route(plan, samples, route, replacement, output)
        same_qkv = all(info['qkv'] == reference_info['qkv'] and info['site'] == reference_info['site'] for info,reference_info in zip(route_info,full_info))
        gates[route] = {'all131072_qkv_mask_scale_equal': same_qkv,
                        'native_scores_equal': bool(np.array_equal(route_raw['native_scores'], full_raw['native_scores']))}
        engine.atomic_json(output / 'gates.json', gates)
        if not same_qkv or not gates[route]['native_scores_equal']:
            raise RuntimeError(f'{route} changed upstream QKV/native mask/scale or failed historical localoff gate')
        for label in LABELS: preds[f'{label}_{route}'] = route_pred[label]
        preds.to_csv(output / 'predictions.csv', index=False)
        actual = route_raw['delta'][:,0]
        expected_cast = torch.as_tensor(replacement).to(torch.bfloat16).float().numpy()
        if not np.array_equal(actual, expected_cast):
            raise RuntimeError(f'{route}: actual native bias differs from intended one-cast BF16 replacement')
        gates[route]['actual_bias_equals_intended_bf16'] = True
        engine.atomic_json(output / 'gates.json', gates)
        if route!='local_off' and (not np.array_equal(actual[~mask],delta[~mask]) or not np.array_equal(actual[~eligible],delta[~eligible])):
            raise RuntimeError(f'{route}: masked coordinates or ineligible rows changed')
        before_mean, before_rms = moments(replacement[eligible],mask[eligible])
        after_mean, after_rms = moments(actual[eligible],mask[eligible])
        source_mean, source_rms = moments(delta[eligible],mask[eligible])
        fidelity.append({'route': route,
            'beforecast_mean_abs_error_to_full': float(np.abs(before_mean - source_mean).max()),
            'beforecast_rms_abs_error_to_full': float(np.abs(before_rms - source_rms).max()),
            'postcast_mean_abs_error_to_intended': float(np.abs(after_mean - before_mean).max()),
            'postcast_rms_abs_error_to_intended': float(np.abs(after_rms - before_rms).max()),
            'postcast_rms_relative_error_p99': float(np.quantile(np.abs(after_rms - before_rms) / np.maximum(before_rms,1e-12), .99)),
            'actual_bias_sha256': digest_array(actual), 'intended_fp32_bias_sha256': digest_array(replacement)})
        diag = pd.DataFrame(row_diagnostics(delta, actual, weights, route_raw['weight_after'][:,0],mask,eligible))
        diag.insert(0, 'row_index', meta.row_index); diag.insert(1,'route',route)
        diag.insert(2,'eligible',eligible)
        diagnostics.append(diag)
        pd.concat(diagnostics,ignore_index=True).to_csv(output / 'row_diagnostics.csv', index=False)
        np.savez_compressed(output / f'attention_{route}.npz', delta=route_raw['delta'], weight=route_raw['weight_after'])
        manifest['route_info'][route] = route_info
        engine.atomic_json(output / 'fidelity.json', fidelity)
        engine.atomic_json(output / 'manifest.json', manifest)
        del route_raw; gc.collect()
    verify_watch(plan['frozen_files_sha256'])
    manifest['finished_git_status'] = subprocess.check_output(['git','status','--short'],text=True)
    engine.atomic_json(output / 'manifest.json', manifest)
    engine.atomic_json(output / 'complete.json', {'completed':True, 'all_gates_passed':True, 'rows':len(meta), 'eligible_rows':int(eligible.sum()),
        'files_sha256': {str(path.relative_to(output)):engine.sha256(path) for path in sorted(output.rglob('*')) if path.is_file()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    if args.prepare: prepare()
    elif args.output_dir: run(args.output_dir)
    else: parser.error('Supply --prepare or --output-dir')


if __name__ == '__main__': main()
