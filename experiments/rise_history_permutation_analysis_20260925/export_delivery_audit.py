"""Read completed results, summarize every sign, and inventory portable evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiments/rise_history_permutation_20260925'
CASES = {'rankmixer_kuairand': 'v3_job_652367', 'hiformer_taac25': 'v3_job_652366'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def masked_weights(scores, mask):
    safe = np.where(mask, scores, 0.)
    maximum = np.max(np.where(mask, safe, -np.inf), axis=1, keepdims=True)
    maximum = np.where(mask.any(axis=1, keepdims=True), maximum, 0.)
    exponent = np.exp(np.where(mask, safe - maximum, -np.inf))
    denominator = exponent.sum(axis=1, keepdims=True)
    return np.divide(exponent, denominator, out=np.zeros_like(exponent), where=denominator > 0)


def main():
    out = HERE / 'completed_results_audit'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(exist_ok=True)
    inventory, inference, dependency, compact = {}, {}, {}, {}
    diagnostics, native_rows, diagnostic_notes = [], [], {}
    for case, job in CASES.items():
        source = ROOT / 'artifacts/rise_history_permutation_20260925' / case / job
        analysis = source / 'analysis'
        manifest, audit = read(analysis / 'analysis_manifest.json'), read(analysis / 'independent_data_audit.json')
        assert audit['status'] == 'RAW_DATA_VERIFIED' and not audit['errors']
        for name, expected in manifest['files_sha256'].items():
            assert sha(analysis / name) == expected, (case, name)
        for p in analysis.iterdir():
            if p.is_file():
                inventory[f'{case}/analysis/{p.name}'] = {'remote_origin': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}
        summary = read(analysis / 'summary.json')
        gates, plan = read(source / 'gates.json'), read(source / 'plan.json')
        metrics = pd.read_csv(analysis / 'metrics.csv')
        contrasts = pd.read_csv(analysis / 'contrasts.csv')
        changes = pd.read_csv(analysis / 'prediction_changes.csv')
        meta = pd.read_csv(source / 'meta.csv', low_memory=False)
        diag = pd.read_csv(source / 'row_diagnostics.csv')
        assert np.array_equal(meta.row_index, np.arange(131072))
        eligible = meta.effective_length.to_numpy() >= 2
        prior_attention = Path(plan['prior_probe_results']) / 'attention_full.npz'
        with np.load(prior_attention, allow_pickle=False) as original:
            native_scores = original['native_scores'][:, 0].astype(np.float64)
            native_mask = original['native_mask'][:, 0].astype(bool)
        sample_path = Path(plan['historical_attention_source']) / 'sample_batches.pt'
        assert sha(sample_path) == plan['frozen_files_sha256'][str(sample_path)]
        samples = torch.load(sample_path, map_location='cpu', weights_only=True, mmap=True)
        actual_mask = np.concatenate([v['batch'][2].numpy().astype(bool) for v in samples])
        assert np.array_equal(actual_mask, native_mask)
        assert np.array_equal(native_mask.sum(axis=1), meta.effective_length.to_numpy())
        del samples, actual_mask
        with np.load(source / 'score_bias_by_route.npz', allow_pickle=False) as score_file:
            scores = {route: score_file[route].astype(np.float64) for route in ('identity', 'off', 'reverse', 'random')}
        assert all(np.isfinite(np.where(native_mask, value, 0.)).all() for value in scores.values())
        count = np.maximum(native_mask.sum(axis=1), 1)
        original_centered = np.where(native_mask, scores['identity'] - np.where(native_mask, scores['identity'], 0.).sum(axis=1, keepdims=True) / count[:, None], 0.)
        original_weights = masked_weights(native_scores + scores['identity'], native_mask)
        bad_input_rows = []
        for route in ('identity', 'off', 'reverse', 'random'):
            route_diag = diag.loc[diag.route == route].sort_values('row_index')
            assert np.array_equal(route_diag.row_index, meta.row_index)
            centered = np.where(native_mask, scores[route] - np.where(native_mask, scores[route], 0.).sum(axis=1, keepdims=True) / count[:, None], 0.)
            recomputed = {'centered_delta_difference_rms': np.sqrt(((centered - original_centered) ** 2).sum(axis=1) / count),
                          'attention_tv': .5 * np.abs(masked_weights(native_scores + scores[route], native_mask) - original_weights).sum(axis=1)}
            for population, selected in (('history_length_ge_2', eligible), ('all_rows', np.ones(len(meta), dtype=bool))):
                for column in ('centered_delta_difference_rms', 'attention_tv'):
                    raw_values = route_diag[column].to_numpy(dtype=np.float64)
                    values = recomputed[column][selected]
                    assert np.isfinite(values).all() and (values >= 0).all()
                    finite = np.isfinite(raw_values)
                    if population == 'all_rows':
                        for row in np.flatnonzero(~finite):
                            bad_input_rows.append({'route': route, 'quantity': column, 'row_index': int(row),
                                'effective_length': int(meta.effective_length.iloc[row]),
                                'cached_mask_valid_keys': int(native_mask[row].sum()),
                                'stored_correction_finite_coordinates': int(np.isfinite(scores[route][row]).sum()),
                                'raw_value': 'nonfinite', 'recomputed_value': float(recomputed[column][row])})
                    diagnostics.append({'case': case, 'population': population, 'route': route,
                        'quantity': column, 'rows': len(values), 'mean': float(values.mean()),
                        'median': float(np.median(values)), 'p90': float(np.quantile(values, .9)),
                        'max': float(values.max()), 'raw_nonfinite_rows': int((~finite & selected).sum()),
                        'raw_finite_max_abs_recomputation_difference': float(np.max(np.abs(raw_values[finite & selected] - recomputed[column][finite & selected]))),
                        'empty_history_zero_mass_rows': int(((native_mask.sum(axis=1) == 0) & selected).sum())})
        diagnostic_notes[case] = {'native_mask_equals_frozen_cached_input': True,
            'mask_source_sample_sha256': plan['frozen_files_sha256'][str(sample_path)],
            'prior_native_attention_sha256': sha(prior_attention), 'score_by_route_sha256': sha(source / 'score_bias_by_route.npz'),
            'nonfinite_original_diagnostics': bad_input_rows,
            'recomputation': 'Float64 valid-key centering and stable masked softmax from saved native scores/corrections and independently verified cached mask. Invalid positions use np.where; empty history has zero attention mass and thus zero L1 change. Raw diagnostic files and all prediction/BCE results remain unchanged.'}
        numerical = gates['native_joint_permutation_numerical_floor']
        for mode, record in numerical['checks'].items():
            native_rows.append({'case': case, 'mode': mode, 'quantity': 'pooling', **record['pooling']})
            for label, item in record['prediction'].items():
                native_rows.append({'case': case, 'mode': mode, 'quantity': label, **item})
        selected = contrasts.loc[(contrasts.population == 'history_length_ge_2') & (contrasts.label != 'label_macro')].copy()
        selected['route_better_than_identity_point_estimate'] = selected.bce_difference < 0
        selected['route_better_than_identity_ci_below_zero'] = selected.ci_high < 0
        compact[case] = {'analysis_integrity_verified': True, 'input_prediction_sha256': summary['input_predictions_sha256'],
            'plan_sha256': summary['plan_sha256'], 'populations': summary['populations'],
            'primary_click': summary['primary_endpoints'], 'off_minus_identity_click': summary['supporting_utility_endpoint'],
            'eligible_click_metrics': metrics.loc[(metrics.population == 'history_length_ge_2') & (metrics.label == 'is_click')].to_dict('records'),
            'eligible_prediction_changes': changes.loc[changes.population == 'history_length_ge_2'].to_dict('records'),
            'all_feedback_label_contrasts': selected.to_dict('records'),
            'adverse_to_original_order_point_estimates': selected.loc[selected.contrast.isin(['reverse_minus_identity', 'random_minus_identity']) & selected.route_better_than_identity_point_estimate].to_dict('records'),
            'adverse_to_keeping_correction_point_estimates': selected.loc[(selected.contrast == 'off_minus_identity') & selected.route_better_than_identity_point_estimate].to_dict('records'),
            'eligibility_identity_with_prior': summary['historical_eligibility_comparison'],
            'native64': numerical,
            'off_scope': summary['off_scope'], 'history_order_scope': summary['history_order_scope']}
        dependency[case] = {'plan_sha256': sha(source / 'plan.json'),
            'frozen_sources_sha256': plan['frozen_files_sha256'],
            'independent_source_hash_receipt': str(analysis / 'independent_data_audit.json'),
            'independent_audit_sha256': sha(analysis / 'independent_data_audit.json')}
        for filename, expected in plan['frozen_files_sha256'].items():
            p = Path(filename)
            if p.is_relative_to(SOURCE):
                assert sha(p) == expected
                if p.name in inference:
                    assert inference[p.name]['sha256'] == expected
                inference[p.name] = {'remote_origin': filename, 'sha256': expected, 'bytes': p.stat().st_size}
    for name, info in inference.items():
        inventory[f'source_code/inference/{name}'] = info
    analysis_sources = read(HERE / 'source_manifest.json')
    for name, expected in analysis_sources['files_sha256'].items():
        assert sha(HERE / name) == expected
        inventory[f'source_code/analysis/{name}'] = {'remote_origin': str(HERE / name), 'sha256': expected, 'bytes': (HERE / name).stat().st_size}
    for name in ('source_manifest.json', 'audit_native_window.py', 'native64_nonvacuity_audit.json', 'native64_v2_rebinding.json'):
        path = HERE / name
        target = 'source_code/analysis/' + name if name.endswith('.py') or name == 'source_manifest.json' else 'independent_reports/' + name
        inventory[target] = {'remote_origin': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
    pd.DataFrame(diagnostics).to_csv(out / 'diagnostic_summary.csv', index=False)
    pd.DataFrame(native_rows).to_csv(out / 'native_numerical_errors.csv', index=False)
    for name, value in (('compact_result_summary.json', compact), ('dependency_manifest.json', dependency), ('auxiliary_diagnostic_audit.json', diagnostic_notes)):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    for path in out.iterdir():
        inventory[f'independent_reports/{path.name}'] = {'remote_origin': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
    inventory['source_code/analysis/export_delivery_audit.py'] = {'remote_origin': str(Path(__file__)), 'sha256': sha(__file__), 'bytes': Path(__file__).stat().st_size}
    (out / 'remote_copy_inventory.json').write_text(json.dumps({'files': inventory, 'scope': 'Only assigned analysis/source/independent-report files; root case JSON owned by root agent'}, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'files_for_copy': len(inventory), 'cases': {case: {'primary_click': value['primary_click'], 'off_minus_identity_click': value['off_minus_identity_click']} for case, value in compact.items()}}, indent=2))


if __name__ == '__main__':
    main()
