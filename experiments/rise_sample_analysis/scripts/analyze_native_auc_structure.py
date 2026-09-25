"""Native HyFormer(A) versus Full RISE(E): exact AUC structure and group contrasts.

Reuses verified pure mathematical functions. No bootstrap is rerun, and existing
component results or prediction files are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa

from analyze_groups import KEYS, LABELS, load_columns, load_predictions, sha256, unique_sorted_keys, write_csv
from analyze_auc_structure import LENGTH_NAMES, exact_pair_decomposition, align_cluster_difference, contrast_masks, update_matrix_digest
from statistical_metrics import auc_and_influence, paired_auc_difference, cluster_interval

ROOT = Path(__file__).resolve().parents[3]


def native_pair_table(y, native_scores, rise_scores, groups):
    native = exact_pair_decomposition(y, native_scores, groups)
    rise = exact_pair_decomposition(y, rise_scores, groups)
    rows = []
    for before, after in zip(native, rise):
        if before['weight'] != after['weight']:
            raise AssertionError('A/E pair weights differ')
        rows.append({'positive_length_group': LENGTH_NAMES[before['positive_group_id']],
                     'negative_length_group': LENGTH_NAMES[before['negative_group_id']],
                     'positive_rows': before['positive_rows'], 'negative_rows': before['negative_rows'],
                     'positive_negative_pairs': before['pair_count'], 'pair_weight': before['weight'],
                     'within_group': before['within_group'], 'A_pair_auc': before['pair_auc'],
                     'E_pair_auc': after['pair_auc'], 'A_contribution': before['contribution'],
                     'E_contribution': after['contribution'],
                     'delta_contribution': after['contribution'] - before['contribution']})
    return rows


def native_decomposition(features, inputs, receipts):
    length = features['effective_length']
    groups = np.select([length == 0, length <= 20, length <= 50, length < 100], [0, 1, 2, 3], default=4)
    users, inverse = np.unique(features['user_id'], return_inverse=True)
    pairs, summary, macro_if = [], [], []
    for label in LABELS:
        y = inputs['A'][label]
        tables = native_pair_table(y, inputs['A'][label + '_pred'], inputs['E'][label + '_pred'], groups)
        points, influences = {}, {}
        for variant in ['A', 'E']:
            points[variant], influences[variant] = auc_and_influence(y, inputs[variant][label + '_pred'])
            if abs(points[variant] - receipts[variant]['metrics'][label]['AUC']) > 2e-12:
                raise AssertionError('A/E exact AUC differs from formal prediction receipt')
            if abs(sum(row[f'{variant}_contribution'] for row in tables) - points[variant]) > 2e-12:
                raise AssertionError('A/E pair decomposition does not reconstruct global AUC')
        contribution = np.bincount(inverse, weights=influences['E'] - influences['A'], minlength=len(users))
        macro_if.append(contribution)
        delta = points['E'] - points['A']
        interval = cluster_interval(delta, contribution)
        within = sum(row['delta_contribution'] for row in tables if row['within_group'])
        between = sum(row['delta_contribution'] for row in tables if not row['within_group'])
        summary.append({'label': label, 'N': len(y), 'positive_rows': int(y.sum()),
                        'A_native_auc': points['A'], 'E_full_rise_auc': points['E'],
                        'delta_auc': delta, 'cluster_se': interval['cluster_se'],
                        'ci_lower': interval['ci_lower'], 'ci_upper': interval['ci_upper'],
                        'within_delta': within, 'between_delta': between,
                        'reconstruction_residual': within + between - delta})
        pairs += [{'label': label, **row} for row in tables]
        print(json.dumps({'stage': 'native_auc_decomposition', 'label': label, 'reconciled': True}), flush=True)
    numeric = ['A_native_auc', 'E_full_rise_auc', 'delta_auc', 'within_delta', 'between_delta', 'reconstruction_residual']
    macro = {key: float(np.mean([row[key] for row in summary])) for key in numeric}
    interval = cluster_interval(macro['delta_auc'], np.mean(macro_if, axis=0))
    summary.append({'label': 'macro_5_tasks', 'N': len(groups), 'positive_rows': None, **macro,
                    'cluster_se': interval['cluster_se'], 'ci_lower': interval['ci_lower'], 'ci_upper': interval['ci_upper']})
    macro_pairs = []
    for g in LENGTH_NAMES.values():
        for h in LENGTH_NAMES.values():
            selected = [r for r in pairs if r['positive_length_group'] == g and r['negative_length_group'] == h]
            macro_pairs.append({'positive_length_group': g, 'negative_length_group': h,
                                'within_group': g == h, 'label_weight': .2,
                                **{key: float(np.mean([r[key] for r in selected])) for key in
                                   ['pair_weight', 'A_contribution', 'E_contribution', 'delta_contribution']}})
    if abs(sum(r['delta_contribution'] for r in macro_pairs) - macro['delta_auc']) > 2e-12:
        raise AssertionError('Native macro pair contributions do not reconcile')
    return pairs, macro_pairs, summary


def native_contrasts(features, inputs, references):
    output = []
    for name, dimension, first_name, second_name, first, second in contrast_masks(features):
        label_effects, all_if, common_users = [], [], None
        for label in LABELS:
            payloads = []
            for mask, group in [(first, first_name), (second, second_name)]:
                point, users, influence = paired_auc_difference(inputs['A'][label][mask], inputs['A'][label + '_pred'][mask],
                    inputs['E'][label + '_pred'][mask], features['user_id'][mask])
                if abs(point['estimate'] - float(references[(dimension, group, label, 'E-A')]['delta_auc'])) > 2e-12:
                    raise AssertionError('Native group effect differs from accepted five-configuration analysis')
                payloads.append((point, users, influence))
            a, b = payloads
            interval, union, influence = align_cluster_difference((a[0]['estimate'], a[1], a[2]), (b[0]['estimate'], b[1], b[2]))
            if common_users is None:
                common_users = union
            if not np.array_equal(union, common_users):
                raise AssertionError('Macro task populations differ')
            label_effects.append(interval['estimate'])
            all_if.append(influence)
            output.append({'contrast': name, 'label': label, 'comparison': 'E-A',
                           'first_group': first_name, 'second_group': second_name,
                           'first_rows': int(first.sum()), 'second_rows': int(second.sum()),
                           'first_positives': a[0]['positives'], 'second_positives': b[0]['positives'],
                           'first_delta_E_minus_A': a[0]['estimate'], 'second_delta_E_minus_A': b[0]['estimate'],
                           'shared_users': int(np.intersect1d(a[1], b[1]).size), **interval})
        interval = cluster_interval(float(np.mean(label_effects)), np.mean(all_if, axis=0))
        output.append({'contrast': name, 'label': 'macro_5_tasks', 'comparison': 'E-A',
                       'first_group': first_name, 'second_group': second_name,
                       'first_rows': int(first.sum()), 'second_rows': int(second.sum()),
                       'macro_labels': 5, **interval})
        print(json.dumps({'stage': 'native_contrast', 'contrast': name}), flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-dir', type=Path, default=ROOT / 'artifacts/rise_sample_analysis/predictions/native_merrec_20260923/A')
    parser.add_argument('--rise-dir', type=Path, default=ROOT / 'artifacts/rise_sample_analysis/predictions/formal_20260923/E')
    parser.add_argument('--feature-dir', type=Path, default=ROOT / 'artifacts/rise_sample_analysis/group_features/features')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    pa.set_cpu_count(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = ROOT / 'experiments/rise_sample_analysis/results'
    reference_file = results / 'group_analysis_native_v1/group_effects.csv'
    reference_manifest = results / 'group_analysis_native_v1/manifest.json'
    profile_file = results / 'population_profile_summary.json'
    profiles_sources = results / 'population_profile_sources.csv'
    source_paths = [Path(__file__), Path(__file__).with_name('analyze_auc_structure.py'),
                    Path(__file__).with_name('analyze_groups.py'), Path(__file__).with_name('statistical_metrics.py'),
                    reference_file, reference_manifest, profile_file, profiles_sources]
    source_sha = {str(p.resolve()): sha256(p) for p in source_paths}
    profile = json.loads(profile_file.read_text())
    known_hashes = {str((ROOT / r['path']).resolve()): r['sha256'] for r in csv.DictReader(profiles_sources.open())}
    features_paths = sorted(args.feature_dir.glob('part-*.parquet'))
    if len(features_paths) != 8:
        raise ValueError('Eight feature parts required')
    for path in features_paths:
        if sha256(path) != known_hashes.get(str(path.resolve())):
            raise ValueError('Feature content differs from verified population profile')
    features = load_columns(features_paths, list(KEYS) + ['c1_entropy', 'target_c1_history_fraction'])
    keys = unique_sorted_keys(features)
    inputs, receipts = {}, {}
    for variant, directory in [('A', args.native_dir), ('E', args.rise_dir)]:
        inputs[variant], receipts[variant] = load_predictions(directory)
        if receipts[variant]['variant'] != variant:
            raise ValueError('Prediction variant does not match native/Full RISE role')
        if not np.array_equal(keys, unique_sorted_keys(inputs[variant])):
            raise ValueError('Prediction-feature row keys differ')
        for field in KEYS:
            if not np.array_equal(inputs[variant][field], features[field]):
                raise ValueError(f'Prediction metadata differs: {field}')
    if receipts['A']['data_version'] != receipts['E']['data_version']:
        raise ValueError('Dataset versions differ')
    for label in LABELS:
        if not np.array_equal(inputs['A'][label], inputs['E'][label]):
            raise ValueError('Native/RISE labels differ')
    row_hash, label_hash = hashlib.sha256(), hashlib.sha256()
    update_matrix_digest(row_hash, [features['part_id'], features['row_offset']], '<i8')
    update_matrix_digest(label_hash, [inputs['A'][label] for label in LABELS], '<f4')
    if len(keys) != 11599872 or row_hash.hexdigest() != profile['row_key_sha256'] or label_hash.hexdigest() != profile['labels_sha256']:
        raise ValueError('Native population identity gate failed')
    print(json.dumps({'stage': 'native_input_audit_passed', 'rows': len(keys)}), flush=True)
    pairs, macro_pairs, summary = native_decomposition(features, inputs, receipts)
    references = {(r['dimension'], r['group'], r['label'], r['comparison']): r for r in csv.DictReader(reference_file.open())}
    contrasts = native_contrasts(features, inputs, references)
    for name, rows in [('length_pair_auc_decomposition.csv', pairs), ('length_pair_auc_decomposition_macro.csv', macro_pairs),
                       ('global_and_structure_metrics.csv', summary), ('native_exploratory_group_contrasts.csv', contrasts)]:
        write_csv(args.output_dir / name, rows)
    if any(sha256(path) != value for path, value in source_sha.items()):
        raise ValueError('Analysis inputs changed during execution')
    manifest = {'status': 'complete', 'finished_utc': datetime.now(timezone.utc).isoformat(),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'comparison': 'Full RISE(E) minus native HyFormer(A)',
                'seed': 20262028, 'rows': len(keys), 'source_sha256': source_sha,
                'row_key_sha256': row_hash.hexdigest(), 'labels_sha256': label_hash.hexdigest(),
                'prediction_complete_sha256': {'A': sha256(args.native_dir / 'complete.json'), 'E': sha256(args.rise_dir / 'complete.json')},
                'feature_sha256': {str(p.resolve()): sha256(p) for p in features_paths},
                'prediction_gate': 'All shard content hashes/identity receipts, six metadata columns, labels and current population SHA verified.',
                'statistics': 'Exact positive-group/negative-group AUC decomposition per label; equal-five-label macro; user-cluster normal95 influence intervals retaining group and label covariance.',
                'contrasts': 'Same three group contrasts inherited from exploratory component analysis; no preregistration or multiplicity correction; fixed trained models only.',
                'bootstrap': 'Not rerun; prior E-B bootstrap is not relabeled as a validation of native E-A intervals.',
                'output_sha256': {p.name: sha256(p) for p in args.output_dir.glob('*.csv')}}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output_dir / 'complete.json').write_text(json.dumps({'identity_verified': True, 'rows': len(keys),
        'label_pair_rows': len(pairs), 'macro_pair_rows': len(macro_pairs), 'contrast_rows': len(contrasts),
        'maximum_auc_reconstruction_residual': max(abs(r['reconstruction_residual']) for r in summary)}) + '\n')
    print(json.dumps({'status': 'complete', 'rows': len(keys)}), flush=True)


if __name__ == '__main__':
    main()
