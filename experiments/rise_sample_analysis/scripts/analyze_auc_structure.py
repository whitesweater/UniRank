"""Exact length-pair AUC decomposition and exploratory paired user uncertainty.

B is Content-only, E is Full RISE. Fixed trained checkpoints are conditioned on;
none of the intervals estimate training-seed variation or causal component effects.
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

from analyze_groups import (KEYS, LABELS, load_columns, load_predictions,
                            sha256, unique_sorted_keys, write_csv)
from statistical_metrics import (auc_and_influence, cluster_interval,
                                 paired_auc_difference, validate_binary)

ROOT = Path(__file__).resolve().parents[3]
LENGTH_NAMES = {0: '0', 1: '1-20', 2: '21-50', 3: '51-99', 4: '100'}
BOOTSTRAP_LABELS = ('Like', 'Offer', 'Purchase')


def exact_pair_decomposition(y, scores, groups, group_count=5):
    """All positive-group/negative-group pairs; half credit for exact ties."""
    y, scores = validate_binary(y, scores)
    groups = np.asarray(groups)
    if len(groups) != len(y) or np.any(groups < 0) or np.any(groups >= group_count):
        raise ValueError('Groups must form a complete nonnegative partition')
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    if not positives or not negatives:
        raise ValueError('Overall AUC requires both classes')
    positive_scores = [scores[(groups == g) & (y == 1)] for g in range(group_count)]
    negative_scores = [np.sort(scores[(groups == h) & (y == 0)]) for h in range(group_count)]
    denominator, output = positives * negatives, []
    for g, pos in enumerate(positive_scores):
        for h, neg in enumerate(negative_scores):
            pairs = len(pos) * len(neg)
            if pairs:
                left = np.searchsorted(neg, pos, side='left')
                right = np.searchsorted(neg, pos, side='right')
                wins = float(np.sum(left + right, dtype=np.int64)) * .5
                pair_auc, contribution = wins / pairs, wins / denominator
            else:
                pair_auc, contribution = None, 0.0
            output.append({'positive_group_id': g, 'negative_group_id': h,
                           'positive_rows': len(pos), 'negative_rows': len(neg),
                           'pair_count': pairs, 'weight': pairs / denominator,
                           'pair_auc': pair_auc, 'contribution': contribution,
                           'within_group': g == h})
    return output


def align_cluster_difference(first, second):
    """A-B with common users retained, including between-group covariance."""
    estimate_a, users_a, values_a = first
    estimate_b, users_b, values_b = second
    users, inverse = np.unique(np.r_[users_a, users_b], return_inverse=True)
    contribution = np.bincount(inverse, weights=np.r_[values_a, -values_b], minlength=len(users))
    interval = cluster_interval(estimate_a - estimate_b, contribution)
    return interval, users, contribution


def prepare_weighted_auc(y, scores, user_inverse):
    """Freeze score order and tie blocks, without approximating bootstrap AUC."""
    y, scores = validate_binary(y, scores)
    if len(user_inverse) != len(y) or np.any(np.asarray(user_inverse) < 0):
        raise ValueError('User mapping must align with the rows')
    order = np.argsort(scores, kind='stable')
    ordered = scores[order]
    starts = np.r_[0, 1 + np.flatnonzero(ordered[1:] != ordered[:-1])]
    return {'starts': starts, 'y': y[order], 'users': np.asarray(user_inverse)[order]}


def exact_weighted_auc(prepared, cluster_weights):
    """Exact AUC of a cluster-resampled multiset, retaining all rows per user."""
    cluster_weights = np.asarray(cluster_weights)
    if np.any(cluster_weights < 0) or np.any(cluster_weights != np.floor(cluster_weights)):
        raise ValueError('Bootstrap cluster multiplicities must be nonnegative integers')
    if prepared['users'].max() >= len(cluster_weights):
        raise ValueError('Missing bootstrap user weights')
    weights = cluster_weights[prepared['users']].astype(np.int64, copy=False)
    positive = np.add.reduceat(weights * prepared['y'], prepared['starts'])
    negative = np.add.reduceat(weights * (1 - prepared['y']), prepared['starts'])
    npos, nneg = int(positive.sum()), int(negative.sum())
    if npos == 0 or nneg == 0:
        return None, npos, nneg
    placements = np.cumsum(negative) - .5 * negative
    return float(np.dot(positive, placements) / (npos * nneg)), npos, nneg


def update_matrix_digest(digest, columns, dtype):
    n = len(columns[0])
    for start in range(0, n, 131072):
        digest.update(np.column_stack([v[start:start + 131072] for v in columns]).astype(dtype).tobytes())


def contrast_masks(features):
    length, entropy, fraction = [features[name] for name in
                                ['effective_length', 'c1_entropy', 'target_c1_history_fraction']]
    fixed100 = length == 100
    return [
        ('length100_minus_length1_20', 'effective_length', '100', '1-20',
         fixed100, (length >= 1) & (length <= 20)),
        ('entropy_gt_ln5_minus_ln2_to_ln5_at100', 'c1_entropy_at_length100',
         'effective_categories_gt5', 'effective_categories_2_to_5',
         fixed100 & (entropy > np.log(5)), fixed100 & (entropy > np.log(2)) & (entropy <= np.log(5))),
        ('candidate_fraction_0_to_025_minus_gt075_at100', 'candidate_category_support_at_length100',
         'positive_le_0.25', 'gt_0.75',
         fixed100 & (fraction > 0) & (fraction <= .25), fixed100 & (fraction > .75)),
    ]


def run_decomposition(inputs, length_group, receipts):
    pairs, summaries = [], []
    for label in LABELS:
        y = inputs['B'][label]
        values, direct = {}, {}
        for variant in ['B', 'E']:
            scores = inputs[variant][label + '_pred']
            values[variant] = exact_pair_decomposition(y, scores, length_group)
            direct[variant], influence = auc_and_influence(y, scores)
            del influence
            if abs(sum(row['contribution'] for row in values[variant]) - direct[variant]) > 2e-12:
                raise AssertionError('Pair decomposition does not reconstruct exact full AUC')
            if abs(direct[variant] - receipts[variant]['metrics'][label]['AUC']) > 2e-12:
                raise AssertionError('Global exact AUC differs from formal prediction receipt')
        within, between = {}, {}
        for variant in ['B', 'E']:
            within[variant] = sum(r['contribution'] for r in values[variant] if r['within_group'])
            between[variant] = sum(r['contribution'] for r in values[variant] if not r['within_group'])
        for before, after in zip(values['B'], values['E']):
            assert before['weight'] == after['weight']
            pairs.append({'label': label, 'positive_length_group': LENGTH_NAMES[before['positive_group_id']],
                          'negative_length_group': LENGTH_NAMES[before['negative_group_id']],
                          'positive_rows': before['positive_rows'], 'negative_rows': before['negative_rows'],
                          'positive_negative_pairs': before['pair_count'], 'pair_weight': before['weight'],
                          'macro_label_weight': .2, 'within_group': before['within_group'],
                          'B_pair_auc': before['pair_auc'], 'E_pair_auc': after['pair_auc'],
                          'B_auc_contribution': before['contribution'], 'E_auc_contribution': after['contribution'],
                          'delta_auc_contribution': after['contribution'] - before['contribution']})
        delta = direct['E'] - direct['B']
        wd, bd = within['E'] - within['B'], between['E'] - between['B']
        summaries.append({'label': label, 'B_global_auc': direct['B'], 'E_global_auc': direct['E'],
                          'delta_auc': delta, 'within_pair_weight': sum(r['weight'] for r in values['B'] if r['within_group']),
                          'between_pair_weight': sum(r['weight'] for r in values['B'] if not r['within_group']),
                          'within_B_contribution': within['B'], 'within_E_contribution': within['E'],
                          'within_delta': wd, 'between_B_contribution': between['B'],
                          'between_E_contribution': between['E'], 'between_delta': bd,
                          'delta_reconstruction_residual': wd + bd - delta})
        print(json.dumps({'stage': 'exact_decomposition', 'label': label, 'reconciled': True}), flush=True)
    numeric = [key for key in summaries[0] if key != 'label']
    summaries.append({'label': 'macro_5_tasks', **{key: float(np.mean([r[key] for r in summaries])) for key in numeric}})
    macro_pairs = []
    for g in LENGTH_NAMES.values():
        for h in LENGTH_NAMES.values():
            selected = [r for r in pairs if r['positive_length_group'] == g and r['negative_length_group'] == h]
            macro_pairs.append({'label': 'macro_5_tasks', 'positive_length_group': g, 'negative_length_group': h,
                                'mean_label_pair_weight': float(np.mean([r['pair_weight'] for r in selected])),
                                'within_group': g == h,
                                **{key: float(np.mean([r[key] for r in selected])) for key in
                                   ['B_auc_contribution', 'E_auc_contribution', 'delta_auc_contribution']}})
    if abs(sum(r['delta_auc_contribution'] for r in macro_pairs) - summaries[-1]['delta_auc']) > 2e-12:
        raise AssertionError('Equal-label macro pair contributions do not reconcile')
    return pairs, macro_pairs, summaries


def run_contrasts(features, inputs, reference_effects):
    rows = []
    for name, dimension, first_name, second_name, first, second in contrast_masks(features):
        macro_clusters, estimates, macro_users = [], [], None
        for label in LABELS:
            effects = []
            for mask, group in [(first, first_name), (second, second_name)]:
                point, users, cluster_if = paired_auc_difference(inputs['B'][label][mask],
                    inputs['B'][label + '_pred'][mask], inputs['E'][label + '_pred'][mask], features['user_id'][mask])
                reference = reference_effects[(dimension, group, label, 'E-B')]
                if abs(point['estimate'] - float(reference['delta_auc'])) > 2e-12:
                    raise AssertionError('Condition effect differs from completed root group analysis')
                effects.append((point, users, cluster_if))
            a, b = effects
            interval, union, cluster_if = align_cluster_difference(
                (a[0]['estimate'], a[1], a[2]), (b[0]['estimate'], b[1], b[2]))
            if macro_users is None:
                macro_users = union
            if not np.array_equal(union, macro_users):
                raise AssertionError('Labels do not share the same contrast user population')
            macro_clusters.append(cluster_if)
            estimates.append(interval['estimate'])
            rows.append({'contrast': name, 'comparison': 'E-B', 'analysis_origin': 'supplemental exploratory contrast frozen before this execution', 'label': label, 'first_group': first_name, 'second_group': second_name,
                         'first_rows': int(first.sum()), 'second_rows': int(second.sum()),
                         'first_positives': a[0]['positives'], 'second_positives': b[0]['positives'],
                         'first_effect_E_minus_B': a[0]['estimate'], 'second_effect_E_minus_B': b[0]['estimate'],
                         'shared_users': int(np.intersect1d(a[1], b[1]).size),
                         **interval, 'interpretation': 'exploratory nominal95; shared-user covariance; fixed trained models'})
        value = float(np.mean(estimates))
        interval = cluster_interval(value, np.mean(macro_clusters, axis=0))
        rows.append({'contrast': name, 'comparison': 'E-B', 'analysis_origin': 'supplemental exploratory contrast frozen before this execution', 'label': 'macro_5_tasks', 'first_group': first_name, 'second_group': second_name,
                     'first_rows': int(first.sum()), 'second_rows': int(second.sum()),
                     **interval, 'macro_labels': 5,
                     'interpretation': 'exploratory nominal95; average five label influences before variance; no seed replication'})
        print(json.dumps({'stage': 'contrast', 'contrast': name, 'labels': 5}), flush=True)
    return rows


def posthoc_offer_contrast(features, inputs, references):
    label = 'Offer'
    length = features['effective_length']
    masks = [(length == 100, '100'), ((length >= 1) & (length <= 20), '1-20')]
    effects = []
    for mask, name in masks:
        point, users, influence = paired_auc_difference(inputs['B'][label][mask],
            inputs['D'][label + '_pred'][mask], inputs['E'][label + '_pred'][mask], features['user_id'][mask])
        if abs(point['estimate'] - float(references[('effective_length', name, label, 'E-D')]['delta_auc'])) > 2e-12:
            raise AssertionError('Posthoc Offer condition effect differs from completed group analysis')
        effects.append((point, users, influence))
    a, b = effects
    interval, _, _ = align_cluster_difference((a[0]['estimate'], a[1], a[2]), (b[0]['estimate'], b[1], b[2]))
    return {'contrast': 'posthoc_offer_ED_length100_minus_length1_20', 'comparison': 'E-D',
            'analysis_origin': 'posthoc exploratory; appended after observing Offer subgroup E-D results; not preregistered',
            'label': label, 'first_group': '100', 'second_group': '1-20',
            'first_rows': int(masks[0][0].sum()), 'second_rows': int(masks[1][0].sum()),
            'first_positives': a[0]['positives'], 'second_positives': b[0]['positives'],
            'first_effect_E_minus_D': a[0]['estimate'], 'second_effect_E_minus_D': b[0]['estimate'],
            'shared_users': int(np.intersect1d(a[1], b[1]).size), **interval,
            'interpretation': 'posthoc nominal95; shared-user covariance; one selected label; no independent seed or causal AM claim'}


def run_bootstrap(features, inputs, max_users, repetitions, selection_seed, bootstrap_seed, output):
    unique_users = np.unique(features['user_id'])
    chosen = np.sort(np.random.default_rng(selection_seed).choice(unique_users, size=min(max_users, len(unique_users)), replace=False))
    selected = np.isin(features['user_id'], chosen)
    users = features['user_id'][selected]
    inverse = np.searchsorted(chosen, users)
    rng = np.random.default_rng(bootstrap_seed)
    weights = rng.multinomial(len(chosen), np.full(len(chosen), 1 / len(chosen)), size=repetitions).astype(np.int32)
    write_csv(output / 'bootstrap_selected_users.csv', [{'user_id': int(user)} for user in chosen])
    sample_key = hashlib.sha256()
    update_matrix_digest(sample_key, [features['part_id'][selected], features['row_offset'][selected]], '<i8')
    replicates, summaries = [], []
    for label in BOOTSTRAP_LABELS:
        y = inputs['B'][label][selected]
        scores = {v: inputs[v][label + '_pred'][selected] for v in ['B', 'E']}
        npos, nneg = int(y.sum()), int(len(y) - y.sum())
        if npos and nneg:
            linear, linear_users, _ = paired_auc_difference(y, scores['B'], scores['E'], users)
            if not np.array_equal(linear_users, chosen):
                raise AssertionError('Selected user bootstrap drops a selected user')
        else:
            linear = {'estimate': None, 'cluster_se': None, 'ci_lower': None, 'ci_upper': None}
        prepared = {v: prepare_weighted_auc(y, scores[v], inverse) for v in ['B', 'E']}
        deltas, invalid = [], {'missing_positive': 0, 'missing_negative': 0}
        for replicate, multiplicities in enumerate(weights):
            before, p, n = exact_weighted_auc(prepared['B'], multiplicities)
            after, p_after, n_after = exact_weighted_auc(prepared['E'], multiplicities)
            if (p, n) != (p_after, n_after):
                raise AssertionError('Paired bootstrap populations differ')
            if p == 0:
                invalid['missing_positive'] += 1
            if n == 0:
                invalid['missing_negative'] += 1
            delta = None if before is None or after is None else after - before
            if delta is not None:
                deltas.append(delta)
            replicates.append({'label': label, 'replicate': replicate, 'weighted_positive_rows': p,
                               'weighted_negative_rows': n, 'B_auc': before, 'E_auc': after,
                               'delta_auc': delta, 'valid': delta is not None})
        enough = len(deltas) > 1
        bootstrap_se = float(np.std(deltas, ddof=1)) if enough else None
        percentiles = np.quantile(deltas, [.025, .975], method='linear') if enough else [None, None]
        ratio = bootstrap_se / linear['cluster_se'] if enough and linear['cluster_se'] else None
        summaries.append({'label': label, 'sample_users': len(chosen), 'sample_rows': int(selected.sum()),
                          'positive_rows': npos, 'negative_rows': nneg,
                          'users_with_positive_rows': int(np.unique(users[y == 1]).size),
                          'users_with_negative_rows': int(np.unique(users[y == 0]).size),
                          'point_delta_auc': linear['estimate'], 'root_linearized_se_same_subsample': linear['cluster_se'],
                          'linearized_ci_lower': linear['ci_lower'], 'linearized_ci_upper': linear['ci_upper'],
                          'bootstrap_repetitions': repetitions, 'valid_repetitions': len(deltas),
                          'invalid_missing_positive': invalid['missing_positive'], 'invalid_missing_negative': invalid['missing_negative'],
                          'exact_user_bootstrap_se': bootstrap_se, 'bootstrap_to_linearized_se_ratio': ratio,
                          'bootstrap_percentile_lower': percentiles[0], 'bootstrap_percentile_upper': percentiles[1],
                          'bootstrap_mean_delta': float(np.mean(deltas)) if deltas else None,
                          'bootstrap_bias_relative_to_sample_point': float(np.mean(deltas) - linear['estimate']) if deltas and linear['estimate'] is not None else None,
                          'gate': 'No forced pass/fail; invalid replicates reported, not resampled until success',
                          'scope': 'same fixed user subsample for exact bootstrap and root linearization; conditional on fixed trained models'})
        print(json.dumps({'stage': 'exact_user_bootstrap', 'label': label, 'valid': len(deltas), 'repetitions': repetitions}), flush=True)
    meta = {'selection_seed': selection_seed, 'bootstrap_seed': bootstrap_seed, 'selection': 'Uniform without replacement among all retained test users; keep every retained row of selected users.',
            'sample_users': len(chosen), 'sample_rows': int(selected.sum()), 'selected_user_ids_sha256': hashlib.sha256(chosen.astype('<i8').tobytes()).hexdigest(),
            'sample_row_key_sha256': sample_key.hexdigest(), 'multinomial_weights_sha256': hashlib.sha256(weights.astype('<i4').tobytes()).hexdigest(),
            'repetitions': repetitions, 'resampling': 'K independent uniform draws with replacement from K selected users; exact weighted AUC on resulting multiset; paired multiplicities shared by labels/models.',
            'percentile_method': 'numpy quantile linear; descriptive if invalid replicates occur', 'numpy_version': np.__version__}
    return replicates, summaries, meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-dir', type=Path, default=ROOT / 'artifacts/rise_sample_analysis/predictions/formal_20260923')
    parser.add_argument('--feature-dir', type=Path, default=ROOT / 'artifacts/rise_sample_analysis/group_features/features')
    parser.add_argument('--group-analysis-dir', type=Path, default=ROOT / 'experiments/rise_sample_analysis/results/group_analysis_components_v1')
    parser.add_argument('--population-profile-dir', type=Path, default=ROOT / 'experiments/rise_sample_analysis/results')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-users', type=int, default=10000)
    parser.add_argument('--bootstrap-repetitions', type=int, default=500)
    parser.add_argument('--selection-seed', type=int, default=20260923)
    parser.add_argument('--bootstrap-seed', type=int, default=20260924)
    args = parser.parse_args(argv)
    if not 1 < args.max_users <= 10000 or args.bootstrap_repetitions != 500:
        raise ValueError('Formal design fixes max10000 users and exactly500 bootstrap repetitions')
    pa.set_cpu_count(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), Path(__file__).with_name('statistical_metrics.py'), Path(__file__).with_name('analyze_groups.py'),
                    args.population_profile_dir / 'population_profile_summary.json', args.population_profile_dir / 'population_profile_sources.csv',
                    args.group_analysis_dir / 'manifest.json', args.group_analysis_dir / 'complete.json', args.group_analysis_dir / 'group_effects.csv']
    source_sha = {str(p.resolve()): sha256(p) for p in source_files}
    population = json.loads((args.population_profile_dir / 'population_profile_summary.json').read_text())
    audited_sources = {str((ROOT / r['path']).resolve()): r['sha256'] for r in csv.DictReader((args.population_profile_dir / 'population_profile_sources.csv').open())}
    feature_paths = sorted(args.feature_dir.glob('part-*.parquet'))
    if len(feature_paths) != 8:
        raise ValueError('Eight frozen feature parts required')
    feature_hashes = {}
    for path in feature_paths:
        value = sha256(path)
        if audited_sources.get(str(path.resolve())) != value:
            raise ValueError('Feature content differs from independently completed population profile')
        feature_hashes[str(path.resolve())] = value
    features = load_columns(feature_paths, list(KEYS) + ['c1_entropy', 'target_c1_history_fraction'])
    keys = unique_sorted_keys(features)
    inputs, receipts = {}, {}
    for variant in ['B', 'D', 'E']:
        inputs[variant], receipts[variant] = load_predictions(args.prediction_dir / variant)
        if not np.array_equal(unique_sorted_keys(inputs[variant]), keys):
            raise ValueError('Prediction-feature physical identity mismatch')
        for key in KEYS:
            if not np.array_equal(features[key], inputs[variant][key]):
                raise ValueError(f'Prediction-feature metadata mismatch: {key}')
    if any(receipts['B']['data_version'] != receipts[v]['data_version'] for v in ['D', 'E']):
        raise ValueError('Prediction dataset versions disagree')
    for label in LABELS:
        if any(not np.array_equal(inputs['B'][label], inputs[v][label]) for v in ['D', 'E']):
            raise ValueError('Paired labels disagree')
    row_digest, label_digest = hashlib.sha256(), hashlib.sha256()
    update_matrix_digest(row_digest, [features['part_id'], features['row_offset']], '<i8')
    update_matrix_digest(label_digest, [inputs['B'][label] for label in LABELS], '<f4')
    if row_digest.hexdigest() != population['row_key_sha256'] or label_digest.hexdigest() != population['labels_sha256']:
        raise ValueError('Predictions do not match population-profile sample keys and source labels')
    if len(keys) != 11599872:
        raise ValueError('Formal test row count mismatch')
    print(json.dumps({'stage': 'input_audit_passed', 'rows': len(keys), 'variants': ['B', 'D', 'E']}), flush=True)
    length = features['effective_length']
    length_group = np.select([length == 0, length <= 20, length <= 50, length < 100], [0, 1, 2, 3], default=4)
    pairs, macro_pairs, structures = run_decomposition(inputs, length_group, receipts)
    write_csv(args.output_dir / 'length_pair_auc_decomposition.csv', pairs)
    write_csv(args.output_dir / 'length_pair_auc_decomposition_macro.csv', macro_pairs)
    write_csv(args.output_dir / 'length_auc_structure_summary.csv', structures)
    references = {(r['dimension'], r['group'], r['label'], r['comparison']): r
                  for r in csv.DictReader((args.group_analysis_dir / 'group_effects.csv').open())}
    contrasts = run_contrasts(features, inputs, references)
    contrasts.append(posthoc_offer_contrast(features, inputs, references))
    write_csv(args.output_dir / 'exploratory_group_effect_contrasts.csv', contrasts)
    replicates, bootstrap, bootstrap_meta = run_bootstrap(features, inputs, args.max_users,
        args.bootstrap_repetitions, args.selection_seed, args.bootstrap_seed, args.output_dir)
    write_csv(args.output_dir / 'exact_user_bootstrap_replicates.csv', replicates)
    write_csv(args.output_dir / 'exact_user_bootstrap_summary.csv', bootstrap)
    if any(sha256(path) != before for path, before in source_sha.items()):
        raise ValueError('Analysis source changed during this run')
    manifest = {'status': 'complete', 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'finished_utc': datetime.now(timezone.utc).isoformat(), 'rows': len(keys), 'labels': LABELS,
                'variants': ['B', 'D', 'E'], 'B_definition': 'Content-only auxiliary branch, not independently trained native baseline',
                'row_key_sha256': row_digest.hexdigest(), 'labels_sha256': label_digest.hexdigest(),
                'sources_sha256': source_sha, 'features_sha256': feature_hashes,
                'prediction_complete_sha256': {v: sha256(args.prediction_dir / v / 'complete.json') for v in ['B', 'D', 'E']},
                'prediction_validation': 'Every shard content SHA and identity receipt verified by unchanged root loader; all three variants align on all six metadata fields and five labels.',
                'decomposition': 'Each label uses P_g*N_h/(P*N); macro averages five label contributions with weight1/5, never sample-weights within-group AUC.',
                'contrasts': 'Three contrasts frozen in source before this supplemental execution, after existing group analyses; exploratory, not preregistered confirmatory tests. Nominal normal95 with shared-user and across-label covariance retained; no multiplicity correction.',
                'posthoc_contrast': 'Offer only: (E-D) at length100 minus (E-D) at length1-20; selected after observed subgroup results; separately tagged and excluded from E-B macro comparisons.',
                'bootstrap': bootstrap_meta,
                'limitations': 'Conditional on fixed trained models; no training-seed uncertainty or causal identification. Bootstrap diagnoses same fixed user subset, not the full-population SE. Invalid class replicates are retained and counted.'}
    outputs = {p.name: sha256(p) for p in args.output_dir.iterdir() if p.is_file()}
    manifest['output_sha256'] = outputs
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output_dir / 'complete.json').write_text(json.dumps({'rows': len(keys), 'identity_verified': True,
        'length_pairs': len(pairs), 'macro_pairs': len(macro_pairs), 'contrast_rows': len(contrasts),
        'bootstrap_replicate_rows': len(replicates), 'bootstrap_summary_rows': len(bootstrap),
        'maximum_decomposition_residual': max(abs(r['delta_reconstruction_residual']) for r in structures)}) + '\n')
    print(json.dumps({'status': 'complete', 'rows': len(keys)}), flush=True)


if __name__ == '__main__':
    main()
