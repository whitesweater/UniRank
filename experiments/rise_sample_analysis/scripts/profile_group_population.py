"""Describe retained MerRec populations; never read model predictions or infer gains."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
LABELS = ('Like', 'Cart', 'Offer', 'Checkout', 'Purchase')
FEATURE_COLUMNS = ['part_id', 'row_offset', 'user_id', 'item_id', 'seq_len',
                   'effective_length', 'user_train_target_count', 'item_train_target_count',
                   'c1_known_count', 'c1_unknown_count', 'c1_ambiguous_count',
                   'c1_entropy', 'target_c1_id', 'target_c1_known',
                   'target_c1_history_fraction']


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def file_identity(path, content_hash=False):
    path = Path(path).resolve()
    stat = path.stat()
    try:
        name = str(path.relative_to(ROOT))
    except ValueError:
        name = str(path)
    return {'path': name, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'sha256': digest(path) if content_hash else ''}


def verify_source_identity(expected, allow_output_mtime_drift=False):
    path = Path(expected['path'])
    stat = path.stat()
    if stat.st_size != expected['bytes']:
        raise ValueError(f'Producer source size changed: {path}')
    if stat.st_mtime_ns != expected['mtime_ns']:
        if not allow_output_mtime_drift:
            raise ValueError(f'Producer source changed: {path}')
        return {'path': str(path), 'bytes_unchanged': stat.st_size,
                'producer_mtime_ns': expected['mtime_ns'], 'current_mtime_ns': stat.st_mtime_ns,
                'mtime_difference_ns': stat.st_mtime_ns - expected['mtime_ns'],
                'scope': 'Generated feature output only. Current content SHA and full row-key/raw alignment are verified separately; original output content SHA was not recorded.'}
    return None


def weighted_quantile(values, weights, probabilities):
    values, weights = np.asarray(values), np.asarray(weights, np.int64)
    if not len(values) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('No positive-weight distribution')
    cumulative = np.cumsum(weights)
    ranks = np.maximum(1, np.ceil(np.asarray(probabilities) * cumulative[-1]).astype(np.int64))
    return values[np.searchsorted(cumulative, ranks, side='left')]


def frequency_profile(counts):
    positive = counts[counts > 0]
    if not len(positive):
        raise ValueError('No observed training entities')
    values, weights = np.unique(positive, return_counts=True)
    cuts = weighted_quantile(values, weights, [.25, .75])
    return {'distinct_observed_entities': int(len(positive)), 'target_rows': int(positive.sum()),
            'q25': int(cuts[0]), 'q75': int(cuts[1]),
            'quantile_method': 'inverted_cdf',
            'quantile_population': 'distinct global IDs with positive train target frequency'}, values, weights


def group_dimensions(features, cuts):
    length = features['effective_length']
    yield 'effective_length', np.select([length == 0, length <= 20, length <= 50, length < 100],
                                       [0, 1, 2, 3], default=4), {
        0: '0', 1: '1-20', 2: '21-50', 3: '51-99', 4: '100'}
    for entity in ('item', 'user'):
        low, high = cuts[entity]
        counts = features[f'{entity}_train_target_count']
        group = np.full(len(counts), 3, np.int8)
        group[counts <= high] = 2
        group[counts <= low] = 1
        group[counts == 0] = 0
        yield f'{entity}_train_target_frequency', group, {
            0: 'train_target_unseen', 1: f'positive_le_{low}',
            2: f'gt_{low}_le_{high}', 3: f'gt_{high}'}
    selected = length == 100
    entropy = features['c1_entropy']
    group = np.full(len(length), -1, np.int8)
    group[selected] = np.select([~np.isfinite(entropy[selected]), entropy[selected] <= 1e-7,
                                 entropy[selected] <= np.log(2), entropy[selected] <= np.log(5)],
                                [0, 1, 2, 3], default=4)
    yield 'c1_entropy_at_length100', group, {0: 'undefined', 1: 'one_effective_category',
        2: 'effective_categories_1_to_2', 3: 'effective_categories_2_to_5', 4: 'effective_categories_gt5'}
    fraction = features['target_c1_history_fraction']
    group = np.full(len(length), -1, np.int8)
    group[selected] = np.select([~np.isfinite(fraction[selected]), fraction[selected] == 0,
                                 fraction[selected] <= .25, fraction[selected] <= .75],
                                [0, 1, 2, 3], default=4)
    yield 'candidate_category_support_at_length100', group, {0: 'undefined', 1: 'zero',
        2: 'positive_le_0.25', 3: 'gt_0.25_le_0.75', 4: 'gt_0.75'}


def update_groups(groups, features, labels, cuts):
    for dimension, membership, names in group_dimensions(features, cuts):
        for group_id, name in names.items():
            mask = membership == group_id
            key = (dimension, int(group_id), name)
            state = groups.setdefault(key, {'rows': 0, 'positive': np.zeros(len(LABELS), np.int64)})
            state['rows'] += int(mask.sum())
            state['positive'] += labels[mask].sum(axis=0).astype(np.int64)


def validate_alignment(features, raw, pid, keep):
    if len(features['row_offset']) != keep:
        raise ValueError('Retained row count mismatch')
    if not np.all(features['part_id'] == pid):
        raise ValueError('Wrong physical part ID')
    if not np.array_equal(features['row_offset'], np.arange(keep)):
        raise ValueError('Physical row identity mismatch, duplicate or reordered rows')
    for name in ['user_id', 'seq_len']:
        if not np.array_equal(features[name], raw[name][:keep]):
            raise ValueError(f'Raw-to-feature {name} mismatch')
    length = features['effective_length']
    if np.any(length < 0) or np.any(length > np.minimum(features['seq_len'], 100)):
        raise ValueError('Impossible effective history length')
    if not np.array_equal(features['c1_known_count'] + features['c1_unknown_count'], length):
        raise ValueError('Known/unknown category counts do not sum to visible history')
    if np.any(features['c1_ambiguous_count'] > features['c1_unknown_count']):
        raise ValueError('Ambiguous count exceeds unknown count')
    if not np.array_equal(np.isfinite(features['c1_entropy']), features['c1_known_count'] > 0):
        raise ValueError('Undefined entropy policy violated')


def independent_window_features(history_items, target_item, categories):
    valid = [int(item) for item in history_items if item > 0]
    observed = [int(categories[item]) for item in valid]
    known = [category for category in observed if category not in {0, 309}]
    counter = Counter(known)
    entropy = None if not known else -sum((n / len(known)) * math.log(n / len(known))
                                          for n in counter.values())
    target = int(categories[int(target_item)])
    matches = sum(category == target for category in observed) if target > 0 else 0
    return {'effective_length': len(valid), 'c1_known_count': len(known),
            'c1_unknown_count': len(valid) - len(known), 'c1_ambiguous_count': observed.count(309),
            'c1_unique_count': len(counter), 'c1_entropy': entropy,
            'target_c1_id': target, 'target_c1_known': target not in {0, 309},
            'target_c1_history_count': matches,
            'target_c1_history_fraction': matches / len(known) if known and target not in {0, 309} else None}


def independent_part3_check(dataset, feature_dir, sample_rows=32):
    filename = 'part-00003.parquet'
    paths = {'data': dataset / 'test/data' / filename,
             'users': dataset / 'test/user_info' / filename,
             'items': dataset / 'test/item_info' / filename,
             'features': feature_dir / filename}
    raw = next(pq.ParquetFile(paths['data']).iter_batches(
        batch_size=sample_rows, columns=['user_index', 'item_index', 'seq_len'])).to_pylist()
    saved = next(pq.ParquetFile(paths['features']).iter_batches(batch_size=sample_rows)).to_pylist()
    if len(raw) != sample_rows or len(saved) != sample_rows:
        raise ValueError('Not enough rows for independent feature audit')
    wanted_users = {row['user_index'] for row in raw}
    histories, user_rows_scanned = {}, 0
    for batch in pq.ParquetFile(paths['users']).iter_batches(
            batch_size=512, columns=['user_index', 'full_item_seq']):
        user_rows_scanned += len(batch)
        for row in batch.to_pylist():
            if row['user_index'] in wanted_users:
                histories[row['user_index']] = row['full_item_seq']
        if wanted_users <= histories.keys():
            break
    if wanted_users - histories.keys():
        raise ValueError('Independent audit could not resolve selected users')
    windows, wanted_items = [], {0}
    for row in raw:
        seq = histories[row['user_index']]
        anchor = row['seq_len']
        if seq[anchor] != row['item_index']:
            raise ValueError('Independent target anchor mismatch')
        window = seq[max(0, anchor - 100):anchor]
        windows.append(window)
        wanted_items.update(window)
        wanted_items.add(row['item_index'])
    wanted_array = np.array(sorted(wanted_items), dtype=np.int64)
    categories, item_rows_scanned = {}, 0
    for batch in pq.ParquetFile(paths['items']).iter_batches(
            batch_size=65536, columns=['item_index', 'c1_id']):
        item_rows_scanned += len(batch)
        ids = batch.column('item_index').to_numpy()
        cats = batch.column('c1_id').to_numpy()
        mask = np.isin(ids, wanted_array)
        categories.update((int(i), int(c)) for i, c in zip(ids[mask], cats[mask]))
        if wanted_items <= categories.keys():
            break
    if wanted_items - categories.keys():
        raise ValueError('Independent audit could not resolve history categories')
    checks = []
    for offset, (row, actual, history) in enumerate(zip(raw, saved, windows)):
        expected = independent_window_features(history, row['item_index'], categories)
        if actual['part_id'] != 3 or actual['row_offset'] != offset:
            raise ValueError('Independent sample row identity mismatch')
        for name, value in expected.items():
            observed = actual[name]
            if value is None:
                matched = observed is None
            elif isinstance(value, float):
                matched = observed is not None and math.isclose(value, observed, rel_tol=1e-6, abs_tol=1e-6)
            else:
                matched = value == observed
            if not matched:
                raise ValueError(f'Independent feature mismatch at part3 row{offset} {name}: {observed} != {value}')
        checks.append({'part_id': 3, 'row_offset': offset, 'seq_len': row['seq_len'], **expected})
    return {'status': 'passed', 'sample_rows': sample_rows,
            'method': 'First32 physical rows, independent Python list slicing/Counter entropy; no call to frozen vectorized feature builder.',
            'user_side_rows_scanned': user_rows_scanned, 'item_side_rows_scanned': item_rows_scanned,
            'checks': checks, 'sources': [file_identity(p) for p in paths.values()],
            'limit': 'Checks these rows and fields, not historical byte identity or all feature values.'}


def read_columns(path, names):
    table = pq.read_table(path, columns=names)
    return {name: table.column(name).to_numpy(zero_copy_only=False) for name in names}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'Empty output {path}')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'datasets/MerRec_Action')
    parser.add_argument('--feature-root', type=Path, default=ROOT / 'artifacts/rise_sample_analysis/group_features')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'experiments/rise_sample_analysis/results')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    pa.set_cpu_count(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / 'population_profile_summary.json'
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f'{summary_path} already exists')
    sources = [file_identity(Path(__file__), True)]
    frequency_dir, feature_dir = args.feature_root / 'frequencies', args.feature_root / 'features'
    frequency_file = frequency_dir / 'manifest.json'
    feature_file = feature_dir / 'manifest_test_all.json'
    fm, manifest = [json.loads(path.read_text()) for path in [frequency_file, feature_file]]
    metadata_file = args.dataset_dir / 'meta_data.json'
    metadata = json.loads(metadata_file.read_text())
    if tuple(metadata['label']) != LABELS:
        raise ValueError('Unexpected MerRec labels')
    if not fm['complete'] or fm['metadata_sha256'] != digest(metadata_file):
        raise ValueError('Incomplete or inconsistent frequency manifest')
    if manifest['frequency_manifest_sha256'] != digest(frequency_file):
        raise ValueError('Feature frequency provenance mismatch')
    if manifest['metadata_sha256'] != fm['metadata_sha256'] or manifest['limit_rows'] is not None:
        raise ValueError('Incomplete or inconsistent feature manifest')
    if manifest['split'] != 'test' or manifest['max_len'] != 100 or manifest['retained_batch_size'] != 16384:
        raise ValueError('This formal profile requires the frozen MerRec test protocol')
    for path in [frequency_file, feature_file, metadata_file]:
        sources.append(file_identity(path, True))
    for source in fm['sources']:
        verify_source_identity(source)
    independent_audit = independent_part3_check(args.dataset_dir, feature_dir)
    print(json.dumps({'independent_part3_audit': 'passed', 'rows_checked': 32}), flush=True)
    train, frequencies, cuts, train_distribution = {}, {}, {}, []
    for entity in ('user', 'item'):
        path = frequency_dir / f'{entity}_train_target_count.npy'
        identity = file_identity(path, True)
        if identity['sha256'] != fm['arrays'][path.name]:
            raise ValueError(f'Frequency array content changed: {entity}')
        sources.append(identity)
        values = np.load(path, mmap_mode='r')
        if values[0] != 0 or int(values.sum()) != fm['train_rows_counted']:
            raise ValueError('Frequency padding or total count failure')
        train[entity], unique, counts = frequency_profile(values)
        frequencies[entity] = values
        cuts[entity] = (train[entity]['q25'], train[entity]['q75'])
        train_distribution += [{'entity': entity, 'train_target_frequency': int(v), 'entities': int(n)}
                               for v, n in zip(unique, counts)]
    seen = {entity: np.zeros(len(values), dtype=bool) for entity, values in frequencies.items()}
    groups, part_rows, raw_history = {}, [], np.zeros(1, np.int64)
    output_mtime_discrepancies = []
    totals = {'rows': 0, 'physical_test_rows': 0, 'history_valid_tokens': 0,
              'history_unknown_tokens': 0, 'history_ambiguous309_tokens': 0,
              'rows_any_unknown_history_category': 0, 'rows_any_ambiguous309_history_category': 0,
              'rows_nonempty_history_without_known_category': 0, 'target_unknown_category_rows': 0,
              'target_zero_category_rows': 0, 'target_ambiguous309_category_rows': 0,
              'rows_effective_length_not_min_raw100': 0, 'raw_history_sum': 0}
    positive = np.zeros(len(LABELS), np.int64)
    label_digest, key_digest = hashlib.sha256(), hashlib.sha256()
    pids = [int(p['part_id']) for p in manifest['parts']]
    if pids != list(range(8)):
        raise ValueError('Expected exactly the eight frozen MerRec test parts 0..7')
    for part in manifest['parts']:
        pid = int(part['part_id'])
        feature_path = feature_dir / f'part-{pid:05d}.parquet'
        raw_path = args.dataset_dir / 'test/data' / feature_path.name
        discrepancy = verify_source_identity(part['output'], allow_output_mtime_drift=True)
        if discrepancy is not None:
            output_mtime_discrepancies.append(discrepancy)
        for source in part['sources']:
            verify_source_identity(source)
        sources.extend([file_identity(feature_path, True), file_identity(raw_path, True)])
        raw_rows = pq.ParquetFile(raw_path).metadata.num_rows
        keep = raw_rows // 16384 * 16384
        if keep != part['rows_written'] or keep != part['retained_available_rows']:
            raise ValueError('Manifest and physical drop_last coverage disagree')
        features = read_columns(feature_path, FEATURE_COLUMNS)
        raw = read_columns(raw_path, ['user_id', 'seq_len', *LABELS])
        validate_alignment(features, raw, pid, keep)
        labels = np.column_stack([raw[name][:keep] for name in LABELS])
        if not np.all((labels == 0) | (labels == 1)):
            raise ValueError('Nonbinary or missing label')
        for entity in ['user', 'item']:
            ids = features[f'{entity}_id']
            if np.any(ids <= 0) or np.any(ids >= len(frequencies[entity])):
                raise ValueError('Invalid global entity ID')
            if not np.array_equal(features[f'{entity}_train_target_count'], frequencies[entity][ids]):
                raise ValueError('Feature target frequencies disagree with source arrays')
            seen[entity][ids] = True
        keys = np.column_stack([features['part_id'], features['row_offset']]).astype('<i8')
        part_key_hash = hashlib.sha256(keys.tobytes()).hexdigest()
        if part_key_hash != part['row_key_sha256']:
            raise ValueError('Row-key checksum mismatch')
        key_digest.update(keys.tobytes())
        label_bytes = labels.astype('<f4').tobytes()
        label_digest.update(label_bytes)
        positive += labels.sum(axis=0).astype(np.int64)
        update_groups(groups, features, labels, cuts)
        hist = np.bincount(features['seq_len'].astype(np.int64))
        if len(hist) > len(raw_history):
            raw_history = np.pad(raw_history, (0, len(hist) - len(raw_history)))
        raw_history[:len(hist)] += hist
        length, unknown, ambiguous = [features[name] for name in
                                     ['effective_length', 'c1_unknown_count', 'c1_ambiguous_count']]
        totals['rows'] += keep
        totals['physical_test_rows'] += raw_rows
        totals['history_valid_tokens'] += int(length.sum())
        totals['history_unknown_tokens'] += int(unknown.sum())
        totals['history_ambiguous309_tokens'] += int(ambiguous.sum())
        totals['rows_any_unknown_history_category'] += int((unknown > 0).sum())
        totals['rows_any_ambiguous309_history_category'] += int((ambiguous > 0).sum())
        totals['rows_nonempty_history_without_known_category'] += int(((length > 0) & (features['c1_known_count'] == 0)).sum())
        totals['target_unknown_category_rows'] += int((~features['target_c1_known']).sum())
        totals['target_zero_category_rows'] += int((features['target_c1_id'] == 0).sum())
        totals['target_ambiguous309_category_rows'] += int((features['target_c1_id'] == 309).sum())
        totals['rows_effective_length_not_min_raw100'] += int((length != np.minimum(features['seq_len'], 100)).sum())
        totals['raw_history_sum'] += int(features['seq_len'].sum())
        part_rows.append({'part_id': pid, 'physical_rows': raw_rows, 'retained_rows': keep,
                          'dropped_tail_rows': raw_rows - keep, 'row_offset_start': 0,
                          'row_offset_end_inclusive': keep - 1, 'row_key_sha256': part_key_hash,
                          'label_sha256': hashlib.sha256(label_bytes).hexdigest()})
        print(json.dumps({'part_id': pid, 'rows_checked': keep, 'cumulative_rows': totals['rows']}), flush=True)
    if totals['rows'] != manifest['rows_written'] or totals['rows'] != 11599872:
        raise ValueError('Formal retained total mismatch')
    if fm['train_rows_counted'] != 144108235 or fm['train_rows_footer'] != 144108235:
        raise ValueError('Formal train target total mismatch')
    test_entities = {}
    for entity in ['user', 'item']:
        mask = seen[entity]
        test_entities[entity] = {'distinct_retained_test_entities': int(mask.sum()),
                                 'distinct_test_entities_seen_in_train_targets': int((mask & (frequencies[entity] > 0)).sum()),
                                 'distinct_test_entities_unseen_in_train_targets': int((mask & (frequencies[entity] == 0)).sum())}
    long_rows, wide_rows = [], []
    for (dimension, group_id, name), state in groups.items():
        n = state['rows']
        wide = {'dimension': dimension, 'group_id': group_id, 'group': name, 'rows': n}
        for label, pos in zip(LABELS, state['positive']):
            rate = float(pos / n) if n else None
            long_rows.append({'dimension': dimension, 'group_id': group_id, 'group': name,
                              'label': label, 'rows': n, 'positive': int(pos),
                              'negative': n - int(pos), 'positive_rate': rate})
            wide[f'{label}_positive'] = int(pos)
            wide[f'{label}_positive_rate'] = rate
        wide_rows.append(wide)
    raw_values = np.flatnonzero(raw_history)
    probabilities = [0, .01, .1, .25, .5, .75, .9, .95, .99, 1]
    quantiles = weighted_quantile(raw_values, raw_history[raw_values], probabilities)
    category_rates = {key + '_rate': value / (totals['history_valid_tokens'] if key.endswith('_tokens') else totals['rows'])
                      for key, value in totals.items() if key.startswith(('history_unknown', 'history_ambiguous', 'rows_any_', 'rows_nonempty_', 'target_'))}
    summary = {'status': 'completed_descriptive_population_profile_no_prediction_or_gain_inference',
               'generated_at_utc': datetime.now(timezone.utc).isoformat(),
               'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'dataset': 'MerRec_Action',
               'train_target_population': train, 'test_entity_population': test_entities,
               'frequency_cuts': {k: {'q25': v[0], 'q75': v[1], 'method': 'inverted_cdf'} for k, v in cuts.items()},
               'totals': totals, 'category_rates': category_rates,
               'raw_history': {'quantile_method': 'inverted_cdf', 'quantiles': {str(p): int(q) for p, q in zip(probabilities, quantiles)},
                               'mean': totals['raw_history_sum'] / totals['rows']},
               'labels': {label: {'positive': int(pos), 'negative': totals['rows'] - int(pos),
                                   'positive_rate': float(pos / totals['rows'])} for label, pos in zip(LABELS, positive)},
               'row_key_sha256': key_digest.hexdigest(), 'labels_sha256': label_digest.hexdigest(),
               'category_policy': manifest['category_policy'],
               'producer_output_mtime_discrepancies': output_mtime_discrepancies,
               'group_boundaries': {'effective_length': ['0', '1-20', '21-50', '51-99', '100'],
                  'frequency': 'zero; positive <= q25; q25 < positive <= q75; positive > q75. Tied quantiles may yield empty groups.',
                  'entropy_at_length100': ['undefined', 'H<=1e-7', '1e-7<H<=ln(2)', 'ln(2)<H<=ln(5)', 'H>ln(5)'],
                  'candidate_category_support_at_length100': ['undefined', '0', '0<Q<=0.25', '0.25<Q<=0.75', 'Q>0.75']},
               'interpretation': ['Test statistics describe the retained 11,599,872 rows, not the full 11,681,154-row split.',
                                  'Observed distinct global IDs are counted from nonzero frequencies or actual retained targets, never vocabulary sizes.',
                                  'Category unknown includes definite code 0 and conservatively excluded ambiguous code 309.',
                                  'These are population properties only; no RISE effects, AUCs, causal explanations or action-time leakage rates are estimated.']}
    outputs = [('population_profile_groups.csv', long_rows), ('population_profile_groups_wide.csv', wide_rows),
               ('population_profile_parts.csv', part_rows), ('population_profile_train_frequency_distribution.csv', train_distribution),
               ('population_profile_raw_history.csv', [{'seq_len': int(v), 'rows': int(raw_history[v])} for v in raw_values]),
               ('population_profile_sources.csv', sources)]
    for name, rows in outputs:
        write_csv(args.output_dir / name, rows)
    summary['independent_part3_feature_audit'] = {'status': independent_audit['status'], 'rows_checked': independent_audit['sample_rows'], 'details': 'population_profile_part3_independent_check.json'}
    (args.output_dir / 'population_profile_part3_independent_check.json').write_text(json.dumps(independent_audit, indent=2) + '\n')
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': 'complete', 'rows': totals['rows'], 'frequency_cuts': summary['frequency_cuts']}), flush=True)


if __name__ == '__main__':
    main()
