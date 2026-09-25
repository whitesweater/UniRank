"""Build MerRec sample-group features without model inference or training.

Only train/data target rows define frequencies. Histories are sliced exactly as
pre-padded UniRank inputs, excluding the target and all later view anchors.
The eventual-outcome action encoding is explicitly NOT claimed point-in-time safe.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import mmap
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / 'datasets/MerRec_Action'
DEFAULT_OUTPUT = ROOT / 'artifacts/rise_sample_analysis/group_features'
SCHEMA_VERSION = 1


def fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, content):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def parts(dataset, split, kind='data'):
    paths = sorted((dataset / split / kind).glob('part-*.parquet'))
    if not paths:
        raise ValueError(f'No {split}/{kind} parquet parts under {dataset}')
    return paths


def part_id(path):
    match = re.fullmatch(r'part-(\d+)\.parquet', Path(path).name)
    if not match:
        raise ValueError(f'Unsupported part filename: {path}')
    return int(match.group(1))


def numpy_column(table, name):
    column = table.column(name)
    if column.null_count:
        raise ValueError(f'Nulls in required column {name}')
    return column.to_numpy(zero_copy_only=False)


def dense_item_lookup(path, columns):
    table = pq.read_table(path, columns=['item_index', *columns])
    index = numpy_column(table, 'item_index').astype(np.int64, copy=False)
    if not len(index) or index.min() < 0:
        raise ValueError(f'Invalid item index in {path}')
    if len(np.unique(index)) != len(index):
        raise ValueError(f'Duplicate item index in {path}')
    size = int(index.max()) + 1
    if size > max(1024, 2 * len(index)):
        raise ValueError(f'Unexpectedly sparse local item index in {path}')
    result = {}
    for name in columns:
        values = numpy_column(table, name)
        if values.min() < 0 or values.max() > np.iinfo(np.int32).max:
            raise ValueError(f'Invalid {name} ID in {path}')
        target = np.full(size, -1, dtype=np.int32)
        target[index] = values
        result[name] = target
    return result


def validate_ids(values, size, name, allow_zero=True):
    if len(values) and (values.min() < (0 if allow_zero else 1) or values.max() >= size):
        raise ValueError(f'{name} outside valid ID range [0, {size})')


def add_counts(destination, values):
    """Bounded-memory bincount for compact IDs; unique counts for wide spans."""
    values = np.asarray(values, dtype=np.int64)
    validate_ids(values, len(destination), 'frequency')
    if not len(values):
        return
    lo, hi = int(values.min()), int(values.max())
    if hi - lo + 1 <= max(4096, 4 * len(values)):
        counts = np.bincount(values - lo, minlength=hi - lo + 1)
        destination[lo:hi + 1] += counts.astype(np.uint64)
    else:
        keys, counts = np.unique(values, return_counts=True)
        destination[keys] += counts.astype(np.uint64)


def count_train(args, metadata):
    directory = args.output_dir / 'frequencies'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'manifest.json').exists() and not args.overwrite:
        raise FileExistsError(f'{directory}/manifest.json exists; choose another output or --overwrite')
    user_path = directory / 'user_train_target_count.npy'
    item_path = directory / 'item_train_target_count.npy'
    user_tmp, item_tmp = user_path.with_suffix('.npy.tmp'), item_path.with_suffix('.npy.tmp')
    users = np.lib.format.open_memmap(user_tmp, mode='w+', dtype=np.uint64,
                                     shape=(metadata['vocab_size']['user_id'],))
    items = np.lib.format.open_memmap(item_tmp, mode='w+', dtype=np.uint64,
                                     shape=(metadata['vocab_size']['item_id'],))
    users[:] = 0
    items[:] = 0
    sources, rows, expected = [], 0, 0
    files = parts(args.dataset_dir, 'train')
    for path in files:
        expected += pq.ParquetFile(path).metadata.num_rows
    for path in files:
        side_path = args.dataset_dir / 'train/item_info' / path.name
        lookup = dense_item_lookup(side_path, ['item_id'])['item_id']
        sources.extend([fingerprint(path), fingerprint(side_path)])
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=args.batch_size, columns=['user_id', 'item_index']):
            remaining = None if args.limit_rows is None else args.limit_rows - rows
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None:
                batch = batch.slice(0, min(len(batch), remaining))
            user_ids = numpy_column(batch, 'user_id')
            local_ids = numpy_column(batch, 'item_index')
            validate_ids(local_ids, len(lookup), 'local item', allow_zero=False)
            global_ids = lookup[local_ids]
            validate_ids(user_ids, len(users), 'user', allow_zero=False)
            validate_ids(global_ids, len(items), 'global item', allow_zero=False)
            add_counts(users, user_ids)
            add_counts(items, global_ids)
            rows += len(batch)
        print(json.dumps({'stage': 'count-train', 'part_id': part_id(path),
                          'train_rows_counted': rows}), flush=True)
        del lookup
        gc.collect()
        if args.limit_rows is not None and rows >= args.limit_rows:
            break
    if int(users.sum()) != rows or int(items.sum()) != rows:
        raise AssertionError('Frequency counts do not reconcile to train target rows')
    users.flush()
    items.flush()
    del users, items
    user_tmp.replace(user_path)
    item_tmp.replace(item_path)
    write_json(directory / 'manifest.json', {
        'schema_version': SCHEMA_VERSION, 'generated_at': datetime.now(timezone.utc).isoformat(),
        'dataset_dir': str(args.dataset_dir.resolve()),
        'metadata_sha256': sha256(args.dataset_dir / 'meta_data.json'),
        'script_sha256': sha256(Path(__file__)), 'train_rows_counted': rows,
        'train_rows_footer': expected, 'complete': rows == expected,
        'definition': 'Counts of target rows in frozen preprocessed train/data, before drop_last; excludes all side-info history occurrences.',
        'sources': sources, 'source_identity_level': 'path_size_mtime; not full parquet content hashes',
        'arrays': {user_path.name: sha256(user_path), item_path.name: sha256(item_path)},
    })


def extract_small_vocab(path, key):
    """Read one small top-level vocabulary without parsing multi-GB identity maps."""
    with Path(path).open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        start = data.rfind(b'\n    "' + key.encode() + b'":')
        if start < 0:
            # Small fixtures or compact files may not use the archived formatting.
            if len(data) <= 1024 * 1024:
                return json.loads(data[:])[key]
            raise ValueError(f'Cannot locate a bounded {key} object in {path}')
        start = data.find(b'{', start)
        end = data.find(b'\n    }', start)
        if end < 0 or end - start > 1024 * 1024:
            raise ValueError(f'{key} vocabulary unexpectedly large or malformed')
        return json.loads(data[start:end + 6])


def category_policy(dataset, metadata):
    """Zero is documented unknown; legacy positive OOV placeholders are ambiguous."""
    unknown, ambiguous = {0}, set()
    legacy = dataset / 'feature_vocab.json'
    evidence = {'zero': 'MerRec preprocessing excludes __MISSING__ and maps missing/rare feature values to 0.'}
    if legacy.exists():
        vocab = extract_small_vocab(legacy, 'c1_id')
        for key in ['__OOV__', '__MISSING__', '__PAD__']:
            if key in vocab and int(vocab[key]) > 0:
                ambiguous.add(int(vocab[key]))
        evidence['legacy_c1_vocab'] = {'source': fingerprint(legacy),
                                      'canonical_object_sha256': hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest(),
                                      'reserved_tokens': {k: int(v) for k, v in vocab.items() if k.startswith('__')}}
        evidence['conflict'] = ('Legacy identity vocabulary reserves its last ID as OOV, while author preprocessing encodes non-missing categories through vocab_size-1. It is not an original raw-to-encoded vocabulary. Positive reserved IDs are excluded conservatively, not proven truly missing.')
    unknown.update(ambiguous)
    validate_ids(np.array(sorted(unknown)), metadata['vocab_size']['c1_id'], 'category policy')
    return {'unknown_ids': sorted(unknown), 'ambiguous_ids': sorted(ambiguous),
            'policy': 'conservative exclusion; retain ambiguous counts for sensitivity analysis',
            'evidence': evidence}


def action_table(metadata):
    labels = metadata['label']
    vocab = metadata['action_vocab']
    table = np.zeros((max(vocab.values()) + 1, len(labels)), dtype=np.int16)
    for name, code in vocab.items():
        tokens = set(name.split('|'))
        unknown = tokens - set(labels) - {'exposure'}
        if unknown:
            raise ValueError(f'Unknown action vocabulary tokens: {unknown}')
        for index, label in enumerate(labels):
            table[code, index] = int(label in tokens)
    return table, labels


def load_sequences(path):
    table = pq.read_table(path, columns=['user_index', 'full_item_seq', 'full_action_seq'])
    users = numpy_column(table, 'user_index').astype(np.int64, copy=False)
    if users.min() < 0 or len(np.unique(users)) != len(users):
        raise ValueError('Invalid or duplicate local user indices')
    lookup = np.full(int(users.max()) + 1, -1, dtype=np.int64)
    lookup[users] = np.arange(len(users))
    item_lists = table.column('full_item_seq').combine_chunks()
    action_lists = table.column('full_action_seq').combine_chunks()
    if item_lists.null_count or action_lists.null_count:
        raise ValueError('Null history list')
    offsets = item_lists.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    if not np.array_equal(offsets, action_lists.offsets.to_numpy(zero_copy_only=False)):
        raise ValueError('Item/action sequence lengths do not match')
    if item_lists.values.null_count or action_lists.values.null_count:
        raise ValueError('Null history token')
    return {'offsets': offsets, 'items': item_lists.values.to_numpy(zero_copy_only=False),
            'actions': action_lists.values.to_numpy(zero_copy_only=False), 'user_lookup': lookup}


def gather_windows(sequences, users, lengths, targets, max_len):
    users, lengths = np.asarray(users, np.int64), np.asarray(lengths, np.int64)
    validate_ids(users, len(sequences['user_lookup']), 'local user')
    row = sequences['user_lookup'][users]
    if np.any(row < 0):
        raise ValueError('Local user absent in side information')
    offsets = sequences['offsets']
    sizes = offsets[row + 1] - offsets[row]
    if np.any(lengths < 0) or np.any(lengths >= sizes):
        raise ValueError('seq_len does not identify a target within the full sequence')
    anchors = offsets[row] + lengths
    if not np.array_equal(sequences['items'][anchors], targets):
        raise ValueError('History target-anchor alignment mismatch')
    positions = lengths[:, None] - max_len + np.arange(max_len)[None, :]
    valid = positions >= 0
    indices = offsets[row, None] + np.maximum(positions, 0)
    items = np.where(valid, sequences['items'][indices], 0)
    actions = np.where(valid, sequences['actions'][indices], 0)
    return items, actions


def compute_history_features(history_items, history_actions, categories, target_categories,
                             action_lookup, labels, category_vocab_size,
                             unknown_ids, ambiguous_ids):
    rows, width = history_items.shape
    valid = history_items > 0
    validate_ids(history_items.ravel(), len(categories), 'history local item')
    cats = categories[history_items]
    if np.any(cats[valid] < 0):
        raise ValueError('History references an item absent from side information')
    validate_ids(cats[valid], category_vocab_size, 'history category')
    validate_ids(target_categories, category_vocab_size, 'target category')
    known = valid & ~np.isin(cats, list(unknown_ids))
    effective = valid.sum(axis=1).astype(np.int16)
    counts_known = known.sum(axis=1).astype(np.int16)
    ambiguous = (valid & np.isin(cats, list(ambiguous_ids))).sum(axis=1).astype(np.int16)
    indices = np.broadcast_to(np.arange(rows)[:, None], (rows, width))[known]
    flattened = indices * category_vocab_size + cats[known]
    counts = np.bincount(flattened, minlength=rows * category_vocab_size).reshape(rows, category_vocab_size)
    positive = counts > 0
    terms = np.zeros_like(counts, dtype=np.float64)
    terms[positive] = counts[positive] * np.log(counts[positive])
    entropy = np.full(rows, np.nan, dtype=np.float32)
    nonempty = counts_known > 0
    entropy[nonempty] = (np.log(counts_known[nonempty].astype(np.float64)) -
                         terms.sum(axis=1)[nonempty] / counts_known[nonempty])
    target_known = ~np.isin(target_categories, list(unknown_ids))
    # Retain positive ambiguous target matches so category-309 sensitivity is recoverable.
    target_matches = (valid & (cats == target_categories[:, None]) &
                      (target_categories[:, None] > 0)).sum(axis=1).astype(np.int16)
    fractions = np.full(rows, np.nan, dtype=np.float32)
    available = nonempty & target_known
    fractions[available] = target_matches[available] / counts_known[available]
    validate_ids(history_actions.ravel(), len(action_lookup), 'action token')
    result = {'effective_length': effective,
              'padding_fraction': (1 - effective / width).astype(np.float32),
              'c1_known_count': counts_known,
              'c1_unknown_count': (effective - counts_known).astype(np.int16),
              'c1_ambiguous_count': ambiguous,
              'c1_unique_count': positive.sum(axis=1).astype(np.int16),
              'c1_entropy': entropy, 'target_c1_id': target_categories.astype(np.int32),
              'target_c1_known': target_known, 'target_c1_history_count': target_matches,
              'target_c1_history_fraction': fractions,
              'history_action_unknown_count': (valid & (history_actions == 0)).sum(axis=1).astype(np.int16),
              'history_action_temporal_status': np.where(effective > 0, 'unverified_eventual_outcomes', 'no_history')}
    for index, label in enumerate(labels):
        result[f'history_{label}_count'] = (action_lookup[history_actions, index] * valid).sum(axis=1).astype(np.int16)
    return result


def read_row_ids(path):
    files = sorted(path.glob('*.parquet')) if path.is_dir() else [path]
    if not files:
        raise ValueError('Empty row-ID input')
    pieces = {}
    for source in files:
        for batch in pq.ParquetFile(source).iter_batches(columns=['part_id', 'row_offset']):
            pids, offsets = numpy_column(batch, 'part_id'), numpy_column(batch, 'row_offset')
            if len(offsets) and (offsets.min() < 0 or pids.min() < 0):
                raise ValueError('Negative row identity')
            for pid in np.unique(pids):
                pieces.setdefault(int(pid), []).append(offsets[pids == pid].astype(np.int64))
    result = {}
    for pid, arrays in pieces.items():
        offsets = np.sort(np.concatenate(arrays))
        if len(offsets) > 1 and np.any(offsets[1:] == offsets[:-1]):
            raise ValueError(f'Duplicate row IDs in part {pid}')
        result[pid] = offsets
    return result


def feature_table(columns):
    result = {}
    for name, values in columns.items():
        if np.issubdtype(np.asarray(values).dtype, np.floating):
            result[name] = pa.array(values, mask=np.isnan(values))
        else:
            result[name] = pa.array(values)
    return pa.table(result)


def build_features(args, metadata):
    frequency_dir = args.frequency_dir or args.output_dir / 'frequencies'
    frequency_manifest = json.loads((frequency_dir / 'manifest.json').read_text())
    if frequency_manifest['metadata_sha256'] != sha256(args.dataset_dir / 'meta_data.json'):
        raise ValueError('Frequency metadata fingerprint does not match current dataset')
    if not frequency_manifest['complete'] and not args.allow_incomplete_frequencies:
        raise ValueError('Frequency arrays are incomplete; only a labeled smoke test may override')
    for source in frequency_manifest['sources']:
        if fingerprint(source['path']) != source:
            raise ValueError(f'Frequency source changed: {source["path"]}')
    for name, expected_digest in frequency_manifest['arrays'].items():
        if sha256(frequency_dir / name) != expected_digest:
            raise ValueError(f'Frequency array content changed: {name}')
    users = np.load(frequency_dir / 'user_train_target_count.npy', mmap_mode='r')
    items = np.load(frequency_dir / 'item_train_target_count.npy', mmap_mode='r')
    output = args.output_dir / ('features' if args.split == 'test' else f'features_{args.split}')
    output.mkdir(parents=True, exist_ok=True)
    policy = category_policy(args.dataset_dir, metadata)
    action_lookup, labels = action_table(metadata)
    selection = None if args.row_ids is None else read_row_ids(args.row_ids)
    all_paths = parts(args.dataset_dir, args.split)
    available_pids = {part_id(path) for path in all_paths}
    if selection is not None and set(selection) - available_pids:
        raise ValueError('Row-ID selection refers to missing data parts')
    paths = [path for path in all_paths if (args.part_id is None or part_id(path) in args.part_id)
             and (selection is None or part_id(path) in selection)]
    if not paths:
        raise ValueError('No selected input parts')
    total_rows, manifests = 0, []
    for path in paths:
        pid = part_id(path)
        destination = output / path.name
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f'{destination} already exists')
        parquet = pq.ParquetFile(path)
        keep = parquet.metadata.num_rows
        if args.retained_batch_size:
            keep = keep // args.retained_batch_size * args.retained_batch_size
        selected = None if selection is None else selection[pid]
        if selected is not None and len(selected) and selected[-1] >= keep:
            raise ValueError(f'Selected row in part {pid} is outside retained coverage [0, {keep})')
        item_path = args.dataset_dir / args.split / 'item_info' / path.name
        user_path = args.dataset_dir / args.split / 'user_info' / path.name
        lookup = dense_item_lookup(item_path, ['item_id', 'c1_id'])
        sequences = load_sequences(user_path)
        temporary = destination.with_suffix('.parquet.tmp')
        writer, offset, written = None, 0, 0
        key_digest = hashlib.sha256()
        try:
            for batch in parquet.iter_batches(batch_size=args.batch_size,
                                              columns=['user_id', 'user_index', 'item_index', 'seq_len']):
                start, end = offset, min(offset + len(batch), keep)
                offset += len(batch)
                if start >= keep:
                    break
                indices = np.arange(start, end, dtype=np.int64)
                if selected is not None:
                    indices = selected[np.searchsorted(selected, start):np.searchsorted(selected, end)]
                if args.limit_rows is not None:
                    indices = indices[:max(0, args.limit_rows - total_rows)]
                if not len(indices):
                    if args.limit_rows is not None and total_rows >= args.limit_rows:
                        break
                    continue
                chosen = batch.take(pa.array(indices - start))
                global_users = numpy_column(chosen, 'user_id').astype(np.int64)
                local_users = numpy_column(chosen, 'user_index')
                local_items = numpy_column(chosen, 'item_index')
                lengths = numpy_column(chosen, 'seq_len').astype(np.int64)
                validate_ids(local_items, len(lookup['item_id']), 'target local item', allow_zero=False)
                global_items = lookup['item_id'][local_items].astype(np.int64)
                validate_ids(global_items, len(items), 'target global item', allow_zero=False)
                validate_ids(global_users, len(users), 'global user', allow_zero=False)
                history_items, history_actions = gather_windows(sequences, local_users, lengths,
                                                                 local_items, args.max_len)
                features = compute_history_features(
                    history_items, history_actions, lookup['c1_id'], lookup['c1_id'][local_items],
                    action_lookup, labels, metadata['vocab_size']['c1_id'],
                    set(policy['unknown_ids']), set(policy['ambiguous_ids']))
                columns = {'part_id': np.full(len(indices), pid, dtype=np.int32),
                           'row_offset': indices, 'user_id': global_users, 'item_id': global_items,
                           'seq_len': lengths, 'user_train_target_count': users[global_users],
                           'item_train_target_count': items[global_items], **features}
                table = feature_table(columns)
                if writer is None:
                    schema = table.schema.with_metadata({b'group_features_schema_version': str(SCHEMA_VERSION).encode()})
                    writer = pq.ParquetWriter(temporary, schema, compression='zstd', use_dictionary=True)
                writer.write_table(table)
                key_digest.update(np.column_stack([np.full(len(indices), pid), indices]).astype('<i8').tobytes())
                written += len(indices)
                total_rows += len(indices)
            if writer is not None:
                writer.close()
                writer = None
                temporary.replace(destination)
        finally:
            if writer is not None:
                writer.close()
        expected_selected = keep if selected is None else len(selected)
        if args.limit_rows is None and written != expected_selected:
            raise AssertionError(f'Feature coverage mismatch in part {pid}: {written} != {expected_selected}')
        manifest = {'part_id': pid, 'rows_written': written, 'retained_available_rows': keep,
                    'row_key_sha256': key_digest.hexdigest(),
                    'sources': [fingerprint(p) for p in [path, item_path, user_path]],
                    'output': None if not written else fingerprint(destination)}
        manifests.append(manifest)
        write_json(output / f'part-{pid:05d}.manifest.json', manifest)
        print(json.dumps({'stage': 'build', 'part_id': pid, 'rows_written': written,
                          'total_rows_written': total_rows}), flush=True)
        del lookup, sequences
        gc.collect()
        if args.limit_rows is not None and total_rows >= args.limit_rows:
            break
    suffix = 'all' if args.part_id is None else '_'.join(str(p) for p in sorted(args.part_id))
    write_json(output / f'manifest_{args.split}_{suffix}.json', {
        'schema_version': SCHEMA_VERSION, 'generated_at': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha256(Path(__file__)), 'dataset_dir': str(args.dataset_dir.resolve()),
        'metadata_sha256': sha256(args.dataset_dir / 'meta_data.json'),
        'split': args.split, 'max_len': args.max_len, 'padding': 'pre',
        'retained_batch_size': args.retained_batch_size, 'limit_rows': args.limit_rows,
        'row_ids': None if args.row_ids is None else str(args.row_ids.resolve()),
        'frequency_manifest_sha256': sha256(frequency_dir / 'manifest.json'),
        'category_policy': policy, 'rows_written': total_rows, 'parts': manifests,
        'action_temporal_status': 'Eventual outcomes of prior view anchors; original action timestamps absent. Point-in-time availability and affected prevalence are unverified.',
        'missing_entropy_and_fraction': 'Arrow null, never zero-imputed',
    })


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATA)
    result.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    subparsers = result.add_subparsers(dest='command', required=True)
    train = subparsers.add_parser('count-train', help='Stream only train targets into global-ID frequency arrays')
    train.add_argument('--batch-size', type=int, default=262144)
    train.add_argument('--limit-rows', type=int, default=None, help='Smoke only; incomplete arrays are marked')
    train.add_argument('--overwrite', action='store_true')
    build = subparsers.add_parser('build', help='Create sample-ID-preserving group-feature Parquet shards')
    build.add_argument('--frequency-dir', type=Path)
    build.add_argument('--split', choices=['test', 'valid', 'train'], default='test')
    build.add_argument('--part-id', type=int, nargs='+')
    build.add_argument('--row-ids', type=Path, help='Parquet file/directory containing unique part_id,row_offset')
    build.add_argument('--batch-size', type=int, default=4096)
    build.add_argument('--max-len', type=int, default=100)
    build.add_argument('--retained-batch-size', type=int, default=16384,
                       help='Per-block historical drop_last; 0 retains all rows')
    build.add_argument('--limit-rows', type=int, default=None, help='Limit rows written; manifest marks this explicitly')
    build.add_argument('--allow-incomplete-frequencies', action='store_true')
    build.add_argument('--overwrite', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.batch_size <= 0 or (args.limit_rows is not None and args.limit_rows <= 0):
        raise ValueError('Batch size and any row limit must be positive')
    if args.command == 'build' and (args.max_len <= 0 or args.max_len > np.iinfo(np.int16).max or
                                     args.retained_batch_size < 0):
        raise ValueError('Invalid window length or retained batch size')
    metadata = json.loads((args.dataset_dir / 'meta_data.json').read_text())
    if args.command == 'count-train':
        count_train(args, metadata)
    else:
        build_features(args, metadata)


if __name__ == '__main__':
    main()
