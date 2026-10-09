"""Read-only class-support/tie preflight; no AUC estimates or bootstrap draws."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ('RankMixer', 20262028, 'experiments/rise_discussion_validation_20260925/results_651654'),
    ('RankMixer', 20262029, 'experiments/rise_score_profile_repeats_20260926/rankmixer_kuairand_seed20262029/results'),
    ('HiFormer', 20262028, 'experiments/rise_discussion_validation_hiformer_20260925/results_651691'),
    ('HiFormer', 20262029, 'experiments/rise_score_profile_repeats_20260926/hiformer_taac25_seed20262029/results'),
]
LABELS = {'RankMixer': ['is_click', 'is_follow', 'is_like', 'is_comment', 'is_forward', 'long_view'],
          'HiFormer': ['is_click', 'is_conversion']}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def missing_support_probability(users, supporting_users):
    if supporting_users == users:
        return {'probability': 0.0, 'log_probability': None,
                'log10_probability': None, 'structurally_impossible': True}
    log_probability = users * math.log1p(-supporting_users / users)
    return {'probability': math.exp(log_probability), 'log_probability': log_probability,
            'log10_probability': log_probability / math.log(10), 'structurally_impossible': False}


def ties(scores, truth):
    values, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    positives = np.bincount(inverse, weights=truth, minlength=len(values)).astype(np.int64)
    negatives = counts - positives
    return {'distinct_scores': len(values), 'tied_score_groups': int(np.count_nonzero(counts > 1)),
            'rows_in_tied_groups': int(counts[counts > 1].sum()),
            'duplicate_rows_beyond_first_per_score': int(len(scores) - len(values)),
            'largest_tie_group': int(counts.max()),
            'cross_class_tied_pairs': int(np.dot(positives, negatives)),
            'cross_class_pair_count': int(truth.sum()) * int(len(truth) - truth.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cases, cohorts = [], {}
    for model, seed, relative in CASES:
        source = ROOT / relative
        complete = json.loads((source / 'complete.json').read_text())
        assert complete['completed'] and complete['all_gates_passed'], (model, seed)
        hashes = {}
        for name in ('predictions.csv', 'meta.csv', 'plan.json', 'gates.json', 'manifest.json'):
            hashes[name] = sha(source / name)
            assert hashes[name] == complete['files_sha256'][name], (model, seed, name)
        hashes['complete.json'] = sha(source / 'complete.json')
        frame = pd.read_csv(source / 'predictions.csv', float_precision='round_trip', low_memory=False)
        meta = pd.read_csv(source / 'meta.csv', low_memory=False)
        assert len(frame) == len(meta) == 131072
        assert np.array_equal(frame.row_index.to_numpy(), np.arange(len(frame)))
        metadata_fields = [field for field in ('row_index', 'part_id', 'row_offset', 'user_id', 'item_id',
                           'seq_len', 'effective_length', 'eligible', *LABELS[model]) if field in meta]
        for field in metadata_fields:
            assert np.array_equal(frame[field].to_numpy(), meta[field].to_numpy()), (model, seed, field)
        if 'eligible' in frame:
            assert frame.eligible.dtype == bool
            frame = frame.loc[frame.eligible].copy()
        identity_fields = ['row_index', 'part_id', 'row_offset', 'user_id', 'item_id', 'effective_length', *LABELS[model]]
        identity = np.ascontiguousarray(frame[identity_fields].to_numpy(dtype='<i8'))
        identity_sha = hashlib.sha256(identity.tobytes()).hexdigest()
        users, inverse = np.unique(frame.user_id.to_numpy(dtype=np.int64), return_inverse=True)
        row_counts = np.bincount(inverse)
        support = {}
        score_ties = {}
        for label in LABELS[model]:
            truth = frame[label].to_numpy(dtype=np.int64)
            assert np.isin(truth, [0, 1]).all()
            positive_by_user = np.bincount(inverse, weights=truth, minlength=len(users)).astype(np.int64)
            negative_by_user = row_counts - positive_by_user
            positives, negatives = int(truth.sum()), int(len(truth) - truth.sum())
            assert positives > 0 and negatives > 0
            positive_users = int(np.count_nonzero(positive_by_user))
            negative_users = int(np.count_nonzero(negative_by_user))
            missing_positive = missing_support_probability(len(users), positive_users)
            missing_negative = missing_support_probability(len(users), negative_users)
            support[label] = {'positive_rows': positives, 'negative_rows': negatives,
                'users_with_positive': positive_users, 'users_with_negative': negative_users,
                'users_with_both_classes': int(np.count_nonzero((positive_by_user > 0) & (negative_by_user > 0))),
                'miss_all_positive': missing_positive, 'miss_all_negative': missing_negative,
                'probability_undefined_auc_per_draw': missing_positive['probability'] + missing_negative['probability'],
                'expected_undefined_auc_draws_out_of_2000': 2000 * (missing_positive['probability'] + missing_negative['probability'])}
            score_ties[label] = {}
            for route in ('full', 'template_rms'):
                scores = frame[f'{label}_{route}'].to_numpy(dtype=np.float64)
                assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
                score_ties[label][route] = ties(scores, truth)
        cohort = {'rows': len(frame), 'users': len(users), 'identity_fields': identity_fields,
                  'ordered_eligible_identity_sha256': identity_sha, 'labels': support}
        if model in cohorts:
            assert cohorts[model] == cohort, (model, 'same-window/cohort/support mismatch across checkpoints')
        else:
            cohorts[model] = cohort
        cases.append({'model': model, 'checkpoint_seed': seed, 'source_dir': str(source),
                      'checked_input_sha256': hashes, 'same_cohort_as_other_seed': True,
                      'rows': len(frame), 'users': len(users), 'score_ties': score_ties})
    result = {'status': 'INPUT_CLASS_SUPPORT_AND_TIES_VERIFIED',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'formula': 'P(no positive support) = ((U - m_positive) / U)^U; analogous for negatives',
              'sampling_assumption': 'Draw exactly U user clusters uniformly with replacement; retain all rows of each selected cluster with multiplicity.',
              'undefined_probability': 'No-positive and no-negative events are disjoint because every selected user has at least one row; their probabilities add.',
              'probability_representation': 'Both natural-log and log10 values retained; null log denotes exact zero when every user supports the class.',
              'tie_definition': 'Exact equality of retained raw prediction values parsed with round_trip; no rounding or jitter.',
              'cohorts': cohorts, 'cases': cases, 'script_sha256': sha(__file__),
              'python_version': sys.version,
              'scope': 'Read-only input audit only. No bootstrap draws, AUC effects, confidence intervals, inference, training, or paper edits.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for model, cohort in cohorts.items():
        print(model, 'rows=', cohort['rows'], 'users=', cohort['users'], flush=True)
        for label, support in cohort['labels'].items():
            print(label, support['positive_rows'], support['negative_rows'],
                  support['users_with_positive'], support['users_with_negative'],
                  'P(undefined)=', support['probability_undefined_auc_per_draw'], flush=True)
    print('Wrote', args.output, flush=True)


if __name__ == '__main__':
    main()
