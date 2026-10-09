"""Exact user-weighted tie-aware ROC AUC and prespecified interval summaries."""
import sys
sys.dont_write_bytecode = True
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class GroupedAUC:
    positive: object
    negative: object
    n_groups: int
    n_users: int
    representation: str

    def evaluate(self, user_weights_transposed):
        """Columns are draws; every row of one user receives that user's weight."""
        w = np.asarray(user_weights_transposed, dtype=np.float64)
        if w.ndim == 1:
            w = w[:, None]
        if w.ndim != 2 or w.shape[0] != self.n_users or not np.isfinite(w).all() or (w < 0).any():
            raise ValueError('Invalid nonnegative user multiplicities')
        positive = np.asarray(self.positive @ w)
        negative = np.asarray(self.negative @ w)
        total_p, total_n = positive.sum(axis=0), negative.sum(axis=0)
        # Ascending score groups: lower-scored negatives count fully, ties half.
        favorable = (positive * (np.cumsum(negative, axis=0) - .5 * negative)).sum(axis=0)
        denominator = total_p * total_n
        auc = np.divide(favorable, denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)
        return auc, total_p, total_n


def prepare_auc(labels, probabilities, user_index, n_users, representation='auto'):
    y, p, users = np.asarray(labels), np.asarray(probabilities, dtype=np.float32), np.asarray(user_index)
    if y.ndim != 1 or y.shape != p.shape or y.shape != users.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Aligned binary labels, predictions, and user indices required')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or users.dtype.kind not in 'iu' or users.min() < 0 or users.max() >= n_users:
        raise ValueError('Invalid probabilities/user indices')
    if representation not in ('auto', 'dense', 'sparse'):
        raise ValueError(representation)
    unique, groups = np.unique(p, return_inverse=True)
    shape = (len(unique), n_users)
    positive = csr_matrix((np.ones(int((y == 1).sum())), (groups[y == 1], users[y == 1])), shape=shape, dtype=np.float64)
    negative = csr_matrix((np.ones(int((y == 0).sum())), (groups[y == 0], users[y == 0])), shape=shape, dtype=np.float64)
    if representation == 'auto':
        representation = 'dense' if shape[0] * shape[1] <= 2_000_000 else 'sparse'
    if representation == 'dense':
        positive, negative = positive.toarray(), negative.toarray()
    return GroupedAUC(positive, negative, shape[0], n_users, representation)


def summarize_interval(draws, point):
    values = np.asarray(draws, dtype=np.float64)
    invalid = int((~np.isfinite(values)).sum())
    record = {'point_difference_full_minus_rms': float(point), 'draws': len(values),
              'invalid_draws': invalid, 'valid_draws': len(values) - invalid,
              'p_value': None, 'p_value_method': 'No p-values; interval-based exploratory comparison'}
    if invalid:
        record.update(ci95=None, ci99_375_bonferroni8=None, adjusted_direction='not_evaluable_missing_class_draws',
                      interval_policy='Withheld: no conditioning on valid draws and no resampling until valid')
    else:
        nominal = np.quantile(values, [.025, .975], method='linear')
        adjusted = np.quantile(values, [.003125, .996875], method='linear')
        direction = 'Full_higher_AUC' if adjusted[0] > 0 else 'RMS_higher_AUC' if adjusted[1] < 0 else 'unresolved'
        record.update(ci95=nominal.tolist(), ci99_375_bonferroni8=adjusted.tolist(), adjusted_direction=direction,
                      interval_policy='All prespecified draws retained; percentile intervals, no centering or studentization')
    return record


def draw_user_weights(n_users, n_draws, seed, batch_size):
    rng = np.random.default_rng(seed)
    probability = np.full(n_users, 1 / n_users)
    for start in range(0, n_draws, batch_size):
        count = min(batch_size, n_draws - start)
        yield start, rng.multinomial(n_users, probability, size=count)
