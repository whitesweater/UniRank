"""Independent paired statistics, matching the previous probe's declared formulae."""
from __future__ import annotations

import numpy as np


def bce(labels, predictions):
    y = np.asarray(labels)
    p = np.asarray(predictions, dtype=np.float64)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("BCE requires aligned binary labels and finite probabilities")
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def clusters(user_ids):
    ids, inverse = np.unique(user_ids, return_inverse=True)
    if len(ids) < 2:
        raise ValueError("User-cluster interval needs at least two users")
    counts = np.bincount(inverse).astype(np.float64)
    return ids, inverse, counts


def cluster_sums(values, inverse, n_users):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return np.column_stack([np.bincount(inverse, weights=values[:, j], minlength=n_users) for j in range(values.shape[1])])


def bootstrap_weights(n_users, seed=20260925, draws=2000):
    return np.random.default_rng(seed).multinomial(n_users, np.full(n_users, 1 / n_users), size=draws)


def paired_cluster_summary(delta_sums, counts, draws):
    """Resample shared user clusters, preserving within-user pairs and row counts."""
    delta_sums = np.asarray(delta_sums, dtype=np.float64)
    if delta_sums.ndim == 1:
        delta_sums = delta_sums[:, None]
    counts = np.asarray(counts, dtype=np.float64)
    if len(counts) != len(delta_sums) or (counts <= 0).any() or not np.isfinite(delta_sums).all():
        raise ValueError("Invalid cluster sums/counts")
    if draws.ndim != 2 or draws.shape[1] != len(counts) or (draws < 0).any() or not np.all(draws.sum(axis=1) == len(counts)):
        raise ValueError("Invalid user bootstrap multiplicities")
    total_rows = draws @ counts
    bootstrap = (draws @ delta_sums) / total_rows[:, None]
    balanced_bootstrap = draws @ (delta_sums / counts[:, None]) / len(counts)
    result = []
    for column in range(delta_sums.shape[1]):
        lo, hi = np.quantile(bootstrap[:, column], [.025, .975])
        blo, bhi = np.quantile(balanced_bootstrap[:, column], [.025, .975])
        result.append({"bce_difference": float(delta_sums[:, column].sum() / counts.sum()),
                       "ci_low": float(lo), "ci_high": float(hi),
                       "user_balanced_difference": float(np.mean(delta_sums[:, column] / counts)),
                       "user_balanced_ci_low": float(blo), "user_balanced_ci_high": float(bhi)})
    return result


def diagnostic_auc(labels, probabilities):
    """Mann-Whitney AUC with half credit for tied predictions; no bootstrap claim."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(probabilities, kind="mergesort")
    p, y = probabilities[order], labels[order]
    starts = np.r_[0, np.flatnonzero(p[1:] != p[:-1]) + 1]
    ends = np.r_[starts[1:], len(y)]
    pos = np.add.reduceat(y.astype(np.int64), starts)
    neg = ends - starts - pos
    return float(np.sum(pos * (np.cumsum(neg) - neg + .5 * neg), dtype=np.float64) / (positives * negatives))
