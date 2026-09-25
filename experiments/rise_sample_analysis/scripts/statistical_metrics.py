"""Exact empirical AUC and user-cluster linearized uncertainty for paired scores.

The confidence interval is an asymptotic cluster-sandwich interval conditional
on the trained models. It is not an exact bootstrap, a training-seed interval,
or an iid DeLong interval. Tied scores receive half credit.
"""

from __future__ import annotations

import math
import numpy as np


def validate_binary(y, scores):
    y = np.asarray(y)
    scores = np.asarray(scores)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not len(y):
        raise ValueError("Labels and scores must be nonempty aligned vectors")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores finite")
    return y.astype(np.uint8, copy=False), scores


def auc_and_influence(y, scores):
    """Return exact AUC and normalized row influence on that AUC.

    Summing row influences within a user gives the cluster contribution. The
    denominator influence of the empirical positive/negative prevalence is
    included, so users containing either or both classes are supported.
    """
    y, scores = validate_binary(y, scores)
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        raise ValueError("AUC requires both positive and negative examples")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_y = y[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1])]
    sizes = np.diff(np.r_[starts, len(y)])
    positive_ties = np.add.reduceat(sorted_y.astype(np.int64), starts)
    negative_ties = sizes - positive_ties
    lower_negative = np.cumsum(negative_ties) - negative_ties
    higher_positive = positives - np.cumsum(positive_ties)
    positive_placements = (lower_negative + 0.5 * negative_ties) / negatives
    negative_placements = (higher_positive + 0.5 * positive_ties) / positives
    auc = float(np.dot(positive_ties, positive_placements) / positives)
    positive_if = np.repeat((positive_placements - auc) / positives, sizes)
    negative_if = np.repeat((negative_placements - auc) / negatives, sizes)
    sorted_influence = np.where(sorted_y == 1, positive_if, negative_if)
    influence = np.empty(len(y), dtype=np.float64)
    influence[order] = sorted_influence
    return auc, influence


def binary_logloss_rows(y, probabilities):
    y, probabilities = validate_binary(y, probabilities)
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Logloss requires probabilities in [0,1]")
    dtype = probabilities.dtype if np.issubdtype(probabilities.dtype, np.floating) else np.dtype("float64")
    epsilon = np.finfo(dtype).eps
    p = np.clip(probabilities.astype(np.float64), epsilon, 1 - epsilon)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def cluster_interval(estimate, cluster_influence, *, quantile=1.959963984540054):
    """Normal interval with the finite-cluster K/(K-1) variance correction."""
    contributions = np.asarray(cluster_influence, dtype=np.float64)
    if contributions.ndim != 1 or len(contributions) < 2:
        raise ValueError("At least two independent sampling clusters required")
    if not np.isfinite(contributions).all():
        raise ValueError("Cluster influence must be finite")
    centered = contributions - contributions.mean()
    variance = len(centered) / (len(centered) - 1) * float(np.dot(centered, centered))
    se = math.sqrt(max(0.0, variance))
    return {"estimate": float(estimate), "cluster_se": se,
            "ci_lower": float(estimate - quantile * se),
            "ci_upper": float(estimate + quantile * se), "clusters": len(centered),
            "ci_method": "user-cluster linearized influence; normal 95%; trained models fixed"}


def paired_auc_difference(y, before, after, users):
    """AUC(after)-AUC(before), preserving pairing and arbitrary within-user dependence."""
    users = np.asarray(users)
    if users.ndim != 1 or len(users) != len(y):
        raise ValueError("User vector must align with examples")
    before_auc, before_if = auc_and_influence(y, before)
    after_auc, after_if = auc_and_influence(y, after)
    unique_users, user_inverse = np.unique(users, return_inverse=True)
    cluster_if = np.bincount(user_inverse, weights=after_if - before_if,
                             minlength=len(unique_users))
    result = cluster_interval(after_auc - before_auc, cluster_if)
    result.update(before_auc=before_auc, after_auc=after_auc, rows=len(y),
                  positives=int(np.sum(y)), negatives=int(len(y) - np.sum(y)))
    return result, unique_users, cluster_if


def group_difference_interval(first, second):
    """Difference of two group effects, retaining users present in both groups.

    Each argument is (effect, user_ids, normalized_user_influences).
    """
    value_a, users_a, influence_a = first
    value_b, users_b, influence_b = second
    users, inverse = np.unique(np.r_[users_a, users_b], return_inverse=True)
    influences = np.bincount(inverse, weights=np.r_[influence_a, -influence_b],
                             minlength=len(users))
    return cluster_interval(value_a - value_b, influences)


def auc_pair_decomposition(y, scores, groups):
    """Exact within/between-group contributions, including single-class groups."""
    y, scores = validate_binary(y, scores)
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(y):
        raise ValueError("Groups must align with examples")
    total_positive = int(y.sum())
    total_negative = len(y) - total_positive
    if not total_positive or not total_negative:
        raise ValueError("Overall AUC requires both classes")
    values = np.unique(groups)
    output = []
    for positive_group in values:
        positive_scores = scores[(groups == positive_group) & (y == 1)]
        for negative_group in values:
            negative_scores = np.sort(scores[(groups == negative_group) & (y == 0)])
            pairs = len(positive_scores) * len(negative_scores)
            weight = pairs / (total_positive * total_negative)
            if pairs:
                lower = np.searchsorted(negative_scores, positive_scores, side="left")
                upper = np.searchsorted(negative_scores, positive_scores, side="right")
                auc = float(np.sum((lower + upper) * 0.5) / pairs)
            else:
                auc = None
            output.append({"positive_group": str(positive_group),
                           "negative_group": str(negative_group), "positive_negative_pairs": pairs,
                           "weight": weight, "pair_auc": auc,
                           "contribution": weight * auc if pairs else 0.0,
                           "within_group": bool(positive_group == negative_group)})
    return output
