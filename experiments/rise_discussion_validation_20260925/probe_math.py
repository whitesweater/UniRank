"""Label-free cross-part correction controls and numerical summaries."""
import numpy as np

DONOR_SEEDS = (20260925, 20260926, 20260927)


def moments(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=1, keepdims=True)
    centered = values - mean
    rms = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
    return mean, rms


def match_moments(source, recipient):
    source, recipient = np.asarray(source, dtype=np.float64), np.asarray(recipient, dtype=np.float64)
    sm, sr = moments(source)
    rm, rr = moments(recipient)
    if np.any((sr == 0) & (rr > 0)):
        raise ValueError("Flat donor cannot match a nonflat recipient")
    scale = np.divide(rr, sr, out=np.zeros_like(rr), where=sr > 0)
    return (source - sm) * scale + rm


def crosspart_templates(delta, parts, masks):
    delta, parts, masks = np.asarray(delta), np.asarray(parts), np.asarray(masks, dtype=bool)
    if delta.ndim != 2 or masks.shape != delta.shape or len(parts) != len(delta):
        raise ValueError("Misaligned template inputs")
    # The frozen input inspection established full-length histories for all rows.
    # Abort on drift rather than silently change coverage or matching rules.
    if not masks.all() or len(np.unique(parts)) != 8 or not np.isfinite(delta).all():
        raise ValueError("Expected eight parts with all native keys valid")
    templates = np.empty(delta.shape, dtype=np.float64)
    receipts = []
    total = delta.sum(axis=0, dtype=np.float64)
    for part in np.unique(parts):
        own = parts == part
        peer_count = int((~own).sum())
        template = (total - delta[own].sum(axis=0, dtype=np.float64)) / peer_count
        templates[own] = template
        receipts.append({"recipient_part": int(part), "recipient_rows": int(own.sum()),
                         "peer_rows": peer_count, "excluded_own_part": True})
    centered = templates - templates.mean(axis=1, keepdims=True)
    raw = centered + moments(delta)[0]
    return raw, match_moments(templates, delta), receipts


def crosspart_donor_indices(parts, seed):
    """One-to-one mapping: derange physical parts, permute rows within each donor part."""
    parts = np.asarray(parts)
    unique = np.unique(parts)
    groups = [np.flatnonzero(parts == part) for part in unique]
    if len(unique) < 2 or len({len(group) for group in groups}) != 1:
        raise ValueError("Donor permutation requires equally sized physical parts")
    rng = np.random.default_rng(seed)
    order = np.arange(len(unique))
    while True:
        donor_parts = rng.permutation(len(unique))
        if np.all(donor_parts != order):
            break
    result = np.empty(len(parts), dtype=np.int64)
    for target, donor in zip(groups, donor_parts):
        result[target] = rng.permutation(groups[donor])
    if np.any(parts[result] == parts) or len(np.unique(result)) != len(parts):
        raise ValueError("Donor map failed no-own-part or one-to-one invariant")
    return result


def bce(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("BCE requires aligned binary outcomes and valid probabilities")
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def row_diagnostics(full_delta, actual_delta, full_weight, actual_weight):
    fm, fr = moments(full_delta)
    am, ar = moments(actual_delta)
    centered_f, centered_a = full_delta - fm, actual_delta - am
    numerator = np.mean(centered_f * centered_a, axis=1)
    denominator = fr[:, 0] * ar[:, 0]
    cosine = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return {"actual_bias_mean": am[:, 0], "actual_bias_centered_rms": ar[:, 0],
            "full_bias_centered_rms": fr[:, 0], "bias_cosine_to_full": cosine,
            "attention_tv_to_full": np.abs(full_weight - actual_weight).sum(axis=1) / 2,
            "top_key_changed_from_full": full_weight.argmax(axis=1) != actual_weight.argmax(axis=1),
            "attention_entropy": -(actual_weight * np.log(np.maximum(actual_weight, 1e-30))).sum(axis=1)}
