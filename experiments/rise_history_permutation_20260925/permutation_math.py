"""Valid-history permutations; candidate and padding stay at their native positions."""
from contextlib import contextmanager
import hashlib
import numpy as np

PERMUTATION_SEED = 20260925


def digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def make_permutation(mask, mode, seed=PERMUTATION_SEED):
    mask = np.asarray(mask, dtype=bool)
    mapping = np.broadcast_to(np.arange(mask.shape[1]), mask.shape).copy()
    rng = np.random.default_rng(seed)
    if mode not in ("identity", "off", "reverse", "random"):
        raise ValueError(mode)
    if mode in ("reverse", "random"):
        for row in range(len(mask)):
            valid = np.flatnonzero(mask[row])
            mapping[row, valid] = valid[::-1] if mode == "reverse" else rng.permutation(valid)
    original = np.broadcast_to(np.arange(mask.shape[1]), mask.shape)
    if not np.array_equal(np.sort(mapping, axis=1), original):
        raise AssertionError("Not a row-wise bijection")
    if not np.array_equal(mapping[~mask], original[~mask]):
        raise AssertionError("Padding moved")
    if not np.array_equal(np.take_along_axis(mask, mapping, axis=1), mask):
        raise AssertionError("Allowed positions changed")
    return mapping.astype(np.int16)


def gather_history(tensor, mapping):
    """Joint gather on dimension 1, retaining all feature dimensions."""
    shape = [mapping.shape[0], mapping.shape[1]] + [1] * (tensor.ndim - 2)
    indices = mapping.reshape(shape).expand(*mapping.shape, *tensor.shape[2:])
    return tensor.gather(1, indices)


def combined_mapping(mapping):
    import torch
    return torch.cat((mapping, mapping.new_full((len(mapping), 1), mapping.shape[1])), dim=1)


def align_bias(bias, mapping):
    inverse = mapping.argsort(dim=1)
    return bias.gather(-1, inverse[:, None, None, :].expand_as(bias))


def joint_raw_batch(batch, mapping):
    """Permute every historical item attribute and masks; keep other data fixed."""
    import torch
    from copy import deepcopy
    result = list(deepcopy(batch))
    full = combined_mapping(mapping)
    for name, values in batch[1].items():
        if values.ndim < 2 or values.shape[:2] != full.shape:
            raise ValueError(f"Unexpected item history layout: {name} {values.shape}")
        result[1][name] = gather_history(values, full)
        if not torch.equal(result[1][name][:, -1], values[:, -1]):
            raise AssertionError("Candidate attribute moved")
    result[2] = gather_history(batch[2], mapping)
    if len(batch) == 4 and batch[3] is not None:
        if isinstance(batch[3], (list, tuple)):
            result[3] = type(batch[3])(gather_history(value, mapping) for value in batch[3])
        elif batch[3].ndim == 3 and batch[3].shape[1:] == mapping.shape:
            result[3] = torch.stack([gather_history(value, mapping) for value in batch[3]])
        else:
            result[3] = gather_history(batch[3], mapping)
    for name in batch[0]:
        if not torch.equal(result[0][name], batch[0][name]):
            raise AssertionError("Nonhistory field or label changed")
    if not torch.equal(result[2], batch[2]):
        raise AssertionError("Main valid mask changed")
    return type(batch)(result)


@contextmanager
def eager_children(model):
    """Use shared original modules for the small numerical check, then restore."""
    replaced = []
    def visit(parent):
        for name, child in list(parent._modules.items()):
            if child is None:
                continue
            if hasattr(child, "_orig_mod"):
                original = child._orig_mod
                replaced.append((parent, name, child))
                parent._modules[name] = original
                child = original
            visit(child)
    visit(model)
    try:
        yield
    finally:
        for parent, name, child in reversed(replaced):
            parent._modules[name] = child
