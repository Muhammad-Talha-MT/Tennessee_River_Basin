"""Transforms shared by fitting and sample loading."""

import numpy as np


def transform_precipitation(values, valid=None, mode="log1p"):
    """Transform raw six-hour mm totals; invalid values become zero placeholders.

    Called by both statistics fitting and the loader. Validity masks are retained
    separately; zero placeholders never participate in normalization statistics.
    """
    if mode not in ("none", "log1p"):
        raise ValueError(f"Unknown precipitation transform: {mode}")
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != values.shape or not np.isfinite(values[valid]).all():
        raise ValueError("Invalid precipitation mask or nonfinite valid values")
    if (values[valid] < 0).any():
        raise ValueError("Negative valid precipitation; inspect source instead of clipping")
    safe = np.where(valid, values, 0.0)
    return np.log1p(safe) if mode == "log1p" else safe
