"""Additive, masked statistics in physical and normalized log units."""

import numpy as np


def sums(pn, pr, yn, yr, mask, mean, std, axis=None):
    """Sufficient statistics: count, target sum/squares, raw SSE/SAE, log SSE, negative count."""
    err = np.where(mask, pr - yr, 0.0)
    logerr = np.where(mask, pn - yn, 0.0)
    return np.stack(
        [
            mask.sum(axis=axis),
            np.where(mask, yr, 0.0).sum(axis=axis),
            np.where(mask, yr * yr, 0.0).sum(axis=axis),
            (err * err).sum(axis=axis),
            np.abs(err).sum(axis=axis),
            (logerr * logerr).sum(axis=axis),
            (mask & (pn * std + mean < 0)).sum(axis=axis),
        ]
    ).astype(np.float64)


def metrics(a):
    n, sy, sy2, se, ae, le, neg = map(float, a)
    if not n:
        return dict(
            count=0,
            mae_m3_s=None,
            rmse_m3_s=None,
            normalized_log_mse=None,
            nse=None,
            negative_predictions_clipped=0,
        )
    denom = sy2 - sy * sy / n
    return dict(
        count=int(n),
        mae_m3_s=ae / n,
        rmse_m3_s=(se / n) ** 0.5,
        normalized_log_mse=le / n,
        nse=1 - se / denom if denom > max(abs(sy2) * 1e-12, 1e-12) else None,
        negative_predictions_clipped=int(neg),
    )
