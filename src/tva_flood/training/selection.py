"""Checkpoint selection score shared by the graph model and the LSTM baseline.

Both are scored on the same validation forecasts: per gauge and lead, NSE over all
validation issues with an observed target and an observation at the issue time
(at least ``min_count`` values); then the median over gauges per lead, averaged
over the 6-120 h leads. Higher is better.
"""

import numpy as np

SELECTION_LEADS = 20  # 6-120 h


def nse_by_lead(prediction, truth, min_count):
    """NSE [lead, gauge] over issues; ``prediction``/``truth`` [issue, lead, gauge], NaN skipped."""
    ok = np.isfinite(prediction) & np.isfinite(truth)
    count = ok.sum(axis=0)
    o = np.where(ok, truth, 0.0)
    s = np.where(ok, prediction, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = o.sum(axis=0) / count
        variance = (np.where(ok, (truth - mean) ** 2, 0.0)).sum(axis=0)
        error = (np.where(ok, (s - o) ** 2, 0.0)).sum(axis=0)
        score = 1 - error / variance
    score[(count < min_count) | ~(variance > 0)] = np.nan
    return score


def selection(scores):
    """Mean over the 6-120 h leads of the median NSE over gauges; ``scores`` [lead, gauge]."""
    with np.errstate(all="ignore"):
        median = np.array([np.nanmedian(s) if np.isfinite(s).any() else np.nan for s in scores])
    return float(np.nanmean(median[:SELECTION_LEADS])), median
