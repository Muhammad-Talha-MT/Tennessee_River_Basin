"""Forecast-like errors for realized rain that stands in for a rain forecast in training.

Training shows the model realized AORC rain where a forecast would be at test time,
and AORC is far better than any forecast. ``degrade`` gives it the errors of TVA's
basin forecast precipitation, measured against TVA's observed basin precipitation at
43 gauges for issues from October 2016 to December 2018 (training years only), per
24-hour lead day:

====  =====================  ====================  ===========================
day   bias (forecast/obs)    misses (forecast < 1  spread of log(forecast/obs)
                             mm when obs > 10 mm)  where both exceed 1 mm
====  =====================  ====================  ===========================
1     0.95                   0.12                  0.84
2     0.91                   0.16                  0.90
3     0.90                   0.18                  0.97
4     0.19                   0.85                  1.04
====  =====================  ====================  ===========================

TVA's rain is real to 72 h; on day 4 it has a fifth of the rain and misses most
storms. Each sample draws, per lead day, either a miss (rain set to zero) or a
log-normal factor whose mean gives the day's bias, plus a whole-forecast timing
shift of one step either way. Factors are shared by every reach of the sample, as
one forecast's errors are spatially coherent.
"""

import numpy as np
import torch

BIAS = (0.95, 0.91, 0.90, 0.19)
MISS = (0.12, 0.16, 0.18, 0.85)
SPREAD = (0.84, 0.90, 0.97, 1.04)
STEPS_PER_DAY = 4


def factors(rng, leads, shift_prob=0.2):
    """Per-lead multipliers [lead] and a timing shift in steps (-1, 0 or 1)."""
    days = -(-leads // STEPS_PER_DAY)
    per_day = np.empty(days)
    for d in range(days):
        k = min(d, len(BIAS) - 1)
        if rng.random() < MISS[k]:
            per_day[d] = 0.0
        else:
            # Mean of the kept factors x (1 - miss) = bias.
            mu = np.log(BIAS[k] / (1 - MISS[k])) - SPREAD[k] ** 2 / 2
            per_day[d] = np.exp(rng.normal(mu, SPREAD[k]))
    shift = int(rng.choice([-1, 1])) if rng.random() < shift_prob else 0
    return np.repeat(per_day, STEPS_PER_DAY)[:leads], shift


def degrade(value, flag, rng, rain, shift_prob=0.2):
    """Degrade normalized future rain ``value`` [lead, node] where ``flag`` [lead, node] is set.

    ``rain`` is (transform, mean, std) of the normalized precipitation channel.
    Returns a new tensor; values without a flag are unchanged.
    """
    transform, mean, std = rain
    leads = value.shape[0]
    scale, shift = factors(rng, leads, shift_prob)
    z = value.double()
    raw = z * std + mean
    raw = torch.expm1(raw).clamp_min(0) if transform == "log1p" else raw.clamp_min(0)
    raw = torch.where(flag, raw, torch.zeros_like(raw))
    if shift:
        moved = torch.zeros_like(raw)
        if shift > 0:
            moved[shift:] = raw[:-shift]
        else:
            moved[:shift] = raw[-shift:]
        raw = moved
    raw = raw * torch.as_tensor(scale, dtype=raw.dtype, device=raw.device)[:, None]
    out = torch.log1p(raw) if transform == "log1p" else raw
    out = ((out - mean) / std).to(value.dtype)
    return torch.where(flag, out, value)
