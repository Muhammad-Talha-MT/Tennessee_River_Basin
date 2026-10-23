"""Forecast-like errors on realized training rain."""

import numpy as np
import torch

from tva_flood.data import rain_error

RAIN = ("log1p", 0.3, 0.7)


def normalized(mm):
    return ((np.log1p(mm) - RAIN[1]) / RAIN[2]).astype(np.float32)


def raw(z):
    return np.expm1(z.numpy().astype(np.float64) * RAIN[2] + RAIN[1])


def test_factors_match_the_measured_bias_per_day():
    rng = np.random.default_rng(0)
    draws = np.array([rain_error.factors(rng, 16, shift_prob=0)[0] for _ in range(40000)])
    for day, bias in enumerate(rain_error.BIAS):
        assert np.mean(draws[:, 4 * day]) == np.float64(np.mean(draws[:, 4 * day + 3]))
        assert abs(np.mean(draws[:, 4 * day]) - bias) < 0.05 * max(bias, 0.2)
        assert abs(np.mean(draws[:, 4 * day] == 0) - rain_error.MISS[day]) < 0.01


def test_degrade_changes_only_flagged_rain_and_keeps_zero_rain_zero():
    mm = np.zeros((16, 3))
    mm[:, 1] = 5.0
    value = torch.as_tensor(normalized(mm))
    flag = torch.ones(16, 3, dtype=torch.bool)
    flag[:, 2] = False
    out = rain_error.degrade(value, flag, np.random.default_rng(1), RAIN, shift_prob=0)
    assert torch.equal(out[:, 2], value[:, 2])
    assert np.allclose(raw(out[:, 0]), 0, atol=1e-6)
    scale, _ = rain_error.factors(np.random.default_rng(1), 16, shift_prob=0)
    assert np.allclose(raw(out[:, 1]), 5.0 * scale, rtol=1e-4, atol=1e-5)


def test_timing_shift_moves_the_whole_forecast_one_step():
    mm = np.zeros((16, 1))
    mm[5] = 10.0
    value = torch.as_tensor(normalized(mm))
    flag = torch.ones(16, 1, dtype=torch.bool)
    for seed in range(50):
        scale, shift = rain_error.factors(np.random.default_rng(seed), 16, shift_prob=1)
        out = raw(rain_error.degrade(value, flag, np.random.default_rng(seed), RAIN, 1))[:, 0]
        assert shift in (-1, 1)
        assert np.allclose(out[5 + shift], 10.0 * scale[5 + shift], rtol=1e-4)
        assert np.allclose(np.delete(out, 5 + shift), 0, atol=1e-6)
