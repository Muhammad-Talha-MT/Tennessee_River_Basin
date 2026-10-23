"""Benchmark pieces: USGS ids, NWM six-hour means, LSTM shapes and rain fields."""

import numpy as np
import torch

from tva_flood.downloads.nwm_forecasts import FLOW_LEADS, six_hour_flow
from tva_flood.downloads.usgs import usgs_id
from tva_flood.models.lstm import ForecastLSTM


def test_usgs_ids_regain_leading_zeros():
    assert usgs_id(3512000) == "03512000"
    assert usgs_id("3606500.0") == "03606500"
    assert usgs_id(345577330) == "0345577330"


def test_six_hour_flow_is_a_trapezoid_of_three_hourly_values():
    hours = FLOW_LEADS.astype(float)
    flow = np.stack([hours, 2 * hours], axis=-1)  # linear in time, two gauges
    mean = six_hour_flow(flow)
    assert mean.shape == (40, 2)
    # A linear series averages to its midpoint: step k covers [6k, 6k+6] h.
    assert np.allclose(mean[1:, 0], 6 * np.arange(1, 40) + 3)
    assert np.allclose(mean[1:, 1], 2 * (6 * np.arange(1, 40) + 3))
    assert np.isclose(mean[0, 0], 4.5)  # step 0: mean of 3 h and 6 h only


def test_forecast_lstm_shapes_and_future_dependence():
    torch.manual_seed(0)
    model = ForecastLSTM(hidden=8, dropout=0.0, horizon=40).eval()
    history, static = torch.randn(3, 120, 18), torch.randn(3, 48)
    future = torch.zeros(3, 40, 2)
    base = model(history, static, future)
    assert base.shape == (3, 40)
    future[:, 10, 0] = 5.0
    changed = model(history, static, future)
    assert torch.allclose(changed[:, :10], base[:, :10])  # causal along leads
    assert not torch.allclose(changed[:, 10:], base[:, 10:])
