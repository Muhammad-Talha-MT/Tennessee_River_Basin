"""Like-for-like comparison pieces: shared selection score, basin forecast rain, rain cache."""

import numpy as np
import scipy.sparse as sp

from tva_flood.data.basin_series import basin_forecast
from tva_flood.data.rain_forecasts import RainForecasts
from tva_flood.training.selection import nse_by_lead, selection


def test_nse_by_lead_matches_the_definition_and_needs_enough_values():
    rng = np.random.default_rng(0)
    truth = rng.gamma(2.0, 3.0, (200, 40, 3))
    prediction = truth + rng.normal(0, 1.0, truth.shape)
    truth[:150, :, 2] = np.nan  # gauge 2 has only 50 values per lead
    scores = nse_by_lead(prediction, truth, min_count=100)
    o, s = truth[:, 5, 0], prediction[:, 5, 0]
    expected = 1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    assert np.isclose(scores[5, 0], expected)
    assert np.isnan(scores[:, 2]).all()
    score, median = selection(scores)
    assert np.isclose(score, np.mean(median[:20]))


def test_basin_forecast_averages_normalized_reach_rain_within_the_window():
    field = np.array([[0.0, 4.0, np.nan], [1.0, 1.0, 1.0], [9.0, 9.0, 9.0]])  # [step, reach]
    rows = sp.csr_matrix(np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]))  # two basins
    covered = np.array([True, True, True])
    out = basin_forecast(field, rows, covered, ("log1p", 0.0, 1.0), steps=2)
    assert out.shape == (2, 40, 2)
    assert np.isclose(out[0, 0, 0], 0.5 * np.log1p(4.0))
    assert np.isclose(out[1, 0, 1], 0.5)  # reach 2 is missing at step 0
    assert np.isclose(out[1, 1, 1], 1.0)
    assert not out[:, 2:].any()  # step 2 lies beyond the rain window


def test_rain_cache_round_trip_and_rebuild(tmp_path):
    def month(name, days, value):
        np.savez(
            tmp_path / name,
            issue=np.array(days),
            rain=np.full((len(days), 2, 3), value, dtype=np.float16),
        )

    month("2019-01.npz", ["2019-01-01", "2019-01-02"], 1.0)
    rain = RainForecasts(tmp_path)
    assert "2019-01-02 06:00" in rain and "2019-01-03" not in rain
    assert rain.get("2019-01-02").shape == (2, 3)
    month("2019-02.npz", ["2019-02-01"], 2.0)
    rain = RainForecasts(tmp_path)  # a new month rebuilds the cache
    assert np.allclose(rain.get("2019-02-01"), 2.0)
    assert len(rain.days) == 3


def test_holdout_folds_are_balanced_by_area_and_skip_untrainable_gauges(tmp_path):
    from types import SimpleNamespace

    import pandas as pd

    from tva_flood.data.folds import held_out, holdout_folds

    (tmp_path / "processed/graph_candidate").mkdir(parents=True)
    n = 23
    comid = np.arange(100, 100 + n)
    pd.DataFrame(dict(comid=comid, totdasqkm=np.linspace(10, 5000, n))).to_parquet(
        tmp_path / "processed/graph_candidate/nodes.parquet"
    )
    times = pd.to_datetime(["2018-01-01", "2018-07-01", "2019-03-01", "2019-09-01"])
    observed = np.ones((4, n), dtype=np.float32)
    observed[:2, 0] = np.nan  # no training-period data: never held out
    camelsh = np.ones(n, dtype=bool)
    camelsh[1] = False  # not a CAMELS-H site: never held out
    catalog = SimpleNamespace(comid=comid, times=times, observed=observed, camelsh=camelsh)
    fold = holdout_folds(tmp_path, catalog, 5)
    assert fold[0] == 0 and fold[1] == 0
    sizes = np.bincount(fold, minlength=6)[1:]
    assert sizes.sum() == n - 2 and sizes.max() - sizes.min() <= 1
    assert np.array_equal(fold, holdout_folds(tmp_path, catalog, 5))
    # Each block of five consecutive areas holds every fold once.
    order = np.argsort(np.linspace(10, 5000, n)[2:])
    assert sorted(fold[2:][order][:5]) == [1, 2, 3, 4, 5]
    assert held_out(tmp_path, catalog, 3, 5).sum() == (fold == 3).sum()
    assert not held_out(tmp_path, catalog, 0, 5).any()
