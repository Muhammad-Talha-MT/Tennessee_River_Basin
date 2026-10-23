"""Event scoring: contingency counts, peak finding, and peak size and timing errors."""

import numpy as np

from tva_flood.evaluation import events


def test_scores_from_counts():
    s = events.scores(np.array([6.0]), np.array([2.0]), np.array([2.0]))
    assert np.allclose(
        [s["pod"][0], s["far"][0], s["csi"][0], s["bias"][0]], [0.75, 0.25, 0.6, 1.0]
    )


def test_categorical_counts_by_year():
    truth = np.array([[[10.0]], [[1.0]], [[10.0]]])  # [issue, lead, gauge]
    forecast = np.array([[[10.0]], [[10.0]], [[1.0]]])
    years = np.array([2020, 2020, 2021])
    counts = events.categorical(truth, forecast, years, np.array([5.0]))
    assert counts[:, :, 0, 0].tolist() == [[1, 0], [0, 1], [1, 0]]  # hits, misses, false alarms


def test_find_events_separates_peaks_and_ignores_nan():
    x = np.zeros(100)
    x[[10, 14, 60]] = [5, 9, 7]
    x[30] = np.nan
    assert events.find_events(x, 4.0).tolist() == [14, 60]
    assert len(events.find_events(x, np.nan)) == 0


def test_event_rows_size_timing_and_detection():
    # One gauge, one observed peak of 100 at index 50. The forecast issued 3 steps
    # earlier (row 0) peaks at 60 one step late; no other issue exists.
    observed = np.zeros(200)
    observed[50] = 100.0
    values = np.full((1, 1, events.STEPS, 1), np.nan, dtype=np.float32)
    values[0, 0, :, 0] = 1.0
    values[0, 0, 3 + 1, 0] = 60.0
    forecasts = dict(values=values, methods=["m"], stations=["G"], time=None)
    rows = events.event_rows(forecasts, np.array([47]), observed, "G", 50.0, "annual", 4)
    assert len(rows) == 1
    row = rows[0]
    assert row["lead_hours"] == 6 * 4 and row["forecast_peak"] == 60.0
    assert np.isclose(row["size_error_pct"], -40.0) and row["timing_error_h"] == 6.0
    assert row["detected"]
    table = events.summarise_events(__import__("pandas").DataFrame(rows))
    assert table.detection_rate.iloc[0] == 1.0
