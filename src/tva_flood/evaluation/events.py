"""Event-based skill from the forecasts that ``evaluate-observed`` saves.

Median NSE hides what a flood forecaster needs, so this scores the same forecasts
two more ways, per lead and per method, beside TVA's own forecasts:

- **Threshold exceedance** (every issue, lead and gauge pooled): was flow above a
  flood threshold, and did the forecast say so? Probability of detection (POD),
  false alarm ratio (FAR), critical success index (CSI) and frequency bias, with a
  paired bootstrap interval on the CSI difference to the reference method, resampled
  over (year, gauge) cells.
- **Observed flood peaks**: for each observed peak above a threshold, the forecast
  issued a given lead before it. Peak size error (forecast maximum within a window
  around the peak, against the observed peak), timing error of that maximum, and
  whether the forecast crossed the threshold at all (detection).

Thresholds are per gauge, from the training years only: the 95th and 99th percentile
of six-hour flow and the median annual maximum ("annual").

Lead hours follow ``evaluate-observed``: TVA lead 6(k+1) h = model step k.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from tva_flood.data.gauges import Catalog

STEPS = 40
THRESHOLDS = ("p95", "p99", "annual")
BINS = ((6, 24), (30, 72), (78, 120), (126, 240))


def load_forecasts(runs):
    """Concatenate the ``forecasts.npz`` of several evaluation folders (one per year)."""
    parts = []
    for run in runs:
        with np.load(Path(run) / "forecasts.npz") as z:
            parts.append({k: z[k] for k in z.files})
    first = parts[0]
    for p in parts[1:]:
        if list(p["methods"]) != list(first["methods"]) or list(p["stations"]) != list(
            first["stations"]
        ):
            raise ValueError("Evaluation folders differ in methods or stations")
    return dict(
        time=pd.DatetimeIndex(np.concatenate([p["time"] for p in parts])),
        truth=np.concatenate([p["truth"] for p in parts]),
        values=np.concatenate([p["values"] for p in parts]),
        methods=[str(m) for m in first["methods"]],
        stations=[str(s) for s in first["stations"]],
    )


def thresholds(observed, times, stations, names, before):
    """Per-gauge flood thresholds [name -> gauge] from flow before ``before``."""
    column = {str(n): i for i, n in enumerate(names)}
    train = pd.DatetimeIndex(times) < before
    result = {t: np.full(len(stations), np.nan) for t in THRESHOLDS}
    for s, station in enumerate(stations):
        x = pd.Series(observed[train, column[station]].astype(float), index=times[train])
        x = x.dropna()
        if len(x) < 1460:
            continue
        result["p95"][s], result["p99"][s] = x.quantile([0.95, 0.99])
        annual = x.groupby(x.index.year).max()
        result["annual"][s] = annual[x.groupby(x.index.year).size() > 1000].median()
    return result


def categorical(truth, forecast, years, threshold):
    """Contingency counts [year, lead, gauge] (hits, misses, false alarms) for one method."""
    valid = np.isfinite(truth) & np.isfinite(forecast)
    observed = truth >= threshold[None, None, :]
    warned = forecast >= threshold[None, None, :]
    counts = []
    for y in np.unique(years):
        rows = years == y
        v = valid[rows]
        counts.append(
            [
                (observed[rows] & warned[rows] & v).sum(axis=0),
                (observed[rows] & ~warned[rows] & v).sum(axis=0),
                (~observed[rows] & warned[rows] & v).sum(axis=0),
            ]
        )
    return np.array(counts).transpose(1, 0, 2, 3)  # [hits|misses|false, year, lead, gauge]


def scores(hits, misses, false):
    """POD, FAR, CSI and frequency bias from counts (NaN where undefined)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(
            pod=hits / (hits + misses),
            far=false / (hits + false),
            csi=hits / (hits + misses + false),
            bias=(hits + false) / (hits + misses),
        )


def csi_difference(a, b, rng, draws):
    """Bootstrap the pooled CSI difference a - b over (year, gauge) cells; [draws, lead]."""
    cells = a.shape[1] * a.shape[3]
    a = a.transpose(0, 2, 1, 3).reshape(3, a.shape[2], cells)
    b = b.transpose(0, 2, 1, 3).reshape(3, b.shape[2], cells)
    out = np.empty((draws, a.shape[1]))
    for d in range(draws):
        pick = rng.integers(0, cells, cells)
        out[d] = (
            scores(*a[:, :, pick].sum(axis=2))["csi"] - scores(*b[:, :, pick].sum(axis=2))["csi"]
        )
    return out


def find_events(series, threshold, separation=16):
    """Indices of independent observed peaks above the threshold (NaN counts as low flow)."""
    if not np.isfinite(threshold):
        return np.array([], dtype=int)
    peaks, _ = find_peaks(np.nan_to_num(series, nan=-1.0), height=threshold, distance=separation)
    return peaks


def event_rows(forecasts, catalog_index, observed, station, threshold, name, window):
    """Per event and lead: forecast peak, size error, timing error, detection, per method.

    ``catalog_index[row]`` is the position of each issue time on the observation
    timeline ``observed`` of this gauge.
    """
    s = forecasts["stations"].index(station)
    values = forecasts["values"][:, :, :, s]  # [issue, method, lead]
    row_of = {int(c): r for r, c in enumerate(catalog_index)}
    first, last = int(catalog_index.min()), int(catalog_index.max()) + STEPS
    rows = []
    for peak in find_events(observed, threshold):
        if not first <= peak <= last or not np.isfinite(observed[peak]):
            continue
        for k in range(STEPS):
            row = row_of.get(int(peak) - k)
            if row is None:
                continue
            steps = np.arange(k - window, k + window + 1)
            steps = steps[(steps >= 0) & (steps < STEPS)]
            part = values[row][:, steps]  # [method, window]
            if not np.isfinite(part).any(axis=1).all():
                continue
            top = np.nanargmax(np.where(np.isfinite(part), part, -np.inf), axis=1)
            for m, method in enumerate(forecasts["methods"]):
                if not np.isfinite(part[m]).any():
                    continue
                maximum = part[m, top[m]]
                rows.append(
                    dict(
                        station=station,
                        peak_index=int(peak),
                        threshold=name,
                        method=method,
                        lead_hours=6 * (k + 1),
                        observed_peak=float(observed[peak]),
                        forecast_peak=float(maximum),
                        size_error_pct=100
                        * float(maximum - observed[peak])
                        / float(observed[peak]),
                        timing_error_h=6 * float(steps[top[m]] - k),
                        detected=bool(maximum >= threshold),
                    )
                )
    return rows


def summarise_events(table):
    """Per threshold, method and lead: events, detection rate, and median size and timing error."""
    grouped = table.groupby(["threshold", "method", "lead_hours"])
    return grouped.agg(
        events=("detected", "size"),
        detection_rate=("detected", "mean"),
        median_size_error_pct=("size_error_pct", "median"),
        median_abs_size_error_pct=("size_error_pct", lambda v: v.abs().median()),
        median_abs_timing_error_h=("timing_error_h", lambda v: v.abs().median()),
        timing_within_6h=("timing_error_h", lambda v: (v.abs() <= 6).mean()),
    ).reset_index()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="evaluate-observed folders")
    p.add_argument("--gauges", type=Path, default=Path("/data/tva_data/processed/gauges/gauges.nc"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", default="tva_forecast")
    p.add_argument("--candidate", default="tuned+qpf")
    p.add_argument("--train-until", default="2019-01-01")
    p.add_argument("--window", type=int, default=4, help="steps either side of an observed peak")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.out.exists():
        p.error(f"{a.out} exists; choose a new --out")
    a.out.mkdir(parents=True)
    forecasts = load_forecasts(a.runs)
    for m in (a.reference, a.candidate):
        if m not in forecasts["methods"]:
            p.error(f"{m} is not among {forecasts['methods']}")
    catalog = Catalog(a.gauges)
    limits = thresholds(
        catalog.observed,
        catalog.times,
        forecasts["stations"],
        catalog.names,
        pd.Timestamp(a.train_until),
    )
    pd.DataFrame(limits, index=forecasts["stations"]).to_csv(a.out / "thresholds.csv")
    years = forecasts["time"].year.to_numpy()
    rng = np.random.default_rng(a.seed)
    rows, intervals = [], []
    for name in THRESHOLDS:
        counts = {
            m: categorical(forecasts["truth"], forecasts["values"][:, i], years, limits[name])
            for i, m in enumerate(forecasts["methods"])
        }
        for m, c in counts.items():
            pooled = scores(*c.sum(axis=(1, 3)))
            for lead in range(STEPS):
                rows.append(
                    dict(
                        threshold=name,
                        method=m,
                        lead_hours=6 * (lead + 1),
                        hits=int(c[0].sum(axis=(0, 2))[lead]),
                        misses=int(c[1].sum(axis=(0, 2))[lead]),
                        false_alarms=int(c[2].sum(axis=(0, 2))[lead]),
                        **{k: float(v[lead]) for k, v in pooled.items()},
                    )
                )
        diff = csi_difference(counts[a.candidate], counts[a.reference], rng, a.bootstrap)
        for lo, hi in BINS:
            keep = [i for i in range(STEPS) if lo <= 6 * (i + 1) <= hi]
            d = np.nanmean(diff[:, keep], axis=1)
            intervals.append(
                dict(
                    threshold=name,
                    leads=f"{lo}-{hi} h",
                    csi_difference=float(np.nanmean(d)),
                    low=float(np.nanpercentile(d, 2.5)),
                    high=float(np.nanpercentile(d, 97.5)),
                )
            )
    categorical_table = pd.DataFrame(rows)
    categorical_table.to_csv(a.out / "exceedance_by_lead.csv", index=False)
    interval_table = pd.DataFrame(intervals)
    interval_table.to_csv(a.out / "csi_difference.csv", index=False)

    # Observed peaks, on the observation timeline of each gauge.
    index = catalog.times.get_indexer(forecasts["time"])
    if (index < 0).any():
        raise ValueError("Issue times outside the gauge catalog timeline")
    names = list(catalog.names)
    events = []
    for name in THRESHOLDS:
        for s, station in enumerate(forecasts["stations"]):
            series = catalog.observed[:, names.index(station)].astype(float)
            for row in event_rows(
                forecasts, index, series, station, limits[name][s], name, a.window
            ):
                row["event"] = str(catalog.times[row.pop("peak_index")])
                events.append(row)
    events = pd.DataFrame(events)
    events.to_csv(a.out / "events.csv.gz", index=False)
    summary = summarise_events(events)
    summary.to_csv(a.out / "events_by_lead.csv", index=False)
    (a.out / "report.md").write_text(render(categorical_table, interval_table, summary, a))
    print((a.out / "report.md").read_text())
    print(f"EVENT EVALUATION COMPLETE: {a.out}")


def render(table, intervals, events, a):
    upper = a.candidate.rsplit("+", 1)[0] + "+aorc"
    extra = [
        m for m in dict.fromkeys((upper, "tuned+aorc", "persistence")) if m in set(table.method)
    ]
    methods = [a.candidate, a.reference] + [m for m in extra if m not in (a.candidate, a.reference)]
    leads = (6, 24, 48, 72, 96, 120)
    lines = [f"# Event-based skill ({', '.join(str(r) for r in a.runs[:1])} ...)", ""]
    for name in ("p99", "annual"):
        lines += [
            f"## Exceedance of the {name} threshold",
            "",
            "| lead | method | POD | FAR | CSI | bias |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for lead in leads:
            for m in methods:
                r = table[
                    (table.threshold == name) & (table.method == m) & (table.lead_hours == lead)
                ]
                if len(r):
                    r = r.iloc[0]
                    lines.append(
                        f"| {lead} h | {m} | {r.pod:.2f} | {r.far:.2f} | {r.csi:.2f} | {r.bias:.2f} |"
                    )
        lines += ["", f"CSI difference, {a.candidate} minus {a.reference} (95% bootstrap):", ""]
        for r in intervals[intervals.threshold == name].itertuples():
            lines.append(f"- {r.leads}: {r.csi_difference:+.3f} [{r.low:+.3f}, {r.high:+.3f}]")
        lines += [
            "",
            f"## Observed {name} peaks",
            "",
            "| lead | method | events | detected | median size error % | median abs timing error h |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for lead in leads:
            for m in methods:
                r = events[
                    (events.threshold == name) & (events.method == m) & (events.lead_hours == lead)
                ]
                if len(r):
                    r = r.iloc[0]
                    lines.append(
                        f"| {lead} h | {m} | {r.events} | {r.detection_rate:.0%} | "
                        f"{r.median_size_error_pct:+.0f} | {r.median_abs_timing_error_h:.0f} |"
                    )
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
