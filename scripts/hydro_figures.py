"""Figures of the graph-routing model (tva_flood.hydro) against the earlier models.

    python scripts/hydro_figures.py --out /data/tva_data/figures/hydro

Reads models/hydro_evaluation (the new model: forecasts at gauges, held-out gauges and
per-reach agreement with the NWM retrospective) and the saved forecasts of the earlier
models, all scored on the same 00Z issues 2020-2025, CAMELS-H gauges and USGS observations:
fair_benchmark (the earlier graph model V2, 3 seeds; NWM; persistence; climatology),
lstm_tuning_benchmark (the tuned LSTM, the strongest LSTM baseline, 3 seeds),
heldout_benchmark_f1-5 (both, with each fold's gauges withheld), and
retrospective_comparison* (per-reach agreement of V2 and NWM with the NWM retrospective).
Scores of a multi-seed model are computed per seed and then averaged. Forecast rain is
NWM's unless a panel says "realized rain" (an upper bound).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparison_figures as cf  # noqa: E402  (sets the plot style)

from comparison_figures import INK, LEGEND, MUTED, plt  # noqa: E402

M = Path("/data/tva_data/models")
ROOT = Path("/data/tva_data")
EVAL = M / "hydro_evaluation"
NEW, V2, LSTM, NWM, PERSIST, CLIM = "#eb6834", "#2a78d6", "#008300", "#6250d6", "#eda100", "#e87ba4"
LEADS = np.arange(1, 41) * 6
DAYS = [3, 7, 11, 15, 19]  # lead indices of 24, 48, 72, 96, 120 h
MODELS = (("new", "New model (graph + routing)", NEW), ("v2", "Earlier GNN V2 (3 seeds)", V2),
          ("lstm", "LSTM, tuned (3 seeds)", LSTM))
REFS = (("nwm", "NWM forecast", NWM), ("persistence", "Persistence", PERSIST), ("climatology", "Climatology", CLIM))
CLASSES = (("< 50 km²", 0, 50), ("50-500", 50, 500), ("500-5,000", 500, 5000), ("> 5,000", 5000, np.inf))
RAIN = (("nwm", "NWM forecast rain"), ("aorc", "Realized rain (perfect forecast)"))


# ---------------------------------------------------------------- data
def read(path):
    z = np.load(path, allow_pickle=True)
    return dict(time=pd.DatetimeIndex(z["time"]), truth=z["truth"], values=z["values"],
                methods=[str(m) for m in z["methods"]] if "methods" in z.files else None,
                stations=[str(s) for s in z["stations"]])


def nse(values, truth, min_count=100):
    """Per-station NSE [lead, station] of forecasts [issue, lead, station] (scored as in the benchmark)."""
    ok = np.isfinite(values) & np.isfinite(truth)
    n = ok.sum(0)
    o, f = np.where(ok, truth, 0.0).astype(np.float64), np.where(ok, values, 0.0).astype(np.float64)
    mean = o.sum(0) / np.maximum(n, 1)
    var = (o * o).sum(0) / np.maximum(n, 1) - mean**2
    mse = ((f - o) ** 2).sum(0) / np.maximum(n, 1)
    good = (n >= min_count) & (mean > 0) & (np.sqrt(np.maximum(var, 0)) >= 1e-3)
    with np.errstate(all="ignore"):
        return np.where(good, 1 - mse / var, np.nan)


def days15(x):
    """Mean over leads 24-120 h of per-station scores [lead, station]."""
    return x[DAYS].mean(axis=0)


def station_info(stations):
    """Drainage area (km2) and the NHDPlus index of each catalog gauge."""
    with xr.open_dataset(ROOT / "processed/gauges_public/gauges.nc") as d:
        names = list(d.gauge.values.astype(str))
        comid = d.comid.values.astype(np.int64)
        keep = pd.DatetimeIndex(d.time.values) < "2019-01-01"
        training = d.observed.values[keep]
    nodes = pd.read_parquet(ROOT / "processed/graph_candidate/nodes.parquet", columns=["comid", "totdasqkm"]).set_index("comid")
    pick = [names.index(s) for s in stations]
    with np.errstate(all="ignore"):
        p99 = np.nanquantile(training[:, pick], 0.99, axis=0)
    return pd.DataFrame(dict(station=stations, comid=comid[pick], area=nodes.totdasqkm.reindex(comid[pick]).to_numpy(), p99=p99))


class Data:
    """Forecasts of every model on the same issues and gauges, as [issue, lead, station]."""

    def __init__(self):
        a, t = read(M / "fair_benchmark/forecasts.npz"), read(M / "lstm_tuning_benchmark/forecasts.npz")
        g, u = read(EVAL / "forecasts_gauged.npz"), read(EVAL / "forecasts_ungauged.npz")
        if a["stations"] != t["stations"]:
            raise ValueError("Benchmarks scored different gauges")
        self.stations = a["stations"]
        self.times = a["time"].intersection(t["time"]).intersection(g["time"])
        ia, it, ig = (x["time"].get_indexer(self.times) for x in (a, t, g))
        sg = pd.Index(g["stations"]).get_indexer(self.stations)
        truth = np.where(np.isfinite(a["truth"][ia]) & np.isfinite(g["truth"][ig][..., sg]), a["truth"][ia], np.nan)
        self.truth = truth
        self.info = station_info(self.stations)

        def clean(x):
            return np.where(np.isfinite(truth), x, np.nan).astype(np.float32)

        mi = {m: i for i, m in enumerate(a["methods"])}
        ti = {m: i for i, m in enumerate(t["methods"])}
        self.gauged = {
            "new": {r: [clean(g["values"][ig][:, k][..., sg])] for k, r in enumerate(("nwm", "aorc"))},
            "v2": {r: [clean(a["values"][ia][:, mi[f"graph_s{s}+{r}"]]) for s in (1, 2, 3)] for r, _ in RAIN},
            "lstm": {r: [clean(t["values"][it][:, ti[f"graph_like_s{s}+{r}"]]) for s in (1, 2, 3)] for r, _ in RAIN},
        }
        self.refs = {name: clean(g["values"][ig][:, 2 + k][..., sg]) for k, name in enumerate(("nwm", "persistence", "climatology"))}
        # Held-out gauges: the new model's fold models, the earlier graph model and LSTM fold models.
        tu = u["truth"][u["time"].get_indexer(self.times)][..., sg]
        self.held = np.isfinite(tu).any(axis=(0, 1))
        self.truth_u = np.where(np.isfinite(tu) & np.isfinite(truth), truth, np.nan)
        iu = u["time"].get_indexer(self.times)

        def clean_u(x):
            return np.where(np.isfinite(self.truth_u), x, np.nan).astype(np.float32)

        old = {("v2", r): np.full_like(truth, np.nan) for r, _ in RAIN}
        old.update({("lstm", f"{r}{s}"): np.full_like(truth, np.nan) for r, _ in RAIN for s in (1, 2, 3)})
        for k in range(1, 6):
            d = read(M / f"heldout_benchmark_f{k}/forecasts.npz")
            idx = d["time"].get_indexer(self.times)
            cols = pd.Index(self.stations).get_indexer(d["stations"])
            names = {m: i for i, m in enumerate(d["methods"])}
            for r, _ in RAIN:
                old[("v2", r)][..., cols] = d["values"][idx][:, names[f"graph+{r}"]]
                for s in (1, 2, 3):
                    old[("lstm", f"{r}{s}")][..., cols] = d["values"][idx][:, names[f"lstm_s{s}+{r}"]]
        self.ungauged = {
            "new": {r: [clean_u(u["values"][iu][:, k][..., sg])] for k, r in enumerate(("nwm", "aorc"))},
            "v2": {r: [clean_u(old[("v2", r)])] for r, _ in RAIN},
            "lstm": {r: [clean_u(old[("lstm", f"{r}{s}")]) for s in (1, 2, 3)] for r, _ in RAIN},
        }
        self.refs_u = {k: clean_u(v) for k, v in self.refs.items()}
        self.gauged_u = {m: {r: [clean_u(x) for x in v] for r, v in d.items()} for m, d in self.gauged.items()}

    def score(self, arrays, truth, tmask=None):
        """Seed-wise per-station NSE [seed][lead, station] for a list of forecast arrays."""
        keep = slice(None) if tmask is None else tmask
        return [nse(x[keep], truth[keep]) for x in arrays]

    def median_curve(self, arrays, truth, stations=None, tmask=None):
        """Median over stations of NSE by lead: (mean over seeds, min, max)."""
        runs = np.array([np.nanmedian(s if stations is None else s[:, stations], axis=1) for s in self.score(arrays, truth, tmask)])
        return runs.mean(0), runs.min(0), runs.max(0)


# ---------------------------------------------------------------- helpers
def lead_axis(ax):
    ax.set_xticks([6, 24, 72, 120, 168, 240])
    ax.set_xlim(6, 240)
    ax.set_xlabel("Lead time (h)")


def save(fig, out, name):
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    print("wrote", name, flush=True)


def bootstrap_median(x, rng, n=1000):
    """95% interval of the median over stations of x [station]."""
    x = x[np.isfinite(x)]
    draws = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return np.percentile(draws, [2.5, 97.5])


def drawn(ax, lines, d, rain, truth, stations=None, band=True):
    """Median NSE curves of the models and references on one panel."""
    for key, label, color in MODELS:
        mean, lo, hi = d.median_curve(lines[key][rain], truth, stations)
        ax.plot(LEADS, mean, color=color, lw=2.6 if key == "new" else 2, label=label)
        if band and len(lines[key][rain]) > 1:
            ax.fill_between(LEADS, lo, hi, color=color, alpha=0.18, lw=0)


# ---------------------------------------------------------------- gauged
def gauged_by_lead(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True)
    for ax, (rain, title) in zip(axes, RAIN):
        drawn(ax, d.gauged, d, rain, d.truth)
        for name, label, color in REFS:
            ax.plot(LEADS, np.nanmedian(nse(d.refs[name], d.truth), axis=1), color=color, lw=1.8, ls="--" if name == "persistence" else "-", label=label)
        ax.set_title(f"{title}, 2020-2025")
        ax.set_ylim(-0.6, 1)
        lead_axis(ax)
    axes[0].set_ylabel("Median NSE over gauges")
    axes[0].legend(loc="lower left", **{**LEGEND, "fontsize": 10})
    fig.suptitle("Gauged: skill at the CAMELS-H gauges against USGS (band: seed range)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, out, "g1_gauged_nse_by_lead.png")


def gauged_summary(d, out):
    rows = [(k, l, c) for k, l, c in MODELS] + [(k, l, c) for k, l, c in REFS]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    table = []
    for i, (key, label, color) in enumerate(rows):
        for rain, marker, dy in (("nwm", "o", 0.13), ("aorc", "^", -0.13)):
            if key in ("nwm", "persistence", "climatology"):
                if rain == "aorc":
                    continue
                runs = [np.nanmedian(nse(d.refs[key], d.truth)[DAYS], axis=1).mean()]
            else:
                runs = [np.nanmean(np.nanmedian(s[DAYS], axis=1)) for s in d.score(d.gauged[key][rain], d.truth)]
            ax.scatter(np.mean(runs), len(rows) - 1 - i + dy, color=color, marker=marker, s=90, zorder=3)
            ax.plot([min(runs), max(runs)], [len(rows) - 1 - i + dy] * 2, color=color, lw=2)
            ax.text(np.mean(runs) + 0.012, len(rows) - 1 - i + dy, f"{np.mean(runs):.3f}", va="center", fontsize=9, color=INK)
            table.append(dict(model=label, rain=rain, days_1_5=np.mean(runs)))
    ax.set_yticks(range(len(rows)), [r[1] for r in rows][::-1])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Mean of median NSE at 24, 48, 72, 96 and 120 h")
    ax.set_title("Days 1-5 skill at gauges (circle: NWM rain, triangle: realized rain)")
    save(fig, out, "g2_gauged_days_1_5.png")
    pd.DataFrame(table).to_csv(out / "g2_gauged_days_1_5.csv", index=False)


def gauged_gain(d, out):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), sharey=True)
    for ax, (rain, title) in zip(axes, RAIN):
        new = np.mean(d.score(d.gauged["new"][rain], d.truth), axis=0)
        for other, label, color in (("v2", "New minus earlier GNN V2", V2), ("lstm", "New minus tuned LSTM", LSTM)):
            gain = new - np.mean(d.score(d.gauged[other][rain], d.truth), axis=0)
            mid = np.nanmedian(gain, axis=1)
            band = np.array([bootstrap_median(gain[i], rng) for i in range(40)])
            ax.plot(LEADS, mid, color=color, lw=2, label=label)
            ax.fill_between(LEADS, band[:, 0], band[:, 1], color=color, alpha=0.2, lw=0)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(title)
        lead_axis(ax)
    axes[0].set_ylabel("NSE gain per gauge (median, 95% bootstrap over gauges)")
    axes[0].legend(**{**LEGEND, "fontsize": 10})
    fig.tight_layout()
    save(fig, out, "g3_gauged_gain_by_lead.png")


def scatter_panels(d, lines, truth, stations, out, name, title):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10))
    for row, lead in enumerate((24, 72)):
        i = lead // 6 - 1
        new = np.mean(d.score(lines["new"]["nwm"], truth), axis=0)[i][stations]
        for col, (other, label, color) in enumerate((("v2", "Earlier GNN V2", V2), ("lstm", "Tuned LSTM", LSTM))):
            y = np.mean(d.score(lines[other]["nwm"], truth), axis=0)[i][stations]
            ok = np.isfinite(new) & np.isfinite(y)
            ax = axes[row, col]
            ax.scatter(y[ok], np.clip(new[ok], -0.5, 1), s=30, color=NEW, edgecolor=INK, lw=0.4)
            ax.plot([-0.5, 1], [-0.5, 1], color=MUTED, lw=0.8)
            ax.set_xlim(-0.5, 1)
            ax.set_ylim(-0.5, 1)
            ax.set_xlabel(f"{label} NSE")
            ax.set_ylabel("New model NSE")
            ax.set_title(f"{lead} h: new model above for {100 * np.mean(new[ok] > y[ok]):.0f}% of {ok.sum()} gauges")
    fig.suptitle(title + " (NWM rain; values below -0.5 shown at -0.5)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, out, name)


def gauged_by_year_season(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), sharey=True)
    groups = (("Year", [(str(y), d.times.year == y) for y in range(2020, 2026)]),
              ("Season", [(n, np.isin(d.times.month, m)) for n, m in (("DJF", [12, 1, 2]), ("MAM", [3, 4, 5]), ("JJA", [6, 7, 8]), ("SON", [9, 10, 11]))]))
    for ax, (title, parts) in zip(axes, groups):
        width = 0.2
        for j, (key, label, color) in enumerate(MODELS + (("nwm", "NWM forecast", NWM),)):
            heights = []
            for _, tm in parts:
                if key == "nwm":
                    s = [nse(d.refs["nwm"][tm], d.truth[tm])]
                else:
                    s = d.score(d.gauged[key]["nwm"], d.truth, tm)
                heights.append(np.mean([np.nanmedian(x[DAYS], axis=1).mean() for x in s]))
            ax.bar(np.arange(len(parts)) + (j - 1.5) * width, heights, width, color=color, label=label)
        ax.set_xticks(range(len(parts)), [p[0] for p in parts])
        ax.set_title(f"By {title.lower()}")
    axes[0].set_ylabel("Mean of median NSE, 24-120 h")
    axes[0].legend(**{**LEGEND, "fontsize": 9}, loc="lower left")
    fig.tight_layout()
    save(fig, out, "g6_gauged_by_year_and_season.png")


def gauged_vs_area(d, out):
    area = d.info.area.to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    new = days15(np.mean(d.score(d.gauged["new"]["nwm"], d.truth), axis=0))
    for other, label, color in (("v2", "V2", V2), ("lstm", "tuned LSTM", LSTM)):
        gain = new - days15(np.mean(d.score(d.gauged[other]["nwm"], d.truth), axis=0))
        ok = np.isfinite(gain) & np.isfinite(area)
        ax.scatter(area[ok], gain[ok], s=26, color=color, alpha=0.6, label=f"new minus {label}")
        order = np.argsort(area[ok])
        roll = pd.Series(gain[ok][order]).rolling(15, center=True, min_periods=5).median()
        ax.plot(area[ok][order], roll, color=color, lw=2.4)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("Drainage area (km²)")
    ax.set_ylabel("Days 1-5 NSE gain per gauge")
    ax.set_title("Where the new model gains or loses at gauges")
    ax.legend(**{**LEGEND, "fontsize": 10})
    save(fig, out, "g7_gauged_gain_vs_area.png")


def contingency(forecast, truth, threshold):
    """Hits, misses and false alarms per lead of [issue, lead, station] forecasts."""
    ok = np.isfinite(forecast) & np.isfinite(truth)
    obs, fc = (truth >= threshold) & ok, (forecast >= threshold) & ok
    return (obs & fc).sum(axis=(0, 2)), (obs & ~fc).sum(axis=(0, 2)), (~obs & fc & ok).sum(axis=(0, 2))


def flood_skill(d, lines, refs, truth, stations, out, name, title):
    thr = d.info.p99.to_numpy()[stations][None, None]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    entries = [(key, label, color, lines[key]["nwm"]) for key, label, color in MODELS] + \
              [(k, l, c, [refs[k]]) for k, l, c in REFS if k != "climatology"]
    for key, label, color, arrays in entries:
        csi, pod, far = [], [], []
        for x in arrays:
            h, m, f = contingency(x[..., stations], truth[..., stations], thr)
            csi.append(h / np.maximum(h + m + f, 1)); pod.append(h / np.maximum(h + m, 1)); far.append(f / np.maximum(h + f, 1))
        for ax, v in zip(axes, (csi, pod, far)):
            ax.plot(LEADS, np.mean(v, axis=0), color=color, lw=2.6 if key == "new" else 2, ls="--" if key == "persistence" else "-", label=label)
    for ax, t in zip(axes, ("Critical success index", "Probability of detection", "False alarm ratio (lower is better)")):
        ax.set_title(t)
        ax.set_ylim(0, 1)
        lead_axis(ax)
    axes[0].legend(**{**LEGEND, "fontsize": 9})
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, out, name)


def flood_peaks(d, out):
    """Observed peaks above the training p99 within leads 12-120 h: timing and size (as the earlier figures)."""
    truth, thr = d.truth, d.info.p99.to_numpy()
    peak_lead = np.nanargmax(np.where(np.isfinite(truth), truth, -np.inf), axis=1)
    peak = np.nanmax(truth, axis=1)
    complete = np.isfinite(truth).sum(axis=1) >= 36
    event = complete & (peak >= thr[None]) & (peak_lead >= 1) & (peak_lead <= 19)
    issue, station = np.nonzero(event)
    day = (peak_lead[issue, station] * 6 + 5) // 24 + 1
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    entries = [(k, l, c, d.gauged[k]["nwm"]) for k, l, c in MODELS] + [("nwm", "NWM forecast", NWM, [d.refs["nwm"]])]
    table = []
    for key, label, color, arrays in entries:
        timing, size = [], []
        for x in arrays:
            v = x[issue, :, station]
            ok = np.isfinite(v).all(axis=1)
            lead = np.argmax(np.where(ok[:, None], v, -np.inf), axis=1)
            timing.append(np.where(ok, np.abs(lead - peak_lead[issue, station]) * 6.0, np.nan))
            size.append(np.where(ok, v.max(axis=1) / peak[issue, station], np.nan))
        timing, size = np.nanmean(timing, axis=0), np.nanmean(size, axis=0)
        t = [np.nanmean(timing[day == k]) for k in range(1, 6)]
        s = [np.nanmedian(size[day == k]) for k in range(1, 6)]
        axes[0].plot(range(1, 6), t, color=color, marker="o", ms=8, lw=2.6 if key == "new" else 2, label=label)
        axes[1].plot(range(1, 6), s, color=color, marker="o", ms=8, lw=2.6 if key == "new" else 2, label=label)
        table.append(dict(model=label, **{f"timing_h_day{k + 1}": t[k] for k in range(5)}, **{f"peak_ratio_day{k + 1}": s[k] for k in range(5)}))
    axes[1].axhline(1, color=INK, lw=1)
    axes[0].set_ylabel("Mean |peak timing error| (h)")
    axes[1].set_ylabel("Median forecast peak / observed peak")
    for ax in axes:
        ax.set_xlabel("Day of the observed peak after issue")
        ax.set_xticks(range(1, 6))
    axes[0].legend(**{**LEGEND, "fontsize": 10})
    fig.suptitle(f"Flood peaks above each gauge's training 99th percentile, NWM rain ({len(issue):,} issue-gauge forecasts)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, out, "g9_flood_peaks.png")
    pd.DataFrame(table).to_csv(out / "g9_flood_peaks.csv", index=False)


def hydrographs(d, out):
    events = (("03455000", "2024-09-26"), ("03453500", "2020-02-05"), ("03451500", "2024-01-08"))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (station, day) in zip(axes, events):
        i, s = d.times.get_loc(pd.Timestamp(day)), d.stations.index(station)
        ax.plot(LEADS, d.truth[i, :, s], color="black", lw=2.8, label="Observed (USGS)")
        for key, label, color in MODELS:
            ax.plot(LEADS, d.gauged[key]["nwm"][0][i, :, s], color=color, lw=2.2 if key == "new" else 1.6, label=label.replace(" (3 seeds)", ", seed 1") if key != "new" else label)
        ax.plot(LEADS, d.refs["nwm"][i, :, s], color=NWM, lw=1.6, ls="--", label="NWM forecast")
        area = d.info.area[s]
        ax.set_title(f"{station}: {area:,.0f} km², issued {day} 00Z")
        lead_axis(ax)
    axes[0].set_ylabel("Flow (m³/s)")
    axes[0].legend(**{**LEGEND, "fontsize": 9})
    fig.tight_layout()
    save(fig, out, "g10_hydrographs.png")


# ---------------------------------------------------------------- ungauged
def ungauged_by_lead(d, out):
    h = np.flatnonzero(d.held)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True)
    for ax, (rain, title) in zip(axes, RAIN):
        drawn(ax, d.ungauged, d, rain, d.truth_u, h)
        for key, label, color in MODELS:
            mean, _, _ = d.median_curve(d.gauged_u[key][rain], d.truth_u, h)
            ax.plot(LEADS, mean, color=color, lw=1.6, ls=":")
        ax.plot(LEADS, np.nanmedian(nse(d.refs_u["nwm"], d.truth_u), axis=1), color=NWM, lw=1.8, label="NWM forecast (may use these gauges)")
        ax.set_title(f"{title}, {len(h)} held-out gauges")
        ax.set_ylim(-0.4, 1)
        lead_axis(ax)
    axes[0].set_ylabel("Median NSE over held-out gauges")
    axes[0].legend(loc="lower left", **{**LEGEND, "fontsize": 10})
    fig.suptitle("Ungauged: forecasts at gauges whose flow the model never saw (dotted: same gauges when their flow was used)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, out, "u1_ungauged_by_lead.png")


def skill_lost(d, out):
    h = np.flatnonzero(d.held)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for key, label, color in MODELS:
        g = np.mean(d.score(d.gauged_u[key]["nwm"], d.truth_u), axis=0)[:, h]
        u = np.mean(d.score(d.ungauged[key]["nwm"], d.truth_u), axis=0)[:, h]
        ax.plot(LEADS, np.nanmedian(g - u, axis=1), color=color, lw=2.6 if key == "new" else 2, label=label)
    ax.set_ylabel("NSE lost without the gauge's own flow (median)")
    ax.axhline(0, color=INK, lw=0.8)
    lead_axis(ax)
    ax.set_title("What each model loses when the gauge is ungauged (NWM rain)")
    ax.legend(**{**LEGEND, "fontsize": 10})
    save(fig, out, "u3_skill_lost.png")


# ---------------------------------------------------------------- every reach
class Reaches:
    def __init__(self):
        z = np.load(EVAL / "reach_metrics.npz", allow_pickle=True)
        self.comid, self.fold = z["comid"], z["reach_fold"]
        self.nse = {k[:-4]: z[k] for k in z.files if k.endswith("/nse")}
        nodes = pd.read_parquet(ROOT / "processed/graph_candidate/nodes.parquet", columns=["comid", "totdasqkm"]).set_index("comid")
        self.area = nodes.totdasqkm.reindex(self.comid).to_numpy()
        self.held = self.fold > 0
        self.series = (("new model+nwm", "New model, NWM rain", NEW, "-"), ("new model+aorc", "New model, realized rain", NEW, ":"),
                       ("earlier graph model+nwm", "Earlier GNN V2, NWM rain", V2, "-"), ("NWM forecast", "NWM forecast", NWM, "-"),
                       ("retrospective persistence (oracle)", "Retrospective persistence (oracle)", PERSIST, "--"))

    def median(self, key, mask):
        return np.nanmedian(self.nse[key][:, mask], axis=1)


def reach_by_lead(r, out):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True)
    for ax, (title, mask) in zip(axes, (("Held-out basins", r.held), ("All reaches", np.ones(len(r.comid), bool)))):
        for key, label, color, ls in r.series:
            ax.plot(LEADS, r.median(key, mask), color=color, ls=ls, lw=2.6 if key.startswith("new") else 2, label=label)
        ax.set_title(f"{title} ({mask.sum():,} reaches)")
        ax.set_ylim(-1, 1)
        ax.axhline(0, color=MUTED, lw=0.8)
        lead_axis(ax)
    axes[0].set_ylabel("Median NSE against the NWM retrospective")
    axes[0].legend(loc="lower left", **{**LEGEND, "fontsize": 9})
    fig.tight_layout()
    save(fig, out, "r1_reach_agreement_by_lead.png")


def reach_by_area(r, out):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5), sharey=True)
    keys = (("new model+nwm", "New model", NEW), ("earlier graph model+nwm", "Earlier GNN V2", V2), ("NWM forecast", "NWM forecast", NWM))
    rows = []
    for ax, lead in zip(axes, (24, 72)):
        i = lead // 6 - 1
        for j, (key, label, color) in enumerate(keys):
            vals = [np.nanmedian(r.nse[key][i][r.held & (r.area >= lo) & (r.area < hi)]) for _, lo, hi in CLASSES]
            bars = ax.bar(np.arange(4) + (j - 1) * 0.27, vals, 0.27, color=color, label=label)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.09), f"{v:.2f}", ha="center", fontsize=8)
            rows += [dict(lead_h=lead, model=label, area=c[0], median_nse=v) for c, v in zip(CLASSES, vals)]
        ax.set_xticks(range(4), [c[0] for c in CLASSES])
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(f"{lead} h, held-out basins")
        ax.set_xlabel("Drainage area")
    axes[0].set_ylabel("Median NSE against the NWM retrospective")
    axes[0].legend(**{**LEGEND, "fontsize": 10})
    fig.tight_layout()
    save(fig, out, "r2_reach_agreement_by_area.png")
    pd.DataFrame(rows).to_csv(out / "r2_reach_agreement_by_area.csv", index=False)


def reach_cdf(r, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, lead in zip(axes, (24, 72)):
        i = lead // 6 - 1
        for key, label, color, ls in r.series[:1] + r.series[2:4]:
            v = np.sort(np.clip(r.nse[key][i][r.held][np.isfinite(r.nse[key][i][r.held])], -1, 1))
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, ls=ls, lw=2.6 if key.startswith("new") else 2, label=label)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{lead} h, held-out basins")
        ax.set_xlabel("NSE against the retrospective (clipped to ±1)")
    axes[0].set_ylabel("Share of reaches below")
    axes[0].legend(**{**LEGEND, "fontsize": 10})
    fig.tight_layout()
    save(fig, out, "r3_reach_agreement_cdf.png")


def flowlines(r):
    import geopandas as gpd

    raw = ROOT / "raw"
    basin = gpd.read_file(raw / "geography/tennessee_huc2.geojson").to_crs(4326)
    lines = gpd.read_file(next((raw / "geography/nhdplus_v21/extracted").rglob("NHDFlowline.shp")), columns=["COMID", "geometry"]).rename(columns={"COMID": "comid"})
    lines = lines.merge(pd.DataFrame(dict(comid=r.comid, position=np.arange(len(r.comid)))), on="comid").to_crs(4326)
    segments = [np.asarray(g.geoms[0].coords if g.geom_type == "MultiLineString" else g.coords)[:, :2] for g in lines.geometry]
    return basin, lines.position.to_numpy(), segments


def reach_maps(r, out):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    basin, position, segments = flowlines(r)
    lead = 72
    i = lead // 6 - 1
    panels = (("r4_maps_72h_new_vs_nwm.png", (("New model vs retrospective, NSE", r.nse["new model+nwm"][i], "RdYlBu", TwoSlopeNorm(0.5, -1, 1)),
                                               ("NWM forecast vs retrospective, NSE", r.nse["NWM forecast"][i], "RdYlBu", TwoSlopeNorm(0.5, -1, 1)),
                                               ("Difference, new minus NWM forecast (purple: new closer)", r.nse["new model+nwm"][i] - r.nse["NWM forecast"][i], "PuOr", TwoSlopeNorm(0, -1, 1)))),
              ("r5_maps_72h_new_vs_v2.png", (("Earlier GNN V2 vs retrospective, NSE", r.nse["earlier graph model+nwm"][i], "RdYlBu", TwoSlopeNorm(0.5, -1, 1)),
                                              ("New model vs retrospective, NSE", r.nse["new model+nwm"][i], "RdYlBu", TwoSlopeNorm(0.5, -1, 1)),
                                              ("Difference, new minus V2 (purple: new closer)", r.nse["new model+nwm"][i] - r.nse["earlier graph model+nwm"][i], "PuOr", TwoSlopeNorm(0, -1, 1)))))
    for name, items in panels:
        fig, axes = plt.subplots(3, 1, figsize=(11, 17))
        for ax, (title, value, cmap, norm) in zip(axes, items):
            v = np.clip(value[position], -1, 1)
            ok = np.isfinite(v)
            ax.set_aspect(1 / np.cos(np.radians(36)))
            ax.grid(False)
            basin.boundary.plot(ax=ax, color=INK, lw=1.2, zorder=3)
            ax.add_collection(LineCollection([segments[k] for k in np.flatnonzero(~ok)], colors="#d9d8d3", linewidths=0.2))
            order = [k for k in np.argsort(np.where(ok, r.area[position], -1)) if ok[k]]
            width = 0.3 + 1.6 * np.clip(np.log10(np.maximum(r.area[position][order], 1)) / 4, 0, 1)
            lc = LineCollection([segments[k] for k in order], array=v[order], cmap=cmap, norm=norm, linewidths=width)
            ax.add_collection(lc)
            ax.autoscale()
            fig.colorbar(lc, ax=ax, shrink=0.7)
            ax.set_title(f"{title}, {lead} h, NWM rain", loc="left", color=INK)
        fig.tight_layout()
        save(fig, out, name)


def heldout_vs_usgs(d, r, out):
    """Held-out gauges against USGS, and whether agreement with the retrospective tracks it."""
    h = np.flatnonzero(d.held)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    ax = axes[0]
    for key, label, color in MODELS:
        mean, _, _ = d.median_curve(d.ungauged[key]["nwm"], d.truth_u, h)
        ax.plot(LEADS, mean, color=color, lw=2.6 if key == "new" else 2, label=f"{label.split(' (')[0]}, ungauged")
    ax.plot(LEADS, np.nanmedian(nse(d.refs_u["nwm"], d.truth_u), axis=1), color=NWM, lw=2, label="NWM forecast")
    retro = np.load(M / "retrospective_comparison/metrics.npz", allow_pickle=True)["gauge/retro/nse"]
    names = list(np.load(M / "retrospective_comparison/metrics.npz", allow_pickle=True)["gauge_name"].astype(str))
    cols = [names.index(d.stations[k]) for k in h]
    ax.plot(LEADS, np.nanmedian(retro[:, cols], axis=1), color="#444", lw=2, ls="--", label="NWM retrospective (realized rain)")
    ax.set_ylabel("Median NSE against USGS")
    ax.set_title(f"{len(h)} held-out gauges")
    ax.legend(**{**LEGEND, "fontsize": 9})
    lead_axis(ax)
    ax = axes[1]
    usgs = np.mean(d.score(d.ungauged["new"]["nwm"], d.truth_u), axis=0)[11][h]
    position = pd.Index(r.comid).get_indexer(d.info.comid.to_numpy()[h])
    agree = r.nse["new model+nwm"][11][position]
    ok = np.isfinite(usgs) & np.isfinite(agree)
    from scipy.stats import spearmanr

    rho = spearmanr(usgs[ok], agree[ok])[0]
    ax.scatter(usgs[ok], np.clip(agree[ok], -1, 1), s=30, color=NEW, edgecolor=INK, lw=0.4)
    ax.plot([-1, 1], [-1, 1], color=MUTED, lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("New model NSE against USGS (held-out gauge), 72 h")
    ax.set_ylabel("New model NSE against the retrospective, 72 h")
    ax.set_title(f"Does agreement with the retrospective track truth? Spearman {rho:.2f}")
    fig.tight_layout()
    save(fig, out, "r6_heldout_gauges_vs_usgs.png")


# ---------------------------------------------------------------- training and model
def training_curves(out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    a = pd.read_json(M / "hydro_A/validation.json")
    axes[0].plot(a["update"], a["nwm_loss"], color=NEW, marker="o", lw=2.4)
    axes[0].set_title("Stage A: validation loss against NWM (2019)")
    axes[0].set_xlabel("Update")
    axes[0].set_ylabel("Bounded NSE loss (lower is better)")
    for name, label, color in [("hydro_B", "all gauges", NEW)] + [(f"hydro_B_f{k}", f"fold {k}", c) for k, c in zip(range(1, 6), ("#2a78d6", "#008300", "#6250d6", "#eda100", "#e87ba4"))]:
        v = pd.read_json(M / name / "validation.json")
        axes[1].plot(v["update"], v["score"], marker="o", lw=2.4 if name == "hydro_B" else 1.4, color=color, label=label)
    axes[1].set_title("Stage B: validation score (mean of gauged, hidden-history, every-reach NSE)")
    axes[1].set_xlabel("Update")
    axes[1].legend(**{**LEGEND, "fontsize": 9})
    fig.tight_layout()
    save(fig, out, "t1_training_curves.png")


def routing_kernels(out):
    state = torch.load(M / "hydro_B/best.pt", map_location="cpu", weights_only=False)["model"]
    p = torch.softmax(state["router.logits"], dim=-1).numpy()  # [27 groups, 48 taps]
    delay = (p * np.arange(p.shape[1])).sum(axis=1) * 6  # hours
    bins = ("0-5", "5-15", "15-35", "35-75", "75-150", "150-300", "300-500", "500-800", "> 800")
    delay = delay.reshape(len(bins), 3)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for j, (label, color) in enumerate((("no lake on the path", NEW), ("1-10 lake reaches", V2), ("> 10 lake reaches", LSTM))):
        axes[0].plot(range(len(bins)), delay[:, j], marker="o", color=color, lw=2.2, label=label)
    axes[0].set_xticks(range(len(bins)), bins, fontsize=8)
    axes[0].set_xlabel("River distance upstream (km)")
    axes[0].set_ylabel("Learned mean travel time (h)")
    axes[0].set_title("Travel time grows with distance and lakes")
    axes[0].legend(**{**LEGEND, "fontsize": 9})
    for b, color in zip((1, 3, 5, 7), ("#2a78d6", "#008300", "#eb6834", "#6250d6")):
        axes[1].plot(np.arange(p.shape[1]) * 6, p[b * 3], color=color, lw=2, label=f"{bins[b]} km")
    axes[1].set_xlabel("Delay (h)")
    axes[1].set_ylabel("Share of the water")
    axes[1].set_title("Routing kernels (no lake): each sums to one")
    axes[1].legend(**{**LEGEND, "fontsize": 9})
    fig.tight_layout()
    save(fig, out, "t2_routing_kernels.png")


def scoreboard(d, r, out):
    h = np.flatnonzero(d.held)
    keys = (("New model", NEW), ("Earlier GNN V2", V2), ("LSTM, tuned", LSTM), ("NWM forecast", NWM))

    def days_mean(arrays, truth, stations=None):
        s = d.score(arrays, truth)
        return np.mean([np.nanmedian((x if stations is None else x[:, stations])[DAYS], axis=1).mean() for x in s])

    gauged = [days_mean(d.gauged[k]["nwm"], d.truth) for k in ("new", "v2", "lstm")] + [days_mean([d.refs["nwm"]], d.truth)]
    ungauged = [days_mean(d.ungauged[k]["nwm"], d.truth_u, h) for k in ("new", "v2", "lstm")] + [days_mean([d.refs_u["nwm"]], d.truth_u, h)]
    reach = [np.nanmedian(r.nse[k][11][r.held]) for k in ("new model+nwm", "earlier graph model+nwm")] + [np.nan, np.nanmedian(r.nse["NWM forecast"][11][r.held])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, vals, title in zip(axes, (gauged, ungauged, reach), ("Gauged: days 1-5 NSE vs USGS", "Ungauged: days 1-5 NSE vs USGS", "Every reach: 72 h NSE vs retrospective")):
        bars = ax.bar(range(4), vals, color=[c for _, c in keys])
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v / 2 if v < 0 else v + 0.01, f"{v:.3f}", ha="center", va="center" if v < 0 else "bottom", fontsize=10, color="white" if v < 0 else INK)
        ax.set_xticks(range(4), [k for k, _ in keys], fontsize=9)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(title)
    fig.suptitle("Three protocols, NWM forecast rain (every reach: held-out basins; no LSTM there)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, out, "s1_scoreboard.png")
    return gauged, ungauged, reach


def tables(d, r, out):
    """Markdown tables of the headline numbers."""
    def row(label, arrays, truth, stations=None):
        s = d.score(arrays, truth)
        v = np.mean([np.nanmedian((x if stations is None else x[:, stations]), axis=1) for x in s], axis=0)
        return f"| {label} | " + " | ".join(f"{v[h // 6 - 1]:.3f}" for h in (6, 24, 72, 120, 240)) + f" | {v[DAYS].mean():.3f} |"

    head = "| method | 6 h | 24 h | 72 h | 120 h | 240 h | days 1-5 |\n|---|---:|---:|---:|---:|---:|---:|"
    h = np.flatnonzero(d.held)
    lines = [f"# Graph-routing model against the earlier models, {len(d.times)} 00Z issues 2020-2025", "",
             f"## Gauged: {int(np.isfinite(nse(d.refs['nwm'], d.truth)[0]).sum())} CAMELS-H gauges against USGS (median NSE)", "", head]
    for rain, _ in RAIN:
        for key, label, _ in MODELS:
            lines.append(row(f"{label}, {rain}", d.gauged[key][rain], d.truth))
    for key, label, _ in REFS:
        lines.append(row(label, [d.refs[key]], d.truth))
    lines += ["", f"## Ungauged: {len(h)} held-out gauges (5 folds)", "", head]
    for rain, _ in RAIN:
        for key, label, _ in MODELS:
            lines.append(row(f"{label}, {rain}", d.ungauged[key][rain], d.truth_u, h))
    for key, label, _ in REFS:
        lines.append(row(f"{label} (may use these gauges)", [d.refs_u[key]], d.truth_u, h))
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("wrote report.md", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("/data/tva_data/figures/hydro"))
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    d, r = Data(), Reaches()
    h = np.flatnonzero(d.held)
    everyone = np.arange(len(d.stations))
    gauged_by_lead(d, out)
    gauged_summary(d, out)
    gauged_gain(d, out)
    scatter_panels(d, d.gauged, d.truth, everyone, out, "g4_gauged_scatter.png", "Per gauge: new model against earlier models")
    gauged_by_year_season(d, out)
    gauged_vs_area(d, out)
    flood_skill(d, d.gauged, d.refs, d.truth, everyone, out, "g8_flood_skill.png", "Gauged flood warnings: flow above each gauge's training 99th percentile, NWM rain")
    flood_peaks(d, out)
    hydrographs(d, out)
    ungauged_by_lead(d, out)
    scatter_panels(d, d.ungauged, d.truth_u, h, out, "u2_ungauged_scatter.png", "Per held-out gauge: new model against earlier models")
    skill_lost(d, out)
    flood_skill(d, d.ungauged, d.refs_u, d.truth_u, h, out, "u5_ungauged_flood_skill.png", "Ungauged flood warnings at held-out gauges, NWM rain")
    reach_by_lead(r, out)
    reach_by_area(r, out)
    reach_cdf(r, out)
    reach_maps(r, out)
    heldout_vs_usgs(d, r, out)
    training_curves(out)
    routing_kernels(out)
    scoreboard(d, r, out)
    tables(d, r, out)
    print(f"FIGURES COMPLETE: {out}", flush=True)


if __name__ == "__main__":
    main()
