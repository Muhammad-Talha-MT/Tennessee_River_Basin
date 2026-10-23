"""Was river routing helpful? Knockout checkpoints, figures and a report (routing_ablation.sh).

    python scripts/routing_analysis.py knockout --checkpoint best.pt --out knocked.pt
    python scripts/routing_analysis.py plots --out /data/tva_data/figures/routing_ablation

``knockout`` copies a whole-basin checkpoint with its routing block's output removed
(the block's zero-initialized projections set back to zero: the block then passes its
input through unchanged). It shows how much a trained model relies on routing.

``plots`` reads whichever benchmark runs exist: routing_benchmark (the ablation ladder),
scratch_benchmark_s1 (the full model, until the ladder is done) and fair_benchmark (the
LSTM and NWM). All share the issues, gauges and USGS observations, which is checked.
Seeds of one variant are averaged per gauge; ablations have one seed each, so the
spread of the full model's seeds is the yardstick for what counts as a difference.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402

ROOT = Path("/data/tva_data")
M = ROOT / "models"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tva_flood.data.river import DISTANCE_BINS, River  # noqa: E402

# Display order is the categorical order (reference palette slots 1-7, never cycled).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
# label, [(benchmark run, method prefix)], one entry per seed; the first source that
# exists is used (the full model falls back to scratch_benchmark_s1 before the ladder).
VARIANTS = [
    (
        "Full model",
        [
            [("routing_benchmark", "full_s1"), ("routing_benchmark", "full_s2")],
            [("scratch_benchmark_s1", "scratch_s1")],
        ],
    ),
    ("No routing block", [[("routing_benchmark", "no_routing")]]),
    ("No routing, no basin means", [[("routing_benchmark", "no_basin")]]),
    ("No river network", [[("routing_benchmark", "no_network")]]),
    ("Full, routing removed at test", [[("routing_benchmark", "knockout_s1")]]),
    ("LSTM (lumped)", [[("fair_benchmark", f"lstm_s{s}") for s in (1, 2, 3)]]),
    ("NWM", [[("fair_benchmark", "nwm")]]),
]
REFERENCES = ("nwm", "persistence", "climatology")  # scored without a rain variant
RAINS = {"nwm": "NWM forecast rain", "aorc": "Realized rain (perfect forecast)"}
LEADS = np.arange(1, 41) * 6


def knockout(a):
    saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    keys = [k for k in saved["model"] if k.startswith("routing_block.projections.")]
    if not keys:
        raise ValueError(f"{a.checkpoint} has no routing block")
    for k in keys:
        saved["model"][k] = torch.zeros_like(saved["model"][k])
    saved["config"] = dict(saved["config"], knockout="routing block output removed at test time")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, a.out)
    print(f"{a.out}: {len(keys)} routing tensors set to zero")


class Runs:
    """Per-station scores and saved forecasts of the benchmark runs, checked to align."""

    def __init__(self):
        self.tables, self.arrays, self.reference = {}, {}, None

    def exists(self, run):
        return (M / run / "report.md").exists()

    def table(self, run):
        if run not in self.tables:
            frame = pd.read_csv(M / run / "per_station.csv", dtype={"usgs_id": str})
            self.tables[run] = frame[frame.scored]
        return self.tables[run]

    def array(self, run):
        if run not in self.arrays:
            data = np.load(M / run / "forecasts.npz", allow_pickle=True)
            arrays = {k: data[k] for k in ("issue", "truth", "values", "methods", "stations")}
            if self.reference is None:
                self.reference = arrays
            elif not (
                np.array_equal(arrays["issue"], self.reference["issue"])
                and np.array_equal(arrays["stations"], self.reference["stations"])
            ):
                raise ValueError(f"{run} scored different issues or gauges")
            self.arrays[run] = arrays
        return self.arrays[run]

    def methods(self, sources, rain):
        """(run, method) per seed of one variant; reference forecasts have no rain variant."""
        return [(run, p if p in REFERENCES else f"{p}+{rain}") for run, p in sources]


def available(runs):
    """{label: (color, seed sources)} of the variants whose benchmark exists."""
    found = {}
    for (label, options), color in zip(VARIANTS, COLORS):
        for sources in options:
            if all(runs.exists(run) for run, _ in sources):
                found[label] = (color, sources)
                break
    return found


def per_gauge(runs, sources, rain, column="nse"):
    """[lead, station] score of each seed, then the seed mean per gauge."""
    frames = []
    for run, method in runs.methods(sources, rain):
        t = runs.table(run)
        t = t[t.method == method].pivot(index="lead_hours", columns="station", values=column)
        frames.append(t)
    return frames, sum(frames) / len(frames)


def style(ax, xlabel, ylabel):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.set_xlabel(xlabel, color=INK, fontsize=12)
    ax.set_ylabel(ylabel, color=INK, fontsize=12)


def legend(ax, **kw):
    ax.legend(frameon=True, fancybox=False, edgecolor="#c3c2b7", fontsize=11, **kw)


def lead_axis(ax):
    ax.set_xlim(6, 240)
    ax.set_xticks([6, 24, 48, 72, 96, 120, 168, 240])


def nse_by_lead(runs, variants, out, rows):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    for ax, (rain, title) in zip(axes, RAINS.items()):
        for label, (color, sources) in variants.items():
            if label == "NWM" and rain != "nwm":
                continue
            seeds, _ = per_gauge(runs, sources, rain)
            medians = np.array([f.median(axis=1).to_numpy() for f in seeds])
            mean = medians.mean(0)
            ax.plot(LEADS, mean, color=color, linewidth=2, label=label)
            if len(seeds) > 1:
                ax.fill_between(LEADS, medians.min(0), medians.max(0), color=color, alpha=0.18)
            rows.append(
                dict(
                    variant=label,
                    rain=rain,
                    seeds=len(seeds),
                    **{f"nse_{h}h": mean[h // 6 - 1] for h in (6, 24, 72, 120, 240)},
                    days_1_5=mean[[3, 7, 11, 15, 19]].mean(),  # 24-120 h, as the benchmark report
                )
            )
        ax.set_title(title, color=INK, fontsize=13, loc="left")
        ax.axhline(0, color="#c3c2b7", linewidth=1)
        style(ax, "Lead time (h)", "Median NSE over gauges" if rain == "nwm" else "")
        lead_axis(ax)
    axes[0].set_ylim(-0.4, 1.0)
    legend(axes[0], loc="lower left")
    fig.suptitle("Skill by lead time (band: seed range)", color=INK, fontsize=14, x=0.07, ha="left")
    fig.tight_layout()
    fig.savefig(out / "01_nse_by_lead.png", dpi=160)
    plt.close(fig)


def bootstrap_median(values, rng, n=1000):
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return np.nan, np.nan, np.nan
    picks = rng.choice(values, (n, len(values)))
    return np.median(values), *np.percentile(np.median(picks, axis=1), [2.5, 97.5])


def gain_by_lead(runs, variants, out, rows):
    """Median over gauges of NSE(full) - NSE(variant), with a gauge bootstrap."""
    rng = np.random.default_rng(0)
    _, full = variants["Full model"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    for ax, (rain, title) in zip(axes, RAINS.items()):
        _, base = per_gauge(runs, full, rain)
        for label, (color, sources) in variants.items():
            if label in ("Full model", "NWM"):  # NWM's gap is in 01; this is about our parts
                continue
            _, other = per_gauge(runs, sources, rain)
            diff = (base - other).dropna(axis=1, how="all")
            stats = np.array([bootstrap_median(diff.loc[h].to_numpy(), rng) for h in LEADS])
            ax.plot(LEADS, stats[:, 0], color=color, linewidth=2, label=label)
            ax.fill_between(LEADS, stats[:, 1], stats[:, 2], color=color, alpha=0.18)
            for h in (24, 72, 120):
                k = h // 6 - 1
                rows.append(
                    dict(
                        variant=label,
                        rain=rain,
                        lead_h=h,
                        median_gain=stats[k, 0],
                        low=stats[k, 1],
                        high=stats[k, 2],
                        gauges_better=float((diff.loc[h] > 0).mean()),
                    )
                )
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_title(title, color=INK, fontsize=13, loc="left")
        style(ax, "Lead time (h)", "NSE gain of the full model" if rain == "nwm" else "")
        lead_axis(ax)
    legend(axes[0], loc="upper left")
    fig.suptitle(
        "How much each part adds: full minus variant, median over gauges (band: 95% bootstrap)",
        color=INK,
        fontsize=14,
        x=0.07,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "02_gain_by_lead.png", dpi=160)
    plt.close(fig)


def gauge_info(stations):
    """Drainage area (NHDPlus) and catalog gauges upstream of each scored gauge."""
    prep = ROOT / "processed/pretraining_full"
    with np.load(prep / "graph.npz") as graph:
        ids = graph["feature_id"].copy()
        edges = np.column_stack([graph["source"], graph["target"]])
    with xr.open_dataset(ROOT / "processed/gauges_public/gauges.nc") as data:
        names = data.gauge.values.astype(str)
        comid = data.comid.values.astype(np.int64)
        has_data = np.isfinite(data.observed.values).any(axis=0)
    river = River.load(ROOT, ids, edges)
    outlet = pd.Index(ids).get_indexer(comid)
    nodes = pd.read_parquet(
        ROOT / "processed/graph_candidate/nodes.parquet", columns=["comid", "totdasqkm"]
    ).set_index("comid")
    rows = []
    for g, name in enumerate(names):
        mask = np.zeros(len(ids), dtype=bool)
        mask[outlet[g]] = True
        basin = river.closure(mask)
        inside = basin[outlet] & has_data
        inside[g] = False
        rows.append(
            dict(
                station=name,
                area_km2=float(nodes.totdasqkm.get(comid[g], np.nan)),
                upstream_gauges=int(inside.sum()),
                basin_reaches=int(basin.sum()),
            )
        )
    return pd.DataFrame(rows).set_index("station").loc[list(stations)]


def gain_vs_basin(runs, variants, info, out, versus):
    _, full = variants["Full model"]
    _, other_sources = variants[versus]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    for ax, h in zip(axes, (24, 72)):
        _, base = per_gauge(runs, full, "aorc")
        _, other = per_gauge(runs, other_sources, "aorc")
        diff = (base - other).loc[h].dropna()
        table = info.loc[diff.index].assign(gain=diff)
        nested = table.upstream_gauges > 0
        ax.scatter(
            table.area_km2[~nested],
            table.gain[~nested],
            s=40,
            color=COLORS[0],
            edgecolor="#fcfcfb",
            linewidth=1,
            label="No gauge upstream",
        )
        ax.scatter(
            table.area_km2[nested],
            table.gain[nested],
            s=46,
            marker="^",
            color=COLORS[1],
            edgecolor="#fcfcfb",
            linewidth=1,
            label="Gauged upstream",
        )
        bins = np.quantile(table.area_km2, np.linspace(0, 1, 5))
        mids, meds = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            part = table[(table.area_km2 >= lo) & (table.area_km2 <= hi)]
            mids.append(np.sqrt(lo * hi))
            meds.append(part.gain.median())
        ax.plot(mids, meds, color=INK, linewidth=2, label="Median by area quartile")
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xscale("log")
        ax.set_title(
            f"{h} h lead, realized rain: nested {table.gain[nested].median():+.3f}, "
            f"not nested {table.gain[~nested].median():+.3f} (medians)",
            color=INK,
            fontsize=12,
            loc="left",
        )
        style(ax, "Drainage area (km²)", f"NSE, full minus {versus}" if h == 24 else "")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="lower center",
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="#c3c2b7",
        fontsize=11,
    )
    fig.suptitle(
        f"Where the full model gains over {versus}, per gauge",
        color=INK,
        fontsize=14,
        x=0.07,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    tag = versus.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    fig.savefig(out / f"03_gain_vs_basin_{tag}.png", dpi=160)
    plt.close(fig)


def thresholds(stations, quantile):
    with xr.open_dataset(ROOT / "processed/gauges_public/gauges.nc") as data:
        names = list(data.gauge.values.astype(str))
        keep = pd.DatetimeIndex(data.time.values) < "2019-01-01"
        values = data.observed.values[keep][:, [names.index(s) for s in stations]]
    with np.errstate(all="ignore"):
        return np.nanquantile(values, quantile, axis=0)


def series(runs, sources, rain):
    """Forecasts [issue, lead, station] of each seed of one variant."""
    out = []
    for run, method in runs.methods(sources, rain):
        arrays = runs.array(run)
        out.append(arrays["values"][:, list(arrays["methods"]).index(method)])
    return out


def peaks(runs, variants, out, rows, name="04_peaks.png"):
    """Observed peaks above the training p99 within leads 12-120 h: timing and size."""
    truth = runs.array("fair_benchmark")["truth"]
    stations = runs.reference["stations"]
    flood = thresholds(stations, 0.99)
    peak_lead = np.nanargmax(np.nan_to_num(truth, nan=-np.inf), axis=1)  # [issue, station]
    peak = np.nanmax(truth, axis=1)
    complete = np.isfinite(truth).sum(axis=1) >= 36
    event = complete & (peak >= flood[None]) & (peak_lead >= 1) & (peak_lead <= 19)
    issue, station = np.nonzero(event)
    day = (peak_lead[issue, station] * 6 + 6 - 1) // 24 + 1  # observed peak day after issue
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for label, (color, sources) in variants.items():
        timing, size = [], []
        for values in series(runs, sources, "nwm"):
            v = values[issue, :, station]
            ok = np.isfinite(v).all(axis=1)
            lead = np.argmax(np.where(ok[:, None], v, -np.inf), axis=1)
            error = np.abs(lead - peak_lead[issue, station]) * 6.0  # per seed, then averaged
            timing.append(np.where(ok, error, np.nan))
            size.append(np.where(ok, v.max(axis=1) / peak[issue, station], np.nan))
        timing, size = np.nanmean(timing, axis=0), np.nanmean(size, axis=0)
        days = np.arange(1, 6)
        t = [np.nanmean(timing[day == d]) for d in days]
        hit = [np.nanmean(timing[day == d] <= 6) for d in days]
        s = [np.nanmedian(size[day == d]) for d in days]
        axes[0].plot(days, t, color=color, linewidth=2, marker="o", markersize=8, label=label)
        axes[1].plot(days, s, color=color, linewidth=2, marker="o", markersize=8, label=label)
        rows.append(
            dict(
                variant=label,
                forecasts=int(np.isfinite(timing).sum()),
                **{f"mean_abs_timing_h_day{d}": t[d - 1] for d in days},
                **{f"within_6h_day{d}": hit[d - 1] for d in days},
                **{f"peak_ratio_day{d}": s[d - 1] for d in days},
            )
        )
    axes[1].axhline(1, color=INK, linewidth=1)
    style(axes[0], "Day of the observed peak after issue", "Mean |peak timing error| (h)")
    style(axes[1], "Day of the observed peak after issue", "Median forecast peak / observed peak")
    for ax in axes:
        ax.set_xticks(range(1, 6))
    legend(axes[0], loc="upper left")
    fig.suptitle(
        f"Flood peaks above each gauge's training 99th percentile, NWM rain ({len(issue):,} "
        "issue-gauge forecasts; one event counts once per issue)",
        color=INK,
        fontsize=13,
        x=0.07,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / name, dpi=160)
    plt.close(fig)


def variability(runs, variants, out):
    truth = runs.array("fair_benchmark")["truth"]
    high = thresholds(runs.reference["stations"], 0.9)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for label, (color, sources) in variants.items():
        rain = "nwm" if label == "NWM" else "aorc"
        _, alpha = per_gauge(runs, sources, rain, "alpha")
        axes[0].plot(LEADS, alpha.median(axis=1), color=color, linewidth=2, label=label)
        ratios = []
        for values in series(runs, sources, rain):
            hi = (truth > high[None, None]) & np.isfinite(values)
            ratios.append(
                np.nansum(np.where(hi, values, 0), axis=(0, 2))
                / np.nansum(np.where(hi, truth, 0), axis=(0, 2))
            )
        axes[1].plot(LEADS, np.mean(ratios, axis=0), color=color, linewidth=2, label=label)
    for ax in axes:
        ax.axhline(1, color=INK, linewidth=1)
        lead_axis(ax)
    style(axes[0], "Lead time (h)", "Median variability ratio (forecast / observed std)")
    style(axes[1], "Lead time (h)", "Forecast / observed volume above the 90th percentile")
    legend(axes[0], loc="lower left")
    fig.suptitle(
        "Damping: do forecasts keep the observed variability? (realized rain; NWM as issued)",
        color=INK,
        fontsize=14,
        x=0.07,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "05_variability.png", dpi=160)
    plt.close(fig)


def kernels(out, rows):
    """Learned travel time and weight of each river-distance bin, per trained full model."""
    models = {
        "Scratch s1": M / "scratch_graph_s1/best.pt",
        "Scratch s2": M / "scratch_graph_s2/best.pt",
        **{f"Fine-tuned s{s}": M / f"fair_graph_s{s}/best.pt" for s in (1, 2, 3)},
    }
    labels = [
        f"{int(lo)}-{int(hi)}" if np.isfinite(hi) else f">{int(lo)}"
        for lo, hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    x = np.arange(len(labels))
    for (name, path), color in zip(models.items(), COLORS):
        if not path.exists():
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)["model"]
        kernel = state["routing_block.kernel"].float().abs()[:, :, 0]  # [bin, channel, lag+1]
        lags = kernel.shape[-1] - 1
        delay = (kernel * torch.arange(lags, -1, -1)).sum(-1) / kernel.sum(-1).clamp_min(1e-12)
        weight = torch.stack(
            [
                state[f"routing_block.projections.{b}.weight"].float().norm(dim=0)
                for b in range(len(labels))
            ]
        )  # [bin, input channel]
        hours = 6 * (weight * delay).sum(1) / weight.sum(1)
        share = weight.sum(1) / weight.sum()
        axes[0].plot(x, share, color=color, linewidth=2, marker="o", markersize=8, label=name)
        axes[1].plot(x, hours, color=color, linewidth=2, marker="o", markersize=8, label=name)
        rows.append(
            dict(
                model=name,
                **{f"share_{lab}": float(v) for lab, v in zip(labels, share)},
                **{f"delay_h_{lab}": float(v) for lab, v in zip(labels, hours)},
            )
        )
    for ax in axes:
        ax.set_xticks(x, labels)
    style(axes[0], "River distance upstream (km)", "Share of the routing block's weight")
    style(axes[1], "River distance upstream (km)", "Learned travel time (h, weight-averaged)")
    legend(axes[1], loc="upper left")
    fig.suptitle(
        "What the routing block learned: does travel time grow with distance?",
        color=INK,
        fontsize=14,
        x=0.07,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "06_routing_kernels.png", dpi=160)
    plt.close(fig)


def hydrographs(runs, variants, info, out, name="07_hydrographs.png"):
    """The largest observed events at the most nested gauges, issued 2 days before the peak."""
    arrays = runs.array("fair_benchmark")
    truth, stations = arrays["truth"], list(arrays["stations"])
    times = pd.DatetimeIndex(np.load(M / "fair_benchmark/forecasts.npz")["time"])
    ranked = info.sort_values(["upstream_gauges", "area_km2"], ascending=False).index
    chosen, used = [], set()
    for name in ranked:  # three gauges, three different storms, a complete observed series
        g = stations.index(name)
        complete = np.isfinite(truth[:, :, g]).all(axis=1)
        at_lead = np.where(complete, truth[:, 7, g], np.nan)  # flow 48 h after issue
        for i in np.argsort(-np.nan_to_num(at_lead, nan=-1))[:50]:
            if complete[i] and all(abs(int(i) - u) > 10 for u in used):
                chosen.append((name, int(i)))
                used.add(int(i))
                break
        if len(chosen) == 3:
            break
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    for ax, (name, i) in zip(axes, chosen):
        g = stations.index(name)
        ax.plot(LEADS, truth[i, :, g], color=INK, linewidth=2.5, label="Observed (USGS)")
        for label, (color, sources) in variants.items():
            values = np.mean([s[i, :, g] for s in series(runs, sources, "nwm")], axis=0)
            ax.plot(LEADS, values, color=color, linewidth=2, label=label)
        ax.set_title(
            f"{name}: {info.area_km2[name]:,.0f} km², {info.upstream_gauges[name]} gauges "
            f"upstream\nissued {times[i]:%Y-%m-%d %H}Z, NWM rain",
            color=INK,
            fontsize=11,
            loc="left",
        )
        style(ax, "Lead time (h)", "Flow (m³/s)" if ax is axes[0] else "")
        lead_axis(ax)
    legend(axes[0], loc="upper right")
    fig.suptitle(
        "Largest events at the most nested gauges", color=INK, fontsize=14, x=0.05, ha="left"
    )
    fig.tight_layout()
    fig.savefig(out / name, dpi=160)
    plt.close(fig)


def table_md(frame, digits=3):
    frame = frame.copy()
    for c in frame.columns:
        if frame[c].dtype.kind == "f":
            frame[c] = frame[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    head = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.to_numpy()]
    return "\n".join([head, rule, *body])


def plots(a):
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    runs = Runs()
    if not runs.exists("fair_benchmark"):
        raise SystemExit("fair_benchmark is missing; run fair_comparison.sh first")
    runs.array("fair_benchmark")  # the reference issues and gauges
    variants = available(runs)
    if "Full model" not in variants:
        raise SystemExit("No benchmark of the full from-scratch model yet")
    for label, (_, sources) in variants.items():
        for run, _ in sources:
            runs.array(run)  # checks alignment
    print("Variants:", ", ".join(f"{k} ({len(v[1])} seed(s))" for k, v in variants.items()))
    info = gauge_info(runs.reference["stations"])
    skill, gain, peak_rows, kernel_rows = [], [], [], []
    nse_by_lead(runs, variants, out, skill)
    gain_by_lead(runs, variants, out, gain)
    for versus in ("No routing block", "LSTM (lumped)"):
        if versus in variants:
            gain_vs_basin(runs, variants, info, out, versus)
    peaks(runs, variants, out, peak_rows)
    variability(runs, variants, out)
    kernels(out, kernel_rows)
    hydrographs(runs, variants, info, out)
    tables = dict(
        skill=pd.DataFrame(skill),
        gain=pd.DataFrame(gain),
        peaks=pd.DataFrame(peak_rows),
        kernels=pd.DataFrame(kernel_rows),
    )
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    info.to_csv(out / "gauges.csv")
    lines = [
        "# Was river routing helpful?",
        "",
        f"Variants: {', '.join(variants)}. Issues 2020-2025 at 00Z, CAMELS-H gauges, USGS "
        "observations; ablations are single seeds, so compare their gaps with the full "
        "model's seed spread (shaded in 01).",
        "",
        "## Median NSE by lead",
        "",
        table_md(tables["skill"]),
        "",
        "## NSE gain of the full model (median over gauges, 95% bootstrap over gauges)",
        "",
        table_md(tables["gain"]),
        "",
        "## Flood peaks (observed above training p99), NWM rain",
        "",
        table_md(tables["peaks"], 2),
        "",
        "## Learned routing per distance bin (km)",
        "",
        table_md(tables["kernels"], 2),
        "",
        "Figures: 01_nse_by_lead, 02_gain_by_lead, 03_gain_vs_basin_*, 04_peaks, "
        "05_variability, 06_routing_kernels, 07_hydrographs.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    k = sub.add_parser("knockout")
    k.add_argument("--checkpoint", type=Path, required=True)
    k.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("plots")
    s.add_argument("--out", type=Path, default=ROOT / "figures/routing_ablation")
    a = p.parse_args()
    knockout(a) if a.command == "knockout" else plots(a)


if __name__ == "__main__":
    main()
