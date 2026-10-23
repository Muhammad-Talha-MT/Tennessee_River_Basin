"""The V1/V4 figure set, redone for the public comparison: GNN vs LSTM, NWM and simple baselines.

    python scripts/comparison_figures.py --out /data/tva_data/figures/gnn_vs_baselines

Reads fair_benchmark (GNN fine-tuned from the NWM-trained stage 1, 3 seeds; the original
LSTM, 3 seeds; NWM; persistence; climatology), lstm_tuning_benchmark (the tuned LSTM, the
headline baseline), scratch_benchmark_s1 (GNN trained from scratch,
1 seed) and their evaluate-events folders. Same 00Z issues 2020-2025, CAMELS-H gauges
and USGS observations for all. Scores of a multi-seed model are computed per seed and
then averaged (never on the averaged forecast). Forecast rain is NWM's unless a panel
says "perfect rain" (realized AORC, an upper bound).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing_analysis as ra  # noqa: E402
from routing_analysis import COLORS, plt  # noqa: E402

from tva_flood.data.gauges import Catalog  # noqa: E402
from tva_flood.training.selection import nse_by_lead  # noqa: E402

M, ROOT = ra.M, ra.ROOT
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SCRATCH, PRETRAINED, LSTM, NWM, PERSIST, CLIM = (COLORS[i] for i in (0, 1, 5, 6, 3, 4))
DAYS = [3, 7, 11, 15, 19]  # 24, 48, 72, 96, 120 h
LEADS = (24, 72, 120)
SEASONS = [("DJF", "Dec-Feb"), ("MAM", "Mar-May"), ("JJA", "Jun-Aug"), ("SON", "Sep-Nov")]
plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "lines.linewidth": 2,
    }
)
LEGEND = dict(frameon=True, framealpha=0.95, edgecolor="#d5dae3", fontsize=12)


def load(run):
    z = np.load(M / run / "forecasts.npz", allow_pickle=True)
    return dict(
        time=pd.DatetimeIndex(z["time"]),
        truth=z["truth"],
        values=z["values"],
        methods=[str(m) for m in z["methods"]],
        stations=[str(s) for s in z["stations"]],
    )


class Data:
    def __init__(self):
        a, b = load("fair_benchmark"), load("scratch_benchmark_s1")
        t = load("lstm_tuning_benchmark")
        for other in (b, t):
            if not (a["time"].equals(other["time"]) and a["stations"] == other["stations"]):
                raise ValueError("The benchmarks scored different issues or gauges")
        self.time, self.truth, self.stations = a["time"], a["truth"], a["stations"]
        self.year, self.month = self.time.year, self.time.month
        self.season = np.select(
            [self.month.isin([12, 1, 2]), self.month.isin([3, 4, 5]), self.month.isin([6, 7, 8])],
            ["DJF", "MAM", "JJA"],
            "SON",
        )

        def pick(d, names):
            return [d["values"][:, d["methods"].index(n)] for n in names]

        seeds = (1, 2, 3)
        # label: color, line style, {rain: [forecast per seed]}, per_station sources
        self.groups = {
            "GNN, from scratch": (
                SCRATCH,
                "-",
                dict(nwm=pick(b, ["scratch_s1+nwm"]), aorc=pick(b, ["scratch_s1+aorc"])),
                [("scratch_benchmark_s1", "scratch_s1")],
            ),
            "GNN, NWM-pretrained": (
                PRETRAINED,
                "-",
                dict(
                    nwm=pick(a, [f"graph_s{s}+nwm" for s in seeds]),
                    aorc=pick(a, [f"graph_s{s}+aorc" for s in seeds]),
                ),
                [("fair_benchmark", f"graph_s{s}") for s in seeds],
            ),
            # The LSTM after the tuning check (the GNN's batch size, learning rate, dropout and
            # gauge-balanced sampling), selected on 2019 validation; the headline baseline.
            "LSTM": (
                LSTM,
                "-",
                dict(
                    nwm=pick(t, [f"graph_like_s{s}+nwm" for s in seeds]),
                    aorc=pick(t, [f"graph_like_s{s}+aorc" for s in seeds]),
                ),
                [("lstm_tuning_benchmark", f"graph_like_s{s}") for s in seeds],
            ),
            "LSTM, original settings": (
                COLORS[2],
                "--",
                dict(
                    nwm=pick(a, [f"lstm_s{s}+nwm" for s in seeds]),
                    aorc=pick(a, [f"lstm_s{s}+aorc" for s in seeds]),
                ),
                [("fair_benchmark", f"lstm_s{s}") for s in seeds],
            ),
            "NWM": (NWM, "-", dict(nwm=pick(a, ["nwm"])), [("fair_benchmark", "nwm")]),
            "Persistence": (
                PERSIST,
                "--",
                dict(nwm=pick(a, ["persistence"])),
                [("fair_benchmark", "persistence")],
            ),
            "Climatology": (
                CLIM,
                ":",
                dict(nwm=pick(a, ["climatology"])),
                [("fair_benchmark", "climatology")],
            ),
        }
        self.runs = ra.Runs()
        self.cache = {}

    def nse(self, label, rain="nwm", rows=None, min_count=100):
        """NSE [lead, station], per seed then averaged; ``rows`` selects issues."""
        key = (label, rain, None if rows is None else rows.tobytes())
        if key not in self.cache:
            r = slice(None) if rows is None else rows
            scores = [
                nse_by_lead(f[r], self.truth[r], min_count) for f in self.groups[label][2][rain]
            ]
            with np.errstate(all="ignore"):
                self.cache[key] = np.nanmean(scores, axis=0)
        return self.cache[key]

    def median(self, label, rain="nwm", rows=None):
        with np.errstate(all="ignore"):
            return np.nanmedian(self.nse(label, rain, rows), axis=1)

    def table(self, label, column):
        """Median over gauges of a per_station.csv score at every lead, seeds averaged."""
        _, values = ra.per_gauge(self.runs, self.groups[label][3], "nwm", column)
        return values.median(axis=1)


def leadticks(ax, upto=240):
    ticks = [t for t in (6, 24, 48, 72, 96, 120, 168, 240) if t <= upto]
    ax.set_xticks(ticks)
    ax.set_xlabel("forecast lead (hours)")
    if upto > 120:
        ax.axvline(120, color=MUTED, lw=0.8, ls=":")


def save(fig, out, name):
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def note(fig, text, y=-0.02):
    fig.text(0.125, y, text, fontsize=9, color=MUTED, ha="left", va="top")


HOURS = np.arange(6, 241, 6)
TRAINED = ("GNN, from scratch", "GNN, NWM-pretrained", "LSTM")


def fig_skill(d, out):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for label, (color, ls, _, _) in d.groups.items():
        ax.plot(HOURS, d.median(label), color=color, ls=ls, label=label)
    ax.legend(loc="upper right", **LEGEND)
    leadticks(ax)
    ax.set_ylim(-0.9, 1.0)
    ax.set_ylabel("median NSE across gauges")
    ax.set_title("Skill vs lead, 2020-2025, NWM forecast rain")
    ax.text(122, -0.6, "day 5", color=MUTED)
    note(
        fig,
        "Multi-seed models: NSE per seed, then averaged. Persistence repeats the last observation.",
    )
    save(fig, out, "1_skill_by_lead.png")


def fig_rain(d, out):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    gaps = {}
    for label in TRAINED:
        color = d.groups[label][0]
        f, p = d.median(label), d.median(label, "aorc")
        gaps[label] = (p - f)[:20].max()
        ax.plot(HOURS, p, color=color, ls="--", lw=1.6)
        ax.plot(HOURS, f, color=color, label=label)
        if label != "GNN, from scratch":
            ax.fill_between(HOURS, f, p, color=color, alpha=0.12, lw=0)
    ax.plot(HOURS, d.median("NWM"), color=NWM, lw=1.5, label="NWM")
    ax.plot([], [], color=MUTED, ls="--", lw=1.6, label="dashed: perfect rain")
    ax.legend(loc="upper right", **LEGEND)
    leadticks(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("median NSE")
    ax.set_title(
        "What a better rain forecast could add: up to "
        f"{gaps['GNN, NWM-pretrained']:+.2f} NSE for the GNN, {gaps['LSTM']:+.2f} for the LSTM"
    )
    note(
        fig,
        "Solid: NWM forecast rain. Dashed: realized rain to 96 h (upper bound). Shading: the gap.",
    )
    save(fig, out, "2_rain_headroom.png")


def fig_scatter(d, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), sharey=True)
    lo = -0.3
    for row, ref in enumerate(("LSTM", "NWM")):
        x_all = d.nse(ref)
        for ax, h in zip(axes[row], LEADS):
            k = h // 6 - 1
            ax.plot([lo, 1], [lo, 1], color=MUTED, lw=0.8)
            lines = []
            for label, marker in (("GNN, NWM-pretrained", "^"), ("GNN, from scratch", "o")):
                y, x = d.nse(label)[k], x_all[k]
                ok = np.isfinite(x) & np.isfinite(y)
                ax.scatter(
                    np.clip(x[ok], lo, None),
                    np.clip(y[ok], lo, None),
                    s=30,
                    marker=marker,
                    color=d.groups[label][0],
                    edgecolor="#fcfcfb",
                    lw=0.8,
                    zorder=3,
                    label=label,
                )
                lines.append(f"{label.split(', ')[1]}: {(y[ok] > x[ok]).sum()} of {ok.sum()}")
            ax.set_title(f"{h} h, GNN better at\n" + "; ".join(lines), fontsize=9.5)
            ax.set_xlim(lo, 1.02)
            ax.set_ylim(lo, 1.02)
            ax.set_xlabel(f"{ref} NSE")
        axes[row, 0].set_ylabel(f"GNN NSE (vs {ref})")
    axes[0, 0].legend(loc="lower right", **{**LEGEND, "fontsize": 10})
    note(
        fig,
        f"Each marker is a gauge (NSE over 2020-2025, clipped at {lo}); above the line: GNN better.",
    )
    fig.tight_layout()
    save(fig, out, "3_gauge_scatter.png")


def gauge_table(stations):
    """Drainage area, gauges upstream, USGS id, name and location of each station."""
    stations = list(stations)
    info = ra.gauge_info(stations)
    public = pd.read_csv(
        ROOT / "processed/gauges_public/gauges.csv", dtype={"gauge": str, "usgs_id": str}
    )
    meta = pd.read_csv(
        ROOT / "raw/camelsh/metadata/tennessee_camelsh_gauges.csv", dtype={"STAID": str}
    ).set_index("STAID")
    usgs = public.set_index("gauge").usgs_id.reindex(stations)
    info["usgs_id"] = usgs.to_numpy()
    info["name"] = meta.STANAME.reindex(usgs).fillna("").to_numpy()
    info["lat"] = meta.LAT_GAGE.reindex(usgs).to_numpy()
    info["lon"] = meta.LNG_GAGE.reindex(usgs).to_numpy()
    return info


def days15(d, label, rows=None):
    """Per-gauge mean NSE over the 24-120 h leads."""
    with np.errstate(all="ignore"):
        return np.nanmean(d.nse(label, rows=rows)[DAYS], axis=0)


def fig_gain_by_gauge(d, info, out):
    base = days15(d, "LSTM")
    p = (
        pd.DataFrame(
            dict(
                gain=days15(d, "GNN, NWM-pretrained") - base,
                scratch=days15(d, "GNN, from scratch") - base,
                name=info["name"].to_numpy(),
                usgs=info["usgs_id"].to_numpy(),
            ),
            index=d.stations,
        )
        .dropna(subset=["gain"])
        .sort_values("gain")
    )
    fig, ax = plt.subplots(figsize=(9, max(8, 0.17 * len(p))))
    y = np.arange(len(p))
    colors = np.where(p.gain > 0, PRETRAINED, LSTM)
    ax.hlines(y, 0, p.gain, color=colors, lw=2.2)
    ax.scatter(p.gain, y, s=30, color=colors, zorder=3)
    ax.scatter(p.scratch, y, s=26, facecolor="none", edgecolor=SCRATCH, lw=1.3, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{u}  {n[:30].title()}" for u, n in zip(p.usgs, p.name)], fontsize=7)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("GNN NSE minus LSTM NSE (mean over 24-120 h leads)")
    ax.grid(axis="y", visible=False)
    ax.set_title(
        f"GNN (NWM-pretrained) ahead of the LSTM at {(p.gain > 0).sum()} of {len(p)} gauges "
        f"(from scratch: {(p.scratch > 0).sum()})"
    )
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=PRETRAINED, marker="o", lw=2.2, label="GNN NWM-pretrained better"),
        Line2D([], [], color=LSTM, marker="o", lw=2.2, label="LSTM better"),
        Line2D(
            [],
            [],
            color=SCRATCH,
            marker="o",
            markerfacecolor="none",
            lw=0,
            label="GNN from scratch",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", **LEGEND)
    save(fig, out, "4_gain_by_gauge.png")
    p.to_csv(out / "gain_by_gauge.csv")


def fig_gain_vs_area(d, info, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    area = info["area_km2"].to_numpy()
    for ax, h in zip(axes, (24, 120)):
        k = h // 6 - 1
        ref = d.nse("LSTM")[k]
        for label, marker in (("GNN, NWM-pretrained", "^"), ("GNN, from scratch", "o")):
            gain = d.nse(label)[k] - ref
            ok = np.isfinite(gain) & np.isfinite(area)
            ax.scatter(
                area[ok],
                gain[ok],
                s=30,
                marker=marker,
                color=d.groups[label][0],
                edgecolor="#fcfcfb",
                zorder=3,
                label=label,
            )
            r = np.corrcoef(np.log(area[ok]), gain[ok])[0, 1]
            ax.text(
                0.98,
                0.95 if "pre" in label else 0.87,
                f"{label}: r with log area {r:+.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color=d.groups[label][0],
            )
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"{h} h lead")
        ax.set_xlabel("drainage area (km², log)")
    axes[0].set_ylabel("GNN NSE minus LSTM NSE")
    axes[0].legend(loc="lower left", **LEGEND)
    fig.suptitle(
        "Does basin size explain where the GNN wins over the LSTM?",
        x=0.125,
        ha="left",
        fontweight="bold",
        y=1.0,
    )
    save(fig, out, "5_gain_vs_area.png")


def fig_by_year(d, out):
    labels = ["GNN, from scratch", "GNN, NWM-pretrained", "LSTM", "NWM"]
    years = sorted(set(d.year))
    fig, ax = plt.subplots(figsize=(11, 5))
    x, w = np.arange(len(years)), 0.2
    rows = []
    for i, label in enumerate(labels):
        v = []
        for y in years:
            with np.errstate(all="ignore"):
                v.append(np.nanmedian(days15(d, label, np.asarray(d.year == y))))
        ax.bar(x + (i - 1.5) * (w + 0.01), v, w, color=d.groups[label][0], label=label)
        for xi, vi in zip(x, v):
            ax.text(
                xi + (i - 1.5) * (w + 0.01),
                vi + 0.01,
                f"{vi:.2f}",
                ha="center",
                fontsize=7.5,
                color=MUTED,
            )
        rows.append(pd.Series(v, index=years, name=label))
    ax.set_xticks(x, years)
    ax.set_ylabel("median-gauge NSE, mean of days 1-5")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=4, bbox_to_anchor=(0, 1.0), **{**LEGEND, "fontsize": 10})
    ax.set_ylim(0, 0.8)
    ax.set_title("Year by year: median NSE, mean of days 1-5 (NWM rain)", pad=10)
    save(fig, out, "6_by_year.png")
    pd.concat(rows, axis=1).round(3).to_csv(out / "by_year.csv")


def fig_peaks_bias(d, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    parts = [
        ("peak_error_at_target_time", "Peak error at the observed peak (%)", 100),
        ("alpha", "Flow variability ratio (1 = right)", 1),
        ("beta", "Volume bias ratio (1 = right)", 1),
    ]
    for ax, (column, title, scale) in zip(axes, parts):
        for label in TRAINED + ("NWM",):
            s = d.table(label, column) * scale
            ax.plot(s.index, s, color=d.groups[label][0], label=label)
        ax.axhline(0 if scale == 100 else 1, color=MUTED, lw=0.8)
        ax.set_title(title, fontsize=10)
        leadticks(ax)
    axes[0].legend(loc="lower left", **{**LEGEND, "fontsize": 10})
    fig.suptitle(
        "Peak error, variability and volume bias at every lead (median over gauges, NWM rain)",
        x=0.125,
        ha="left",
        fontweight="bold",
        y=1.03,
        fontsize=11,
    )
    save(fig, out, "7_peaks_and_bias.png")


def fig_every_6h(d, out):
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.08)
    )
    for label, (color, ls, _, _) in d.groups.items():
        a1.plot(HOURS, d.median(label), color=color, ls=ls, marker="o", ms=3, lw=1.8, label=label)
    a1.plot(
        HOURS,
        d.median("GNN, NWM-pretrained", "aorc"),
        color=PRETRAINED,
        ls="--",
        lw=1.4,
        label="GNN NWM-pretrained, perfect rain",
    )
    a1.legend(loc="lower left", ncol=2, **{**LEGEND, "fontsize": 10.5})
    a1.set_ylabel("median NSE")
    a1.set_ylim(-0.9, 1)
    a1.axvline(120, color=MUTED, lw=0.8, ls=":")
    a1.set_title("NSE at every six-hour lead, 6 h to 240 h")
    diff = d.median("GNN, NWM-pretrained") - d.median("LSTM")
    a2.bar(HOURS, diff, width=4.4, color=np.where(diff > 0, PRETRAINED, LSTM))
    a2.plot(
        HOURS,
        d.median("GNN, from scratch") - d.median("LSTM"),
        color=SCRATCH,
        marker="o",
        ms=3,
        lw=1.4,
        label="GNN from scratch minus LSTM",
    )
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_ylabel("GNN minus LSTM")
    a2.set_xticks(range(6, 241, 12))
    a2.tick_params(axis="x", labelsize=8)
    a2.set_xlabel("forecast lead (hours)")
    a2.grid(axis="x", visible=False)
    from matplotlib.patches import Patch

    a2.legend(
        handles=[
            Patch(color=PRETRAINED, label="GNN NWM-pretrained better"),
            Patch(color=LSTM, label="LSTM better"),
            *a2.get_legend_handles_labels()[0],
        ],
        loc="upper right",
        **{**LEGEND, "fontsize": 10},
    )
    save(fig, out, "8_nse_every_6h.png")
    table = pd.DataFrame({label: d.median(label) for label in d.groups}, index=HOURS)
    table["GNN pretrained, perfect rain"] = d.median("GNN, NWM-pretrained", "aorc")
    table.round(3).to_csv(out / "nse_every_6h.csv")


def fig_other(d, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (column, title) in zip(
        axes[:2], (("kge", "Median KGE"), ("rmse", "Median RMSE (m³/s)"))
    ):
        for label in TRAINED + ("NWM",):
            s = d.table(label, column)
            ax.plot(s.index, s, color=d.groups[label][0], marker="o", ms=2.5, label=label)
        ax.set_title(title)
        leadticks(ax)
    axes[0].legend(loc="lower left", **{**LEGEND, "fontsize": 10})
    for ref, color, ls in (("LSTM", PRETRAINED, "-"), ("NWM", NWM, "--")):
        g, r = d.nse("GNN, NWM-pretrained"), d.nse(ref)
        ok = np.isfinite(g) & np.isfinite(r)
        share = 100 * ((g > r) & ok).sum(1) / ok.sum(1)
        axes[2].plot(HOURS, share, color=color, ls=ls, label=f"vs {ref}")
    axes[2].axhline(50, color=MUTED, lw=0.8)
    axes[2].set_ylim(0, 100)
    axes[2].set_title("Gauges where the GNN (pretrained) wins (%)")
    axes[2].legend(loc="lower left", **{**LEGEND, "fontsize": 10})
    leadticks(axes[2])
    fig.suptitle(
        "Other scores at every six-hour lead (median over gauges, NWM rain)",
        x=0.125,
        ha="left",
        fontweight="bold",
        y=1.03,
        fontsize=11,
    )
    save(fig, out, "9_other_metrics_every_6h.png")


def fig_diff_by_year(d, out):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for y, color in zip(sorted(set(d.year)), COLORS):
        rows = np.asarray(d.year == y)
        diff = d.median("GNN, NWM-pretrained", rows=rows) - d.median("LSTM", rows=rows)
        ax.plot(HOURS, diff, color=color, lw=1.6, label=str(y))
    ax.axhline(0, color=MUTED, lw=0.8)
    leadticks(ax)
    ax.legend(loc="upper right", ncol=3, title="test year", **{**LEGEND, "fontsize": 10})
    ax.set_ylabel("GNN NSE minus LSTM NSE")
    ax.set_title("GNN (pretrained) minus LSTM at every lead, one line per test year")
    save(fig, out, "10_difference_by_year.png")


def events_tables():
    ex = pd.concat(
        [
            pd.read_csv(M / "events_fair_benchmark/exceedance_by_lead.csv"),
            pd.read_csv(M / "events_scratch_benchmark_s1/exceedance_by_lead.csv"),
            pd.read_csv(M / "events_lstm_tuning_benchmark/exceedance_by_lead.csv"),
        ]
    )
    ev = pd.concat(
        [
            pd.read_csv(M / "events_fair_benchmark/events_by_lead.csv"),
            pd.read_csv(M / "events_scratch_benchmark_s1/events_by_lead.csv"),
            pd.read_csv(M / "events_lstm_tuning_benchmark/events_by_lead.csv"),
        ]
    )
    names = {
        "GNN, from scratch": ["scratch_s1+nwm"],
        "GNN, NWM-pretrained": [f"graph_s{s}+nwm" for s in (1, 2, 3)],
        "LSTM": [f"graph_like_s{s}+nwm" for s in (1, 2, 3)],
        "LSTM, original settings": [f"lstm_s{s}+nwm" for s in (1, 2, 3)],
        "NWM": ["nwm"],
        "Persistence": ["persistence"],
    }

    def average(table, columns):
        parts = []
        for label, methods in names.items():
            t = (
                table[table.method.isin(methods)]
                .groupby(["threshold", "lead_hours"])[columns]
                .mean()
            )
            parts.append(t.reset_index().assign(model=label))
        return pd.concat(parts)

    return (
        average(ex, ["csi", "pod", "far"]),
        average(ev, ["detection_rate", "median_size_error_pct", "median_abs_timing_error_h"]),
    )


def fig_floods(d, out):
    ex, ev = events_tables()
    style = {k: (d.groups[k][0], d.groups[k][1]) for k in ex.model.unique()}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, (th, title) in zip(
        axes,
        (
            ("p95", "p95 of flow"),
            ("p99", "p99 of flow (about 3.6 days/yr)"),
            ("annual", "median annual maximum"),
        ),
    ):
        for label, (color, ls) in style.items():
            t = ex[(ex.threshold == th) & (ex.model == label)]
            ax.plot(t.lead_hours, t.csi, color=color, ls=ls, label=label)
        ax.set_title(title, fontsize=10)
        leadticks(ax)
    axes[0].set_ylabel("critical success index (1 = perfect)")
    axes[2].legend(loc="upper right", **{**LEGEND, "fontsize": 10})
    fig.suptitle(
        "Flood warning skill (CSI) at every lead, NWM rain",
        x=0.125,
        ha="left",
        fontweight="bold",
        y=1.03,
        fontsize=11,
    )
    save(fig, out, "11_flood_csi_by_lead.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (column, title) in zip(
        axes,
        (
            ("detection_rate", "Observed p99 peaks reached (share)"),
            ("median_size_error_pct", "Median peak size error (%)"),
            ("median_abs_timing_error_h", "Median absolute timing error (h)"),
        ),
    ):
        for label, (color, ls) in style.items():
            t = ev[(ev.threshold == "p99") & (ev.model == label)]
            ax.plot(t.lead_hours, t[column], color=color, ls=ls, label=label)
        ax.set_title(title, fontsize=10)
        leadticks(ax)
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper right", **{**LEGEND, "fontsize": 10})
    fig.suptitle(
        "Observed flood peaks (p99): detection, size error and timing, NWM rain",
        x=0.125,
        ha="left",
        fontweight="bold",
        y=1.03,
        fontsize=11,
    )
    save(fig, out, "12_peak_detection_size_timing.png")
    ex.round(4).to_csv(out / "flood_csi.csv", index=False)
    ev.round(3).to_csv(out / "flood_peaks.csv", index=False)


def fig_cases(d, info, out):
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    catalog = Catalog(ROOT / "processed/gauges_public/gauges.nc")
    names = list(catalog.names)
    events = pd.read_csv(M / "events_fair_benchmark/events.csv.gz")
    th = pd.read_csv(M / "events_fair_benchmark/thresholds.csv", index_col=0)
    e = events[
        (events.threshold == "annual") & (events.method == "nwm") & (events.lead_hours == 24)
    ].copy()
    e["ratio"] = e.observed_peak / th["annual"].reindex(e.station).to_numpy()
    row = {t: i for i, t in enumerate(d.time)}
    picked, seen = [], set()
    for r in e.sort_values("ratio", ascending=False).itertuples():
        peak = pd.Timestamp(r.event)
        issues = [(peak - pd.Timedelta(days=k)).normalize() for k in (1, 3)]
        key = (peak.year, peak.month)
        if key in seen or not all(i in row for i in issues):
            continue
        seen.add(key)
        picked.append((r, issues))
        if len(picked) == 4:
            break
    shown = ("GNN, NWM-pretrained", "LSTM", "NWM")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    for ax, (r, issues) in zip(axes.ravel(), picked):
        peak = pd.Timestamp(r.event)
        s, g = d.stations.index(r.station), names.index(r.station)
        tt = pd.date_range(peak - pd.Timedelta(days=5), peak + pd.Timedelta(days=5), freq="6h")
        ax.plot(tt, catalog.on(tt)[:, g], color=INK, lw=2.4)
        for issue, ls in zip(issues, ("-", "--")):
            x = pd.date_range(issue, periods=40, freq="6h")
            for label in shown:
                f = np.mean([v[row[issue], :, s] for v in d.groups[label][2]["nwm"]], axis=0)
                ax.plot(x, f, color=d.groups[label][0], lw=1.6, ls=ls)
        ax.axhline(th["annual"][r.station], color=MUTED, lw=0.8, ls="-.")
        ax.set_xlim(tt[0], tt[-1])
        ax.set_title(
            f"{info.usgs_id[r.station]} {info.name[r.station][:42].title()}\n"
            f"peak {peak:%Y-%m-%d}: {r.observed_peak:.0f} m³/s",
            fontsize=10,
        )
        ax.set_ylabel("flow (m³/s)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", labelsize=8)
    handles = [Line2D([], [], color=INK, lw=2.4, label="Observed (USGS)")]
    handles += [Line2D([], [], color=d.groups[k][0], lw=1.8, label=k) for k in shown]
    handles += [
        Line2D([], [], color=MUTED, lw=1.6, ls="-", label="Issued 1-2 days before the peak"),
        Line2D([], [], color=MUTED, lw=1.6, ls="--", label="Issued 3-4 days before"),
        Line2D([], [], color=MUTED, lw=0.8, ls="-.", label="Median annual maximum"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.01),
        **{**LEGEND, "fontsize": 10.5},
    )
    fig.subplots_adjust(bottom=0.19, hspace=0.35)
    fig.suptitle(
        "Four of the largest observed floods (relative to each gauge's annual maximum), NWM rain",
        x=0.125,
        ha="left",
        fontweight="bold",
        y=0.97,
        fontsize=11,
    )
    note(fig, "Seed-averaged forecasts shown for the 3-seed models.", y=0.115)
    save(fig, out, "13_flood_case_studies.png")


def fig_season(d, out):
    labels = ["GNN, from scratch", "GNN, NWM-pretrained", "LSTM", "NWM"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, h in zip(axes, LEADS):
        x = np.arange(4)
        for i, label in enumerate(labels):
            v = [d.median(label, rows=np.asarray(d.season == s))[h // 6 - 1] for s, _ in SEASONS]
            ax.bar(x + (i - 1.5) * 0.2, v, 0.19, color=d.groups[label][0], label=label)
        ax.set_xticks(x, [n for _, n in SEASONS])
        ax.set_title(f"{h} h lead")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("median NSE (season of issue)")
    axes[2].legend(loc="upper right", **{**LEGEND, "fontsize": 9.5})
    fig.suptitle(
        "By season of issue (2020-2025, NWM rain)",
        x=0.125,
        ha="left",
        fontweight="bold",
        y=1.03,
        fontsize=11,
    )
    save(fig, out, "15_by_season.png")


def flow_classes(d):
    edges = np.stack(
        [ra.thresholds(d.stations, q) for q in (0.5, 0.9, 0.99)], axis=1
    )  # [station, 3]
    truth = d.truth
    with np.errstate(invalid="ignore"):
        return (
            (truth > edges[None, None, :, 0]).astype(int)
            + (truth > edges[None, None, :, 1])
            + (truth > edges[None, None, :, 2])
        ), edges


def fig_flow_level(d, out):
    classes, _ = flow_classes(d)
    labels = ["GNN, from scratch", "GNN, NWM-pretrained", "LSTM", "NWM"]
    names = ["below\nmedian", "median\nto p90", "p90\nto p99", "above\np99"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), sharex=True)
    for j, h in enumerate(LEADS):
        k = h // 6 - 1
        for i, label in enumerate(labels):
            bias, nrmse = [], []
            for q in range(4):
                b, n = [], []
                for f in d.groups[label][2]["nwm"]:
                    m = np.isfinite(f[:, k]) & np.isfinite(d.truth[:, k]) & (classes[:, k] == q)
                    o, v = d.truth[:, k][m], f[:, k][m]
                    b.append(100 * (v.sum() - o.sum()) / o.sum())
                    n.append(np.sqrt(((v - o) ** 2).mean()) / o.mean())
                bias.append(np.mean(b))
                nrmse.append(np.mean(n))
            x = np.arange(4) + (i - 1.5) * 0.2
            axes[0, j].bar(x, bias, 0.19, color=d.groups[label][0], label=label)
            axes[1, j].bar(x, nrmse, 0.19, color=d.groups[label][0])
        axes[0, j].axhline(0, color=MUTED, lw=0.8)
        axes[0, j].set_title(f"{h} h: volume bias (%)")
        axes[1, j].set_title(f"{h} h: normalised RMSE")
        axes[1, j].set_xticks(range(4), names, fontsize=8)
        for a in axes[:, j]:
            a.grid(axis="x", visible=False)
    axes[0, 0].legend(loc="lower left", **{**LEGEND, "fontsize": 9.5})
    note(fig, "Classes are per-gauge percentiles of 1979-2018 flow (public catalog).", y=0.01)
    fig.suptitle(
        "By observed flow level (NWM rain)",
        x=0.125,
        ha="left",
        fontweight="bold",
        y=0.99,
        fontsize=11,
    )
    save(fig, out, "16_by_flow_level.png")


def fig_blend(d, out):
    hs = np.arange(6, 121, 6)
    gnn, lstm = d.groups["GNN, NWM-pretrained"][2]["nwm"], d.groups["LSTM"][2]["nwm"]
    blend = [(a + b) / 2 for a, b in zip(gnn, lstm)]  # seed k with seed k
    d.groups["_blend"] = ("#8a8984", "--", dict(nwm=blend), [])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for label, color, ls in (
        ("GNN, NWM-pretrained", PRETRAINED, "-"),
        ("LSTM", LSTM, "-"),
        ("_blend", "#8a8984", "--"),
    ):
        a1.plot(
            hs,
            d.median(label)[: len(hs)],
            color=color,
            ls=ls,
            label="average of GNN and LSTM" if label == "_blend" else label,
        )
    a1.set_title("Averaging the GNN and the LSTM")
    a1.set_xlabel("forecast lead (hours)")
    a1.set_ylabel("median NSE")
    a1.set_xticks([6, 24, 48, 72, 96, 120])
    a1.legend(loc="upper right", **{**LEGEND, "fontsize": 10.5})
    sd = np.nanstd(d.truth, axis=0)  # [lead, station] spread per gauge
    for ref, color in (("LSTM", LSTM), ("NWM", NWM)):
        cor = []
        for h in hs:
            k = h // 6 - 1
            e1 = (gnn[0][:, k] - d.truth[:, k]) / sd[k]
            e2 = (d.groups[ref][2]["nwm"][0][:, k] - d.truth[:, k]) / sd[k]
            ok = np.isfinite(e1) & np.isfinite(e2)
            cor.append(np.corrcoef(e1[ok], e2[ok])[0, 1])
        a2.plot(hs, cor, color=color, label=f"GNN vs {ref}")
    a2.set_ylim(0, 1)
    a2.set_title("Correlation of forecast errors")
    a2.set_xlabel("forecast lead (hours)")
    a2.set_ylabel("correlation of normalised errors")
    a2.set_xticks([6, 24, 48, 72, 96, 120])
    a2.legend(loc="lower right", **{**LEGEND, "fontsize": 10.5})
    del d.groups["_blend"]
    save(fig, out, "17_blend_and_error_correlation.png")


def fig_size(d, info, out):
    area = info["area_km2"].to_numpy()
    size = pd.qcut(area, 3, labels=["small basins", "medium basins", "large basins"]).astype(str)
    nested = np.where(
        info["upstream_gauges"].to_numpy() > 0, "gauged upstream", "no gauge upstream"
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    shades = ["#f5b79c", PRETRAINED, "#a64514"]  # one hue, light to dark with lead
    for ax, (groups, order, title) in zip(
        axes,
        (
            (size, ["small basins", "medium basins", "large basins"], "By basin size (terciles)"),
            (nested, ["no gauge upstream", "gauged upstream"], "By whether a gauge lies upstream"),
        ),
    ):
        x = np.arange(len(order))
        for i, h in enumerate(LEADS):
            k = h // 6 - 1
            diff = d.nse("GNN, NWM-pretrained")[k] - d.nse("LSTM")[k]
            v = [np.nanmedian(diff[groups == o]) for o in order]
            ax.bar(x + (i - 1) * 0.26, v, 0.24, color=shades[i], label=f"{h} h")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, [f"{o}\n(n={(groups == o).sum()})" for o in order], fontsize=9)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("GNN minus LSTM NSE (median over gauges)")
    axes[1].legend(loc="upper right", title="lead", **{**LEGEND, "fontsize": 10})
    fig.suptitle(
        "Where the GNN (pretrained) beats the LSTM",
        x=0.125,
        ha="left",
        fontweight="bold",
        y=1.03,
        fontsize=11,
    )
    save(fig, out, "18_by_size_and_nesting.png")


def fig_low_flow(d, out):
    _, edges = flow_classes(d)
    low = d.truth < edges[None, None, :, 1]
    high = d.truth >= edges[None, None, :, 1]
    labels = ["GNN, from scratch", "GNN, NWM-pretrained", "LSTM", "NWM"]

    def bias(label, k, sel):
        out_ = []
        for f in d.groups[label][2]["nwm"]:
            m = np.isfinite(f[:, k]) & np.isfinite(d.truth[:, k]) & sel[:, k]
            o = d.truth[:, k][m]
            out_.append(100 * (f[:, k][m].sum() - o.sum()) / o.sum())
        return np.mean(out_)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (sel, title) in zip(
        axes[:2],
        ((low, "Flow below the 90th percentile"), (high, "Flow above the 90th percentile")),
    ):
        for label in labels:
            ax.plot(
                HOURS,
                [bias(label, h // 6 - 1, sel) for h in HOURS],
                color=d.groups[label][0],
                label=label,
            )
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title)
        leadticks(ax)
    axes[0].set_ylabel("volume bias (%)")
    axes[1].legend(loc="lower left", **{**LEGEND, "fontsize": 9.5})
    x = np.arange(4)
    for i, label in enumerate(labels):
        v = [bias(label, 11, low & (d.season == s)[:, None, None]) for s, _ in SEASONS]
        axes[2].bar(x + (i - 1.5) * 0.2, v, 0.19, color=d.groups[label][0])
    axes[2].axhline(0, color=MUTED, lw=0.8)
    axes[2].set_xticks(x, [n for _, n in SEASONS])
    axes[2].set_title("Low flow at 72 h, by season")
    axes[2].grid(axis="x", visible=False)
    fig.suptitle(
        "Volume bias at low and high flow (NWM rain)",
        x=0.125,
        ha="left",
        fontweight="bold",
        y=1.03,
        fontsize=11,
    )
    save(fig, out, "19_low_flow_bias.png")


def fig_month_size(d, info, out):
    from matplotlib.colors import LinearSegmentedColormap

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    x = np.arange(12)
    for i, (h, color) in enumerate(((24, "#f5b79c"), (72, PRETRAINED))):
        k = h // 6 - 1
        v = [
            d.median("GNN, NWM-pretrained", rows=np.asarray(d.month == m))[k]
            - d.median("LSTM", rows=np.asarray(d.month == m))[k]
            for m in range(1, 13)
        ]
        a1.bar(x + (i - 0.5) * 0.38, v, 0.36, color=color, label=f"{h} h")
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_xticks(x, list("JFMAMJJASOND"))
    a1.set_xlabel("month of issue")
    a1.set_ylabel("median NSE: GNN minus LSTM")
    a1.legend(loc="upper left", ncol=2, **{**LEGEND, "fontsize": 10})
    a1.set_title("Month by month (GNN pretrained)")
    a1.grid(axis="x", visible=False)
    size = pd.qcut(info["area_km2"].to_numpy(), 3, labels=False)
    grid = np.full((3, 4), np.nan)
    for i in range(3):
        for j, (s, _) in enumerate(SEASONS):
            rows = np.asarray(d.season == s)
            diff = d.nse("GNN, NWM-pretrained", rows=rows)[3] - d.nse("LSTM", rows=rows)[3]
            grid[i, j] = np.nanmedian(diff[size == i])
    cmap = LinearSegmentedColormap.from_list("d", [LSTM, "#f0efec", PRETRAINED])
    lim = np.nanmax(np.abs(grid))
    a2.imshow(grid, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    a2.grid(False)
    for i in range(3):
        for j in range(4):
            a2.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center", fontsize=11, color=INK)
    a2.set_xticks(range(4), [n for _, n in SEASONS])
    a2.set_yticks(range(3), ["small basins", "medium", "large basins"])
    a2.set_title("24 h: GNN minus LSTM (orange = GNN better)")
    for s in a2.spines.values():
        s.set_visible(False)
    fig.suptitle(
        "When and where the GNN gains over the LSTM",
        x=0.125,
        ha="left",
        fontweight="bold",
        y=1.03,
        fontsize=11,
    )
    save(fig, out, "20_season_month_size.png")


def fig_map(d, info, out):
    gain = days15(d, "GNN, NWM-pretrained") - days15(d, "LSTM")
    gain_map(
        info.assign(gain=gain),
        "GNN minus LSTM NSE, days 1-5 (orange = GNN better)",
        "Where the GNN (pretrained) beats the LSTM",
        out / "22_gain_map.png",
    )


def gain_map(table, label, title, path, better=PRETRAINED, worse=LSTM):
    """Gauges coloured by ``table.gain`` (lat, lon columns) on the basin and state map."""
    import geopandas as gpd
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    raw = ROOT / "raw"
    basin = gpd.read_file(raw / "geography/tennessee_huc2.geojson").to_crs(4326)
    states = gpd.read_file(raw / "geography/us_states/cb_2022_us_state_5m.shp").to_crs(4326)
    minx, miny, maxx, maxy = basin.total_bounds
    pad = 0.55
    box = (minx - pad, miny - pad, maxx + pad, maxy + pad)
    states = states.cx[box[0] : box[2], box[1] : box[3]]
    nodes = pd.read_parquet(
        ROOT / "processed/graph_candidate/nodes.parquet", columns=["comid", "totdasqkm"]
    )
    lines = (
        gpd.read_file(
            next((raw / "geography/nhdplus_v21/extracted").rglob("NHDFlowline.shp")),
            columns=["COMID", "geometry"],
        )
        .rename(columns={"COMID": "comid"})
        .merge(nodes, on="comid")
    )
    big = lines[lines.totdasqkm > 1500].to_crs(4326)
    t = table.dropna(subset=["gain", "lat"])
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_aspect(1 / np.cos(np.radians(36)))
    ax.grid(False)
    states.boundary.plot(ax=ax, color="#8a8984", lw=0.9, zorder=1)
    basin.plot(ax=ax, facecolor="#f3f2ee", edgecolor="none", zorder=0)
    big.plot(ax=ax, color="#9ec5f4", lw=0.9, zorder=2)
    basin.boundary.plot(ax=ax, color=INK, lw=1.8, zorder=3)
    lim = float(np.nanquantile(np.abs(t.gain), 0.95))  # a few outliers would wash out the rest
    cmap = LinearSegmentedColormap.from_list("d", [worse, "#f0efec", better])
    sc = ax.scatter(
        t.lon,
        t.lat,
        c=t.gain,
        cmap=cmap,
        norm=TwoSlopeNorm(0, -lim, lim),
        s=60,
        edgecolor=INK,
        lw=0.5,
        zorder=5,
    )
    labels = {
        "Tennessee": (-86.4, 35.85, "center"),
        "Kentucky": (-86.2, 37.55, "center"),
        "Virginia": (-80.85, 37.45, "right"),
        "North Carolina": (-80.85, 35.15, "right"),
        "Georgia": (-84.6, 33.85, "center"),
        "Alabama": (-87.3, 33.8, "center"),
    }
    for name, (x, y, ha) in labels.items():
        ax.text(
            x,
            y,
            name,
            fontsize=9,
            color=MUTED,
            style="italic",
            ha=ha,
            zorder=8,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )
    ax.set_xlim(box[0] - 0.15, box[2])
    ax.set_ylim(box[1], box[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    bar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01, extend="both")
    bar.set_label(label, color=INK)
    ax.set_title(f"{title}: {(t.gain > 0).sum()} of {len(t)} gauges", fontsize=11.5, color=INK)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_lstm_versions(d, info, out):
    """How much of the GNN's lead over the LSTM survives the LSTM's tuning."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    rows = []
    for ax, gnn in zip(axes, ("GNN, from scratch", "GNN, NWM-pretrained")):
        for i, (label, shade) in enumerate(
            (("LSTM, original settings", "#9ec5f4"), ("LSTM", LSTM))
        ):
            gains = days15(d, gnn) - days15(d, label)
            gains = gains[np.isfinite(gains)]
            stats = ra.bootstrap_median(gains, rng)
            ax.bar(i, stats[0], 0.55, color=shade)
            ax.errorbar(
                i,
                stats[0],
                yerr=[[stats[0] - stats[1]], [stats[2] - stats[0]]],
                color=INK,
                capsize=5,
            )
            ax.text(
                i,
                stats[2] + 0.002,
                f"{stats[0]:+.3f}\nbetter at {100 * (gains > 0).mean():.0f}%",
                ha="center",
                fontsize=10,
                color=INK,
            )
            rows.append(
                dict(
                    gnn=gnn,
                    lstm=label,
                    median_gain=stats[0],
                    low=stats[1],
                    high=stats[2],
                    gauges=len(gains),
                    share_better=float((gains > 0).mean()),
                )
            )
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_ylim(-0.002, 0.043)  # room above the bars for the labels
        ax.set_xticks([0, 1], ["LSTM, original\nsettings", "LSTM, tuned\n(GNN-like settings)"])
        ax.set_title(gnn)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("GNN minus LSTM NSE, days 1-5 (median, 95% bootstrap)")
    fig.suptitle(
        "How much of the GNN's gauged-site lead survives tuning the LSTM",
        x=0.125,
        ha="left",
        fontweight="bold",
        y=1.02,
        fontsize=11,
    )
    save(fig, out, "23_lstm_tuning_effect.png")
    pd.DataFrame(rows).round(4).to_csv(out / "lstm_tuning_effect.csv", index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "figures/gnn_vs_baselines")
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    d = Data()
    info = gauge_table(d.stations)
    for label in ("GNN, from scratch", "GNN, NWM-pretrained", "LSTM", "NWM"):
        m = d.median(label)
        print(
            f"{label}: median NSE 24 h {m[3]:.3f}, 72 h {m[11]:.3f}, days 1-5 {m[DAYS].mean():.3f}"
        )
    for f in (
        fig_skill,
        fig_rain,
        fig_scatter,
        fig_by_year,
        fig_peaks_bias,
        fig_every_6h,
        fig_other,
        fig_diff_by_year,
        fig_floods,
        fig_season,
        fig_flow_level,
        fig_blend,
    ):
        f(d, out)
        print("done", f.__name__)
    for f in (
        fig_gain_by_gauge,
        fig_gain_vs_area,
        fig_cases,
        fig_size,
        fig_month_size,
        fig_map,
        fig_lstm_versions,
    ):
        f(d, info, out)
        print("done", f.__name__)
    (out / "README.md").write_text(
        "# GNN vs public baselines\n\n"
        "LSTM = the tuned LSTM (GNN-like batch size, learning rate, dropout and gauge-balanced "
        "sampling; selected on 2019 validation). 'LSTM, original settings' is the first LSTM "
        "baseline, shown in figures 1 and 8 and in 23 for comparison. All models trained on the "
        "same public data, rain and budget; multi-seed models average per-seed scores.\n"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
