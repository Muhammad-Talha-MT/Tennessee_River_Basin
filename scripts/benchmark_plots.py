"""Figures and a report comparing the graph models with the public baselines.

    python scripts/benchmark_plots.py --out /data/tva_data/figures/fair_comparison

Models (all trained on public data only, same budget and checkpoint selection):
the GNN from scratch (scratch_benchmark_s1), the GNN fine-tuned from the NWM-trained
stage 1 (fair_benchmark, 3 seeds), the lumped LSTM (fair_benchmark, 3 seeds); and
the references NWM medium range, persistence and climatology. Test: 00Z issues
2020-2025 at the CAMELS-H gauges against USGS observations, with NWM forcing rain,
GEFSv12 operational rain (from 2020-09-23: *_gefs runs) and realized rain.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing_analysis as ra  # noqa: E402
from routing_analysis import COLORS, INK, LEADS, legend, plt, style  # noqa: E402

# label, color, benchmark sources per test period ("main" 2020-2025, "gefs" from 2020-09-23)
MODELS = [
    (
        "GNN, from scratch",
        COLORS[0],
        dict(
            main=[("scratch_benchmark_s1", "scratch_s1")],
            gefs=[("scratch_benchmark_gefs_s1", "scratch_s1")],
        ),
    ),
    (
        "GNN, NWM-pretrained",
        COLORS[1],
        dict(
            main=[("fair_benchmark", f"graph_s{s}") for s in (1, 2, 3)],
            gefs=[("fair_benchmark_gefs", f"graph_s{s}") for s in (1, 2, 3)],
        ),
    ),
    (
        "LSTM",
        COLORS[5],
        dict(
            main=[("fair_benchmark", f"lstm_s{s}") for s in (1, 2, 3)],
            gefs=[("fair_benchmark_gefs", f"lstm_s{s}") for s in (1, 2, 3)],
        ),
    ),
    (
        "NWM",
        COLORS[6],
        dict(main=[("fair_benchmark", "nwm")], gefs=[("fair_benchmark_gefs", "nwm")]),
    ),
    (
        "Persistence",
        COLORS[3],
        dict(
            main=[("fair_benchmark", "persistence")], gefs=[("fair_benchmark_gefs", "persistence")]
        ),
    ),
    (
        "Climatology",
        COLORS[4],
        dict(
            main=[("fair_benchmark", "climatology")], gefs=[("fair_benchmark_gefs", "climatology")]
        ),
    ),
]
TRAINED = ("GNN, from scratch", "GNN, NWM-pretrained", "LSTM")
RAINS = {
    "nwm": "NWM forecast rain, 2020-2025",
    "gefs": "GEFSv12 forecast rain, 2020-09 to 2025",
    "aorc": "Realized rain (perfect forecast), 2020-2025",
}
DAYS = [3, 7, 11, 15, 19]  # 24-120 h, as "days 1-5 mean" in the benchmark reports


def period(rain):
    return "gefs" if rain == "gefs" else "main"


def models(runs, rain, names=None):
    """{label: (color, sources)} for one rain, of the models whose benchmark exists."""
    found = {}
    for label, color, sources in MODELS:
        if names and label not in names:
            continue
        if rain == "aorc" and label not in TRAINED:
            continue  # references have no perfect-rain version
        chosen = sources[period(rain)]
        if all(runs.exists(run) for run, _ in chosen):
            found[label] = (color, chosen)
    return found


def nse_by_lead(runs, out, rows):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8), sharey=True)
    for ax, (rain, title) in zip(axes, RAINS.items()):
        for label, (color, sources) in models(runs, rain).items():
            seeds, _ = ra.per_gauge(runs, sources, rain)
            medians = np.array([f.median(axis=1).to_numpy() for f in seeds])
            mean = medians.mean(0)
            dashed = label in ("Persistence", "Climatology")
            ax.plot(
                LEADS,
                mean,
                color=color,
                linewidth=2,
                linestyle="--" if dashed else "-",
                label=label,
            )
            if len(seeds) > 1:
                ax.fill_between(LEADS, medians.min(0), medians.max(0), color=color, alpha=0.18)
            rows.append(
                dict(
                    model=label,
                    rain=rain,
                    seeds=len(seeds),
                    **{f"nse_{h}h": mean[h // 6 - 1] for h in (6, 24, 72, 120, 240)},
                    days_1_5=mean[DAYS].mean(),
                    days_1_5_seed_min=medians[:, DAYS].mean(1).min(),
                    days_1_5_seed_max=medians[:, DAYS].mean(1).max(),
                )
            )
        ax.axhline(0, color="#c3c2b7", linewidth=1)
        ax.set_title(title, color=INK, fontsize=13, loc="left")
        style(ax, "Lead time (h)", "Median NSE over gauges" if rain == "nwm" else "")
        ra.lead_axis(ax)
    axes[0].set_ylim(-0.6, 1.0)
    legend(axes[0], loc="lower left")
    fig.suptitle(
        "Skill by lead time at the CAMELS-H gauges (band: seed range)",
        color=INK,
        fontsize=14,
        x=0.05,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "01_nse_by_lead.png", dpi=160)
    plt.close(fig)


def summary(skill, out):
    """Days 1-5 mean NSE per model and rain source, seed range as a line."""
    markers = dict(nwm="o", gefs="s", aorc="^")
    order = [m for m, _, _ in MODELS if m in set(skill.model)][::-1]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for y, label in enumerate(order):
        color = dict((m, c) for m, c, _ in MODELS)[label]
        for rain, marker in markers.items():
            row = skill[(skill.model == label) & (skill.rain == rain)]
            if row.empty:
                continue
            r = row.iloc[0]
            offset = {"nwm": 0.18, "gefs": 0.0, "aorc": -0.18}[rain]
            ax.plot(
                [r.days_1_5_seed_min, r.days_1_5_seed_max],
                [y + offset] * 2,
                color=color,
                linewidth=2,
            )
            ax.plot(
                r.days_1_5,
                y + offset,
                marker=marker,
                markersize=10,
                color=color,
                markeredgecolor="#fcfcfb",
                linestyle="none",
            )
            ax.annotate(
                f"{r.days_1_5:.3f}",
                (r.days_1_5, y + offset),
                xytext=(8, -4),
                textcoords="offset points",
                fontsize=10,
                color=INK,
            )
    ax.set_yticks(range(len(order)), order)
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    style(ax, "Mean of median NSE at 24, 48, 72, 96 and 120 h", "")
    for rain, marker in markers.items():
        ax.plot([], [], marker=marker, color=INK, linestyle="none", markersize=9, label=RAINS[rain])
    legend(ax, loc="lower right")
    fig.suptitle(
        "Days 1-5 skill by model and rain source (line: seed range)",
        color=INK,
        fontsize=14,
        x=0.05,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "02_days_1_5_summary.png", dpi=160)
    plt.close(fig)


def gain_over_lstm(runs, out, rows):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8), sharey=True)
    for ax, (rain, title) in zip(axes, RAINS.items()):
        found = models(runs, rain)
        if "LSTM" not in found:
            continue
        _, lstm = ra.per_gauge(runs, found["LSTM"][1], rain)
        for label in ("GNN, from scratch", "GNN, NWM-pretrained"):
            if label not in found:
                continue
            color, sources = found[label]
            _, model = ra.per_gauge(runs, sources, rain)
            diff = (model - lstm).dropna(axis=1, how="all")
            stats = np.array([ra.bootstrap_median(diff.loc[h].to_numpy(), rng) for h in LEADS])
            ax.plot(LEADS, stats[:, 0], color=color, linewidth=2, label=label)
            ax.fill_between(LEADS, stats[:, 1], stats[:, 2], color=color, alpha=0.18)
            for h in (24, 72, 120, 240):
                k = h // 6 - 1
                rows.append(
                    dict(
                        model=label,
                        rain=rain,
                        lead_h=h,
                        median_gain=stats[k, 0],
                        low=stats[k, 1],
                        high=stats[k, 2],
                        gauges_better=float((diff.loc[h] > 0).mean()),
                        gauges=int(diff.loc[h].notna().sum()),
                    )
                )
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_title(title, color=INK, fontsize=13, loc="left")
        style(ax, "Lead time (h)", "NSE gain over the LSTM (per gauge)" if rain == "nwm" else "")
        ra.lead_axis(ax)
    legend(axes[0], loc="upper right")
    fig.suptitle(
        "GNN minus LSTM, median over gauges (band: 95% bootstrap over gauges)",
        color=INK,
        fontsize=14,
        x=0.05,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "03_gain_over_lstm.png", dpi=160)
    plt.close(fig)


def scatter(runs, out):
    found = models(runs, "nwm")
    _, lstm = ra.per_gauge(runs, found["LSTM"][1], "nwm")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, h in zip(axes, (24, 72)):
        notes = []
        for label in ("GNN, NWM-pretrained", "GNN, from scratch"):
            color, sources = found[label]
            _, model = ra.per_gauge(runs, sources, "nwm")
            both = pd.concat([lstm.loc[h], model.loc[h]], axis=1, keys=["x", "y"]).dropna()
            marker = "o" if "scratch" in label else "^"
            ax.scatter(
                both.x.clip(-0.5),
                both.y.clip(-0.5),
                s=40,
                marker=marker,
                color=color,
                edgecolor="#fcfcfb",
                linewidth=1,
                label=label,
            )
            notes.append(f"{label}: {100 * (both.y > both.x).mean():.0f}% of gauges above")
        ax.plot([-0.5, 1], [-0.5, 1], color=INK, linewidth=1)
        ax.set_xlim(-0.5, 1)
        ax.set_ylim(-0.5, 1)
        ax.set_aspect("equal")
        ax.set_title(
            f"{h} h lead, NWM rain\n" + "\n".join(notes), color=INK, fontsize=11, loc="left"
        )
        style(ax, "LSTM NSE", "GNN NSE" if h == 24 else "")
    legend(axes[0], loc="lower right")
    fig.suptitle(
        "Per gauge: GNN vs LSTM (values below -0.5 shown at -0.5)",
        color=INK,
        fontsize=14,
        x=0.05,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "04_gauge_scatter.png", dpi=160)
    plt.close(fig)


def kge_parts(runs, out):
    found = {k: v for k, v in models(runs, "nwm").items() if k in TRAINED + ("NWM",)}
    parts = [("r", "Correlation r"), ("alpha", "Variability ratio"), ("beta", "Bias ratio")]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.6))
    for ax, (column, name) in zip(axes, parts):
        for label, (color, sources) in found.items():
            _, values = ra.per_gauge(runs, sources, "nwm", column)
            ax.plot(LEADS, values.median(axis=1), color=color, linewidth=2, label=label)
        if column != "r":
            ax.axhline(1, color=INK, linewidth=1)
        style(ax, "Lead time (h)", f"Median {name.lower()}")
        ra.lead_axis(ax)
        ax.set_title(name, color=INK, fontsize=13, loc="left")
    legend(axes[0], loc="lower left")
    fig.suptitle(
        "Error components of KGE, NWM rain (1 is perfect for all three)",
        color=INK,
        fontsize=14,
        x=0.05,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "05_kge_components.png", dpi=160)
    plt.close(fig)


def flood_skill(runs, out, rows):
    """Exceedance of each gauge's training 99th percentile, pooled over gauges and issues."""
    truth = runs.array("fair_benchmark")["truth"]
    flood = ra.thresholds(runs.reference["stations"], 0.99)
    observed = truth >= flood[None, None]
    found = {k: v for k, v in models(runs, "nwm").items() if k != "Climatology"}
    names = [
        ("csi", "Critical success index"),
        ("pod", "Probability of detection"),
        ("far", "False alarm ratio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.6))
    for label, (color, sources) in found.items():
        scores = []
        for values in ra.series(runs, sources, "nwm"):
            ok = np.isfinite(values) & np.isfinite(truth)
            warned = values >= flood[None, None]
            hits = (observed & warned & ok).sum(axis=(0, 2))
            misses = (observed & ~warned & ok).sum(axis=(0, 2))
            false = (~observed & warned & ok).sum(axis=(0, 2))
            with np.errstate(divide="ignore", invalid="ignore"):
                scores.append(
                    dict(
                        csi=hits / (hits + misses + false),
                        pod=hits / (hits + misses),
                        far=false / (hits + false),
                    )
                )
        mean = {k: np.mean([s[k] for s in scores], axis=0) for k in ("csi", "pod", "far")}
        for ax, (key, _) in zip(axes, names):
            ax.plot(
                LEADS,
                mean[key],
                color=color,
                linewidth=2,
                linestyle="--" if label == "Persistence" else "-",
                label=label,
            )
        for h in (6, 24, 48, 72, 120):
            k = h // 6 - 1
            rows.append(dict(model=label, lead_h=h, **{key: mean[key][k] for key in mean}))
    for ax, (key, name) in zip(axes, names):
        style(ax, "Lead time (h)", name)
        ra.lead_axis(ax)
        ax.set_ylim(0, 1)
        ax.set_title(
            name + (" (lower is better)" if key == "far" else ""),
            color=INK,
            fontsize=13,
            loc="left",
        )
    legend(axes[0], loc="upper right")
    fig.suptitle(
        "Flood warnings: flow above each gauge's training 99th percentile, NWM rain",
        color=INK,
        fontsize=14,
        x=0.05,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "06_flood_skill.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, default=ra.ROOT / "figures/fair_comparison")
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    runs = ra.Runs()
    for run in ("fair_benchmark", "scratch_benchmark_s1"):
        if not runs.exists(run):
            raise SystemExit(f"{run} is missing")
        runs.array(run)  # same issues and gauges, checked
    skill, gain, floods, peaks = [], [], [], []
    nse_by_lead(runs, out, skill)
    skill = pd.DataFrame(skill)
    summary(skill, out)
    gain_over_lstm(runs, out, gain)
    scatter(runs, out)
    kge_parts(runs, out)
    flood_skill(runs, out, floods)
    events = {k: v for k, v in models(runs, "nwm").items() if k in TRAINED + ("NWM",)}
    ra.peaks(runs, events, out, peaks, name="07_flood_peaks.png")
    info = ra.gauge_info(runs.reference["stations"])
    ra.hydrographs(runs, events, info, out, name="08_hydrographs.png")
    tables = dict(
        skill=skill,
        gain_over_lstm=pd.DataFrame(gain),
        flood_skill=pd.DataFrame(floods),
        flood_peaks=pd.DataFrame(peaks),
    )
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    gefs_note = (
        ""
        if runs.exists("scratch_benchmark_gefs_s1")
        else (" The GNN from scratch has no GEFS-rain run yet.")
    )
    lines = [
        "# GNN vs public baselines",
        "",
        "00Z issues, CAMELS-H gauges, USGS observations; median NSE over gauges. GNN "
        "NWM-pretrained and LSTM: 3 seeds (mean, seed range); GNN from scratch: 1 seed."
        + gefs_note,
        "",
        "## Skill",
        "",
        ra.table_md(skill),
        "",
        "## GNN gain over the LSTM (median over gauges, 95% bootstrap over gauges)",
        "",
        ra.table_md(tables["gain_over_lstm"]),
        "",
        "## Flood warnings above the training 99th percentile, NWM rain",
        "",
        ra.table_md(tables["flood_skill"]),
        "",
        "## Flood peaks (observed above training p99), NWM rain",
        "",
        ra.table_md(tables["flood_peaks"], 2),
        "",
        "Figures: 01_nse_by_lead, 02_days_1_5_summary, 03_gain_over_lstm, 04_gauge_scatter, "
        "05_kge_components, 06_flood_skill, 07_flood_peaks, 08_hydrographs.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
