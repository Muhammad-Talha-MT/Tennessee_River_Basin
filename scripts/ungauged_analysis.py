"""LSTM tuning check and ungauged test of experiments.sh: selection, figures and reports.

    python scripts/ungauged_analysis.py best-lstm        # prints the selected LSTM variant
    python scripts/ungauged_analysis.py figures --folds 5

``best-lstm`` picks the tuning variant with the highest 2019 validation score (the
mean over seeds of each run's best selection score); the test years play no part.

``figures`` writes /data/tva_data/figures/lstm_tuning (validation and 2020-2025 test
of every variant, against the graph model from scratch) and figures/ungauged: every
held-out fold scored on its own gauges with their flow history hidden, beside the same
gauges in the gauged (in-sample) runs. "Observed upstream" means another catalog gauge
inside the basin that reports flow in 2020-2025 and is not in the held-out fold: the
information a river network can pass downstream and a lumped model cannot.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparison_figures as cf  # noqa: E402
import routing_analysis as ra  # noqa: E402
from comparison_figures import INK, LEGEND, LSTM, MUTED, NWM, SCRATCH, plt  # noqa: E402

from tva_flood.data.folds import holdout_folds  # noqa: E402
from tva_flood.data.gauges import Catalog  # noqa: E402
from tva_flood.data.river import River  # noqa: E402

M, ROOT = ra.M, ra.ROOT
HOURS = np.arange(6, 241, 6)
DAYS = [24, 48, 72, 96, 120]
VARIANT_ORDER = ["base", "small_batch", "lr_5e-4", "no_dropout", "balanced", "graph_like"]


def tuning_runs():
    """{variant: [run folder per seed]} of finished tuning runs."""
    found = {}
    for path in sorted(M.glob("lstm_tune_*_s[0-9]")):
        variant = path.name[len("lstm_tune_") :].rsplit("_s", 1)[0]
        found.setdefault(variant, []).append(path)
    return found


def selection_table():
    rows = []
    for variant, paths in tuning_runs().items():
        for path in paths:
            progress = path / "validation_progress.csv"
            if not (path / "DONE").exists() or not progress.exists():
                raise SystemExit(f"{path} has not finished")
            rows.append(
                dict(
                    variant=variant,
                    seed=path.name.rsplit("_s", 1)[1],
                    selection=pd.read_csv(progress).selection.max(),
                )
            )
    return pd.DataFrame(rows)


def best_lstm():
    table = selection_table()
    if table.empty:
        raise SystemExit("No LSTM tuning runs")
    means = table.groupby("variant").selection.mean()
    print(means.idxmax())


def median_by_lead(table, method):
    t = table[(table.method == method) & table.scored]
    return t.pivot(index="lead_hours", columns="station", values="nse")


def days15(frame):
    return frame.loc[DAYS].mean(axis=0)


def tuning_figures(out):
    out.mkdir(parents=True, exist_ok=True)
    sel = selection_table()
    order = [v for v in VARIANT_ORDER if v in set(sel.variant)]
    test = pd.read_csv(M / "lstm_tuning_benchmark/per_station.csv", dtype={"usgs_id": str})
    graph = pd.read_csv(M / "scratch_benchmark_s1/per_station.csv", dtype={"usgs_id": str})
    rows = []
    for variant in order:
        for seed in sorted(sel[sel.variant == variant].seed):
            nse = median_by_lead(test, f"{variant}_s{seed}+nwm").median(axis=1)
            rows.append(
                dict(
                    variant=variant,
                    seed=seed,
                    test_days_1_5=nse.loc[DAYS].mean(),
                    **{f"test_{h}h": nse.loc[h] for h in (24, 72, 120)},
                )
            )
    scores = sel.merge(pd.DataFrame(rows), on=["variant", "seed"])
    scores.to_csv(out / "lstm_tuning.csv", index=False)
    graph_days = median_by_lead(graph, "scratch_s1+nwm").median(axis=1).loc[DAYS].mean()
    best = sel.groupby("variant").selection.mean().idxmax()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    x = np.arange(len(order))
    for ax, column, title in (
        (axes[0], "selection", "2019 validation score (selects the variant)"),
        (axes[1], "test_days_1_5", "2020-2025 test, days 1-5 median NSE, NWM rain"),
    ):
        for i, variant in enumerate(order):
            v = scores[scores.variant == variant][column]
            color = LSTM if variant == best else "#8a8984"
            ax.bar(i, v.mean(), 0.6, color=color, alpha=0.35)
            ax.scatter([i] * len(v), v, color=color, s=36, zorder=3)
            ax.text(i, v.max() + 0.003, f"{v.mean():.3f}", ha="center", fontsize=9, color=INK)
        lo = scores[column].min() - 0.03
        if column == "test_days_1_5":
            ax.axhline(
                graph_days, color=SCRATCH, lw=2, label=f"GNN from scratch ({graph_days:.3f})"
            )
            lo = min(lo, graph_days - 0.03)
            ax.legend(loc="lower right", **{**LEGEND, "fontsize": 10.5})
        ax.set_ylim(
            lo, max(scores[column].max(), graph_days if column != "selection" else 0) + 0.02
        )
        ax.set_xticks(x, [v.replace("_", " ") for v in order], fontsize=9.5)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        f"LSTM tuning: one setting moved toward the GNN's at a time (dots: seeds; green: selected, {best})",
        x=0.125,
        ha="left",
        fontweight="bold",
        y=1.03,
        fontsize=11,
    )
    cf.save(fig, out, "1_lstm_tuning.png")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for variant, color, ls in ((best, LSTM, "-"), ("base", "#8a8984", "--")):
        seeds = [median_by_lead(test, f"{variant}_s{s}+nwm").median(axis=1) for s in (1, 2, 3)]
        ax.plot(HOURS, np.mean(seeds, axis=0), color=color, ls=ls, label=f"LSTM, {variant}")
    ax.plot(
        HOURS,
        median_by_lead(graph, "scratch_s1+nwm").median(axis=1),
        color=SCRATCH,
        label="GNN, from scratch",
    )
    cf.leadticks(ax)
    ax.set_ylabel("median NSE")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", **LEGEND)
    ax.set_title("Selected LSTM vs the GNN, 2020-2025, NWM rain")
    cf.save(fig, out, "2_lstm_vs_gnn_by_lead.png")
    return best


def upstream_observed(catalog, fold, folds):
    """Per gauge: catalog gauges observed in 2020-2025 inside its basin, outside its fold."""
    prep = ROOT / "processed/pretraining_full"
    with np.load(prep / "graph.npz") as g:
        ids = g["feature_id"].copy()
        edges = np.column_stack([g["source"], g["target"]])
    river = River.load(ROOT, ids, edges)
    outlet = pd.Index(ids).get_indexer(catalog.comid)
    reporting = np.isfinite(catalog.observed[catalog.times >= "2020-01-01"]).any(axis=0)
    area = (
        pd.read_parquet(
            ROOT / "processed/graph_candidate/nodes.parquet", columns=["comid", "totdasqkm"]
        )
        .set_index("comid")
        .totdasqkm.reindex(catalog.comid)
        .to_numpy()
    )
    rows = []
    for g, name in enumerate(catalog.names):
        if not fold[g]:
            continue
        mask = np.zeros(len(ids), dtype=bool)
        mask[outlet[g]] = True
        inside = river.closure(mask)[outlet] & reporting & (fold != fold[g])
        inside[g] = False
        share = area[inside].max() / area[g] if inside.any() else 0.0
        rows.append(
            dict(
                station=name,
                fold=int(fold[g]),
                area_km2=area[g],
                observed_upstream=int(inside.sum()),
                gauged_area_share=float(share),
            )
        )
    return pd.DataFrame(rows).set_index("station")


def ungauged_figures(out, folds, best):
    out.mkdir(parents=True, exist_ok=True)
    runs = [M / f"heldout_benchmark_f{k}" for k in range(1, folds + 1)]
    done = [r for r in runs if (r / "report.md").exists()]
    if not done:
        print("No held-out benchmark yet")
        return
    held = pd.concat([pd.read_csv(r / "per_station.csv", dtype={"usgs_id": str}) for r in done])
    gauged_graph = pd.read_csv(M / "scratch_benchmark_s1/per_station.csv", dtype={"usgs_id": str})
    gauged_lstm = pd.read_csv(M / "lstm_tuning_benchmark/per_station.csv", dtype={"usgs_id": str})
    stations = sorted(set(held[held.scored].station))

    def nse(table, methods):
        frames = [median_by_lead(table, m).reindex(columns=stations) for m in methods]
        return sum(frames) / len(frames)

    lstm3 = [f"lstm_s{s}" for s in (1, 2, 3)]
    curves = {
        ("GNN, ungauged", "nwm"): nse(held, ["graph+nwm"]),
        ("LSTM, ungauged", "nwm"): nse(held, [f"{m}+nwm" for m in lstm3]),
        ("GNN, gauged", "nwm"): nse(gauged_graph, ["scratch_s1+nwm"]),
        ("LSTM, gauged", "nwm"): nse(gauged_lstm, [f"{best}_s{s}+nwm" for s in (1, 2, 3)]),
        ("NWM", "nwm"): nse(held, ["nwm"]),
        ("GNN, ungauged", "aorc"): nse(held, ["graph+aorc"]),
        ("LSTM, ungauged", "aorc"): nse(held, [f"{m}+aorc" for m in lstm3]),
        ("GNN, gauged", "aorc"): nse(gauged_graph, ["scratch_s1+aorc"]),
        ("LSTM, gauged", "aorc"): nse(gauged_lstm, [f"{best}_s{s}+aorc" for s in (1, 2, 3)]),
    }
    style = {
        "GNN, ungauged": (SCRATCH, "-"),
        "LSTM, ungauged": (LSTM, "-"),
        "GNN, gauged": (SCRATCH, ":"),
        "LSTM, gauged": (LSTM, ":"),
        "NWM": (NWM, "-"),
    }
    partial = "" if len(done) == folds else f" ({len(done)} of {folds} folds so far)"
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    for ax, rain, title in (
        (axes[0], "nwm", "NWM forecast rain"),
        (axes[1], "aorc", "Perfect rain"),
    ):
        for (label, r), frame in curves.items():
            if r != rain:
                continue
            color, ls = style[label]
            m = frame.median(axis=1)
            ax.plot(HOURS, m, color=color, ls=ls, label=label)
            rows.append(
                dict(
                    model=label,
                    rain=rain,
                    gauges=int(frame.notna().any().sum()),
                    days_1_5=m.loc[DAYS].mean(),
                    **{f"nse_{h}h": m.loc[h] for h in (6, 24, 72, 120, 240)},
                )
            )
        ax.set_title(title)
        cf.leadticks(ax)
        ax.set_ylim(-0.5, 1)
    axes[0].set_ylabel(f"median NSE over the {len(stations)} held-out gauges")
    axes[0].legend(loc="lower left", **LEGEND)
    fig.suptitle(
        f"Ungauged: forecasts at gauges whose flow no model saw{partial}",
        x=0.125,
        ha="left",
        fontweight="bold",
        y=1.02,
        fontsize=12,
    )
    cf.note(
        fig,
        "Dotted: the same gauges when their flow was used (in-sample runs). NWM's "
        "analysis may assimilate these gauges.",
    )
    cf.save(fig, out, "1_ungauged_by_lead.png")
    skill = pd.DataFrame(rows)

    gnn, lstm = curves[("GNN, ungauged", "nwm")], curves[("LSTM, ungauged", "nwm")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True, sharey=True)
    for ax, h in zip(axes, (24, 72, 120)):
        x, y = lstm.loc[h].clip(-0.5), gnn.loc[h].clip(-0.5)
        ok = x.notna() & y.notna()
        ax.plot([-0.5, 1], [-0.5, 1], color=MUTED, lw=0.8)
        ax.scatter(x[ok], y[ok], s=32, color=SCRATCH, edgecolor="#fcfcfb", zorder=3)
        ax.set_title(f"{h} h: GNN better at {(y[ok] > x[ok]).sum()} of {ok.sum()}")
        ax.set_xlabel("LSTM NSE (ungauged)")
    axes[0].set_ylabel("GNN NSE (ungauged)")
    cf.note(fig, "Each dot is a held-out gauge, NWM rain; values below -0.5 shown at -0.5.")
    cf.save(fig, out, "2_ungauged_scatter.png")

    catalog = Catalog(ROOT / "processed/gauges_public/gauges.nc")
    info = upstream_observed(catalog, holdout_folds(ROOT, catalog, folds), folds).reindex(stations)
    gain = gnn - lstm
    rng = np.random.default_rng(0)
    groups = {
        "no observed gauge upstream": info.observed_upstream == 0,
        "observed gauge upstream": info.observed_upstream > 0,
        "upstream gauge covers half the basin or more": info.gauged_area_share >= 0.5,
    }
    colors = ["#8a8984", SCRATCH, "#1c5cab"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8))
    gain_rows = []
    for (name, mask), color in zip(groups.items(), colors):
        stats = np.array(
            [ra.bootstrap_median(gain.loc[h][mask.to_numpy()].to_numpy(), rng) for h in HOURS]
        )
        a1.plot(HOURS, stats[:, 0], color=color, label=f"{name} (n={int(mask.sum())})")
        a1.fill_between(HOURS, stats[:, 1], stats[:, 2], color=color, alpha=0.15)
        for h in (24, 72, 120):
            k = h // 6 - 1
            gain_rows.append(
                dict(
                    group=name,
                    gauges=int(mask.sum()),
                    lead_h=h,
                    median_gain=stats[k, 0],
                    low=stats[k, 1],
                    high=stats[k, 2],
                )
            )
    a1.axhline(0, color=INK, lw=0.8)
    cf.leadticks(a1)
    a1.set_ylabel("GNN minus LSTM NSE (ungauged, median)")
    a1.set_title("Does an observed gauge upstream help the GNN?")
    a1.legend(loc="upper right", **{**LEGEND, "fontsize": 10})
    d15 = days15(gain).clip(-1, 1)  # a few LSTM failures (NSE below -10) would hide the rest
    a2.scatter(
        info.gauged_area_share, d15.reindex(info.index), s=34, color=SCRATCH, edgecolor="#fcfcfb"
    )
    a2.axhline(0, color=INK, lw=0.8)
    a2.set_xlabel("share of the basin above the largest observed upstream gauge")
    a2.set_ylabel("GNN minus LSTM NSE, days 1-5 (clipped at +-1)")
    a2.set_title("Gain against how much of the basin is gauged upstream")
    fig.suptitle(
        "Where the river network should matter: ungauged gauges with observed flow upstream",
        x=0.125,
        ha="left",
        fontweight="bold",
        y=1.03,
        fontsize=11,
    )
    cf.save(fig, out, "3_gain_by_upstream_gauges.png")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for model, color in (("GNN", SCRATCH), ("LSTM", LSTM)):
        lost = (curves[(f"{model}, gauged", "nwm")] - curves[(f"{model}, ungauged", "nwm")]).median(
            axis=1
        )
        ax.plot(HOURS, lost, color=color, label=model)
    ax.axhline(0, color=MUTED, lw=0.8)
    cf.leadticks(ax)
    ax.set_ylabel("NSE lost without the gauge's own flow (median)")
    ax.set_title("What each model loses when the gauge is ungauged (NWM rain)")
    ax.legend(loc="upper right", **LEGEND)
    cf.save(fig, out, "4_skill_lost.png")

    table = cf.gauge_table(stations)
    cf.gain_map(
        table.assign(gain=d15.reindex(table.index).to_numpy()),
        "GNN minus LSTM NSE, days 1-5, ungauged (blue = GNN better; clipped at +-1)",
        "Ungauged: where the GNN beats the LSTM",
        out / "5_ungauged_gain_map.png",
        better=SCRATCH,
        worse=LSTM,
    )

    floods = flood_csi(done, stations)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for name, color in (("GNN", SCRATCH), ("LSTM", LSTM), ("NWM", NWM)):
        ax.plot(HOURS, floods[name], color=color, label=name)
    cf.leadticks(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("critical success index, p99")
    ax.set_title("Ungauged flood warnings above each gauge's training 99th percentile (NWM rain)")
    ax.legend(loc="upper right", **LEGEND)
    cf.save(fig, out, "6_ungauged_flood_csi.png")

    gains = pd.DataFrame(gain_rows)
    skill.to_csv(out / "skill.csv", index=False)
    gains.to_csv(out / "gain_by_upstream.csv", index=False)
    info.assign(gain_days_1_5=d15.reindex(info.index)).to_csv(out / "gauges.csv")
    pd.DataFrame(floods, index=HOURS).round(4).to_csv(out / "flood_csi.csv")
    (out / "report.md").write_text(
        "\n".join(
            [
                f"# Ungauged test{partial}",
                "",
                f"{len(stations)} held-out gauges; each scored by models trained without its flow and "
                f"given no flow history there. LSTM variant: {best} (3 seeds); GNN from scratch (1 seed).",
                "",
                "## Median NSE over the held-out gauges",
                "",
                ra.table_md(skill),
                "",
                "## GNN minus LSTM (ungauged) by upstream observation, 95% bootstrap over gauges",
                "",
                ra.table_md(gains),
                "",
            ]
        )
        + "\n"
    )
    print(f"Wrote {out}")


def flood_csi(runs, stations):
    """CSI above each gauge's training p99, pooled over gauges, per lead (seeds averaged)."""
    parts = [np.load(r / "forecasts.npz", allow_pickle=True) for r in runs]
    everyone = sorted({str(s) for p in parts for s in p["stations"]})  # scored or not
    flood = ra.thresholds(everyone, 0.99)
    index = {s: i for i, s in enumerate(everyone)}
    sums = {k: np.zeros((3, 40)) for k in ("GNN", "LSTM", "NWM")}
    members = dict(GNN=["graph+nwm"], LSTM=[f"lstm_s{s}+nwm" for s in (1, 2, 3)], NWM=["nwm"])
    for p in parts:
        methods = [str(m) for m in p["methods"]]
        cols = [index[str(s)] for s in p["stations"]]
        truth = p["truth"]
        observed = truth >= flood[cols][None, None]
        for name, names in members.items():
            for m in names:
                v = p["values"][:, methods.index(m)]
                ok = np.isfinite(v) & np.isfinite(truth)
                warned = v >= flood[cols][None, None]
                sums[name] += np.stack(
                    [
                        (observed & warned & ok).sum(axis=(0, 2)),
                        (observed & ~warned & ok).sum(axis=(0, 2)),
                        (~observed & warned & ok).sum(axis=(0, 2)),
                    ]
                ) / len(names)
    return {k: v[0] / v.sum(0) for k, v in sums.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("best-lstm")
    f = sub.add_parser("figures")
    f.add_argument("--folds", type=int, default=5)
    a = p.parse_args()
    if a.command == "best-lstm":
        best_lstm()
        return
    best = tuning_figures(ROOT / "figures/lstm_tuning")
    ungauged_figures(ROOT / "figures/ungauged", a.folds, best)


if __name__ == "__main__":
    main()
