"""Does the graph model agree with the NWM retrospective at every reach?

The retrospective (NWM v3.0 driven by realized AORC, 1979-Jan 2023) exists at all
58,834 reaches, gauged or not. It is a model, not an observation: agreement with it
says where the graph model behaves like NWM's physics, not where it is right. Two
checks keep that honest: at the gauges the retrospective is also scored against USGS,
so its own error is known, and the graph model is scored against both.

  compute  graph model (whole-basin, from scratch) forecasts at every reach of the
           gauges' basins for 00Z issues 2020-01 to 2022-12, scored against the
           retrospective flow over the following ten days. Rain is the realized AORC
           (the retrospective's own forcing: a like-for-like test) and, separately,
           NWM operational forecast rain. Reference: the retrospective's own latest
           flow carried forward ("persistence", not available to a real forecast).
  plots    maps and curves (figures/retrospective).
  ungauged held-out fold models and NWM forecasts at every reach (after the fold and
           --reference-only runs; figures/retrospective_ungauged).

  PYTHONPATH=src python scripts/retrospective_comparison.py compute --every 3
  PYTHONPATH=src python scripts/retrospective_comparison.py plots

Ungauged mode: --holdout-fold k with the fold-k model (scratch_graph_heldout_fk) hides that
fold's gauges' flow history, as the held-out benchmark does. NWM operational forecasts:
--reference-only scores NWM medium-range streamflow at every reach (nwm_all_reaches.py)
against the same retrospective instead of a graph model (CPU is enough).

GEFS rain (--rain gefs): the GEFSv12 operational control (--gefs) replaces NWM's forecast
rain, on the days that have it and an NWM forecast at every reach (2020-09-23 on), as
``tva_flood.hydro.evaluate`` selects them; with --reference-only it gives the NWM forecast on
those same issues.

Another checkpoint: --checkpoint PATH --tag NAME (outputs get the suffix _NAME). On CPU
(--device cpu, ~2 min per issue), split the issues: --shard k --shards n in n processes,
then merge --tag NAME.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/data/tva_data")
MODELS = ROOT / "models"
OUT = MODELS / "retrospective_comparison"  # a --tag adds a suffix (set in main)
FIGURES = ROOT / "figures/retrospective"
TEST = ROOT / "processed/test_preparation_2020_2025"
GAUGES = ROOT / "processed/gauges_public/gauges.nc"
CHECKPOINT = MODELS / "scratch_graph_s1/best.pt"
RETRO_END = pd.Timestamp("2023-01-01")  # the retrospective ends in January 2023
RETRO_LAST_ISSUE = pd.Timestamp("2023-01-20")  # ten days of targets remain


def whole_graph(inputs, gauges, n):
    """``inputs.graph(gauges)`` over every reach: forecasts at all of them, gauged or not."""
    import torch as t

    nodes = np.arange(n)
    graph = inputs.sampler.subgraph(nodes, np.ones(n, dtype=bool))
    graph["gauges"] = gauges
    graph["local"] = inputs.outlet[gauges]
    inside = np.arange(len(inputs.outlet))
    graph["visible_gauges"], graph["visible_local"] = inside, inputs.outlet
    graph["owner"] = t.as_tensor(inputs.owner[nodes])
    graph["forecast_owner"] = t.as_tensor(inputs.forecast_owner[nodes])
    graph["static"] = t.as_tensor(inputs.data.static[nodes], dtype=t.float32)
    return graph


class NWMReaches:
    """NWM medium-range six-hour flow [lead, reach] by issue day, one month in memory."""

    def __init__(self, folder):
        self.paths = {p.stem: p for p in Path(folder).glob("????-??.npz")}
        self.loaded, self.month = {}, None

    def days(self):
        result = set()
        for path in self.paths.values():
            with np.load(path) as d:
                result.update(pd.Timestamp(str(x)) for x in d["issue"])
        return result

    def get(self, day):
        key = f"{day:%Y-%m}"
        if key != self.month:
            with np.load(self.paths[key]) as d:
                self.loaded = {pd.Timestamp(str(x)): d["flow"][i] for i, x in enumerate(d["issue"])}
            self.month = key
        return self.loaded[day]


def compute(a):
    from tva_flood.data.folds import held_out
    from tva_flood.data.gauges import Catalog
    from tva_flood.data.inputs import ForecastInputs
    from tva_flood.data.rain_forecasts import RainForecasts
    from tva_flood.data.river import River
    from tva_flood.data.schema import HORIZON
    from tva_flood.data.ssl import EncoderWindows
    from tva_flood.evaluation.benchmark import load_nwm
    from tva_flood.evaluation.hydrology import (
        Accumulator,
        load_forecast,
        raw_flow,
        read_forecast,
        reach_metrics,
    )
    from tva_flood.models.sparse import sparse_supports

    device = torch.device(a.device)
    catalog = Catalog(GAUGES)
    data = EncoderWindows(
        ROOT, TEST, "test", HORIZON, 4, array_cache=Path(str(TEST) + "_arrays"), future_rain=True
    )
    with np.load(TEST / "graph.npz") as g:
        ids, edges = g["feature_id"].copy(), np.column_stack([g["source"], g["target"]])
    river = River.load(ROOT, ids, edges)
    saved = read_forecast(a.checkpoint)
    trained = saved["config"].get("observed_finetune", {}).get("arguments", {})
    if not a.reference_only and (trained.get("holdout_fold") or 0) != a.holdout_fold:
        raise SystemExit(f"{a.checkpoint} was not trained with fold {a.holdout_fold} withheld")
    held = held_out(ROOT, catalog, a.holdout_fold, a.folds)
    architecture = saved["config"]["architecture"]
    halo = architecture["graph_layers"] + saved["config"]["decoder"]["decoder_graph_layers"]
    supports = sparse_supports(len(ids), edges)
    inputs = ForecastInputs(
        data, catalog, None, edges, supports, halo, river if architecture.get("direction") == "upstream" else None
    )
    everything = np.arange(len(catalog))
    graph = whole_graph(inputs, everything, len(ids))
    nodes = graph["nodes"]
    print(f"graph of {len(nodes)} of {len(ids)} reaches", flush=True)
    graph_supports = [s.to(device) for s in graph["supports"]]
    static = graph["static"].to(device)
    model = None if a.reference_only else load_forecast(saved, data, TEST, supports, device)
    reaches = NWMReaches(a.nwm_reaches) if a.reference_only or a.rain == "gefs" else None
    available = reaches.days() if reaches else None
    gefs = RainForecasts(a.gefs) if a.rain == "gefs" else None
    steps = saved["config"].get("rain_hours", 96) // 6
    mean, std = inputs.mean, inputs.std
    nwm, _ = load_nwm(ROOT / "processed/benchmarks/nwm_medium_range")

    times = data.times
    items = [
        i
        for i, issue in enumerate(data.issues)
        if times[issue].hour == 0
        and pd.Timestamp("2020-01-01") <= times[issue] <= RETRO_LAST_ISSUE
        and times[issue].normalize() in nwm
        and (gefs is None or (times[issue].normalize() in gefs and times[issue].normalize() in available))
    ][:: a.every]
    if available is not None and gefs is None:
        items = [i for i in items if times[data.issues[i]].normalize() in available]
    items = items[a.shard :: a.shards]
    if a.limit:
        items = items[: a.limit]
    print(f"{len(items)} issues", flush=True)

    models = ["nwm_forecast"] if a.reference_only else ["gnn+aorc", f"gnn+{a.rain}"]
    totals = {m: Accumulator(len(nodes), device) for m in models + ["retro_persistence"]}
    at_gauges = {m: Accumulator(len(everything), device) for m in ["retro"] + models}
    began = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device.type, torch.bfloat16, enabled=a.amp):
        for position, item in enumerate(items):
            sample = inputs.load(data, item, graph, held if a.holdout_fold else None)
            issue = sample["issue"]
            retro = raw_flow(sample["target"].to(device), mean, std)
            valid = sample["target_valid"].to(device).bool()
            latest_valid = sample["valid"][-1].to(device).bool()
            latest = raw_flow(sample["flow"][-1].to(device), mean, std)
            mask = valid & latest_valid[None]
            forecasts = {}
            if a.reference_only:
                flow = torch.as_tensor(reaches.get(times[issue].normalize()), device=device)
                mask = mask & torch.isfinite(flow)
                forecasts["nwm_forecast"] = torch.nan_to_num(flow).double()
            else:
                dynamic = sample["dynamic"][None].to(device)
                rain_aorc = sample["future"].clone()
                rain_aorc[steps:] = 0
                day = times[issue].normalize()
                forecast_mm = gefs.get(day) if gefs is not None else nwm[day][1]
                rain_forecast = inputs.rain_field(sample, graph, forecast_mm, steps)
                for tag, rain in (("aorc", rain_aorc), (a.rain, rain_forecast)):
                    out = model(dynamic, static, rain[None].to(device), graph_supports)[0]
                    forecasts[f"gnn+{tag}"] = raw_flow(out.float(), mean, std)
            forecasts["retro_persistence"] = latest[None].expand(HORIZON, -1)
            retro0 = torch.nan_to_num(retro)
            for name, value in forecasts.items():
                totals[name].add(value.double(), retro0.double(), mask)
            # At the gauges: the retrospective and the graph model against USGS.
            obs = torch.as_tensor(inputs.observed[issue : issue + HORIZON], device=device)
            local = torch.as_tensor(graph["local"], device=device)
            truth = torch.nan_to_num(obs)
            ok = torch.isfinite(obs) & mask[:, local]
            at_gauges["retro"].add(retro0[:, local].double(), truth.double(), ok)
            for name in models:
                at_gauges[name].add(forecasts[name][:, local].double(), truth.double(), ok)
            if position % 25 == 0:
                print(
                    f"issue {position + 1}/{len(items)}; {(time.perf_counter() - began) / 60:.1f} min",
                    flush=True,
                )

    OUT.mkdir(parents=True, exist_ok=True)
    sums = {
        f"{label}/{name}": accumulator.sums.cpu().numpy()
        for label, group in (("reach", totals), ("gauge", at_gauges))
        for name, accumulator in group.items()
    }
    common = dict(
        comid=ids[nodes],
        gauge_comid=catalog.comid,
        gauge_name=catalog.names,
        held_gauges=held,
        holdout_fold=a.holdout_fold,
    )
    if a.shards > 1:
        (OUT / "parts").mkdir(exist_ok=True)
        np.savez(OUT / f"parts/part_{a.shard}.npz", issues=len(items), **common, **sums)
        print(f"PART {a.shard} COMPLETE ({(time.perf_counter() - began) / 60:.1f} min)", flush=True)
        return
    write_metrics(sums, common, len(items), a.min_count)
    print(f"COMPLETE: {OUT / 'metrics.npz'} ({(time.perf_counter() - began) / 60:.1f} min)", flush=True)


def write_metrics(sums, common, issues, min_count):
    """Per-reach and per-gauge metrics from additive sums (peaks are not used here)."""
    from tva_flood.evaluation.hydrology import reach_metrics

    arrays = {}
    for key, total in sums.items():
        none = np.zeros(total.shape[:-1])
        m = reach_metrics(total, none, none, none, min_count=min_count)
        for metric in ("nse", "kge", "r", "alpha", "beta", "count", "mean_flow"):
            arrays[f"{key}/{metric}"] = m[metric].astype(np.float32)
    np.savez_compressed(OUT / "metrics.npz", issues=issues, **common, **arrays)


def merge(a):
    parts = sorted((OUT / "parts").glob("part_*.npz"))
    loaded = [np.load(path, allow_pickle=True) for path in parts]
    keys = [k for k in loaded[0].files if "/" in k]
    sums = {k: sum(part[k] for part in loaded) for k in keys}
    common = {k: loaded[0][k] for k in ("comid", "gauge_comid", "gauge_name", "held_gauges", "holdout_fold")}
    issues = int(sum(int(part["issues"]) for part in loaded))
    write_metrics(sums, common, issues, a.min_count)
    print(f"Merged {len(parts)} parts, {issues} issues: {OUT / 'metrics.npz'}")


def markdown(table):
    head = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "---|" * len(table.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def tables():
    """Per-reach table (area, gauge basin) and the saved metrics."""
    d = np.load(OUT / "metrics.npz", allow_pickle=True)
    nodes = pd.read_parquet(
        ROOT / "processed/graph_candidate/nodes.parquet", columns=["comid", "totdasqkm", "streamorde"]
    ).set_index("comid")
    reach = nodes.reindex(d["comid"]).reset_index()
    reach["gauged"] = reach.comid.isin(d["gauge_comid"])
    return d, reach


def curve(d, label, name, key="nse"):
    return d[f"{label}/{name}/{key}"]


def plots(a):
    import geopandas as gpd
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    import comparison_figures as cf
    from comparison_figures import INK, LEGEND, LSTM, MUTED, NWM, SCRATCH, plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    d, reach = tables()
    leads = np.arange(1, 41) * 6
    area = reach.totdasqkm.to_numpy()
    classes = [("< 50 km²", area < 50), ("50-500", (area >= 50) & (area < 500)),
               ("500-5,000", (area >= 500) & (area < 5000)), ("> 5,000", area >= 5000)]
    rows = []

    def median(x, mask=None):
        v = x if mask is None else x[:, mask]
        return np.nanmedian(np.where(np.isfinite(v), v, np.nan), axis=1)

    # 1. agreement by lead -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, (key, title, ylim) in zip(
        axes, (("nse", "NSE against the retrospective", (-0.5, 1)), ("r", "Correlation", (0, 1)), ("beta", "Volume ratio (forecast / retrospective)", (0.4, 1.6)))
    ):
        for name, color, label in (
            ("gnn+aorc", SCRATCH, "Graph model, realized rain"),
            ("gnn+nwm", NWM, "Graph model, NWM forecast rain"),
            ("retro_persistence", LSTM, "Retrospective persistence (oracle)"),
        ):
            ax.plot(leads, median(curve(d, "reach", name, key)), color=color, lw=2, label=label)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Lead (hours)")
        ax.set_ylim(*ylim)
        if key == "beta":
            ax.axhline(1, color=MUTED, lw=0.8)
    axes[0].legend(**LEGEND)
    fig.suptitle(f"All {len(reach):,} reaches, median over reaches; {int(d['issues'])} 00Z issues 2020-2022", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "1_agreement_by_lead.png", dpi=170)
    plt.close(fig)

    # 2. by drainage area ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, lead in zip(axes, (24, 72)):
        i = lead // 6 - 1
        for (name, color, label) in (
            ("gnn+aorc", SCRATCH, "Graph model, realized rain"),
            ("retro_persistence", LSTM, "Retrospective persistence (oracle)"),
        ):
            vals = [np.nanmedian(curve(d, "reach", name)[i][m]) for _, m in classes]
            ax.plot(range(len(classes)), vals, "o-", color=color, lw=2, label=label)
            for k, (_, m) in enumerate(classes):
                rows.append(dict(model=name, lead_h=lead, size=classes[k][0], reaches=int(np.isfinite(curve(d, "reach", name)[i][m]).sum()), median_nse=vals[k]))
        ax.set_xticks(range(len(classes)), [c for c, _ in classes])
        ax.set_xlabel("Drainage area")
        ax.set_title(f"{lead} h", loc="left", color=INK)
        ax.set_ylim(-0.5, 1)
    axes[0].set_ylabel("Median NSE against retrospective")
    axes[0].legend(**LEGEND)
    fig.tight_layout()
    fig.savefig(FIGURES / "2_agreement_by_area.png", dpi=170)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(FIGURES / "agreement_by_area.csv", index=False)

    # 3. the retrospective is itself imperfect: at the gauges ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    for name, color, label in (
        ("retro", NWM, "NWM retrospective vs USGS"),
        ("gnn+aorc", SCRATCH, "Graph model vs USGS"),
    ):
        ax.plot(leads, median(curve(d, "gauge", name)), color=color, lw=2, label=label)
    ax.plot(leads, median(curve(d, "reach", "gnn+aorc"), reach.gauged.to_numpy()), color=INK, lw=2, ls="--", label="Graph model vs retrospective (same gauges)")
    ax.set_xlabel("Lead (hours)")
    ax.set_ylabel("Median NSE at the gauges")
    ax.set_ylim(-0.2, 1)
    ax.set_title("Gauges, realized rain", loc="left", color=INK)
    ax.legend(**LEGEND)
    ax = axes[1]
    i = 72 // 6 - 1
    gauge_ok = np.isfinite(curve(d, "gauge", "retro")[i])
    obs = curve(d, "gauge", "gnn+aorc")[i][gauge_ok]
    ret = curve(d, "gauge", "retro")[i][gauge_ok]
    ax.scatter(ret, obs, s=28, color=SCRATCH, edgecolor=INK, lw=0.4)
    ax.plot([-1, 1], [-1, 1], color=MUTED, lw=0.8)
    ax.set_xlim(-0.5, 1)
    ax.set_ylim(-0.5, 1)
    ax.set_xlabel("NSE of the retrospective vs USGS, 72 h")
    ax.set_ylabel("NSE of the graph model vs USGS, 72 h")
    ax.set_title(f"{gauge_ok.sum()} gauges", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "3_retrospective_vs_observations.png", dpi=170)
    plt.close(fig)

    # 4. map ----------------------------------------------------------------------------
    raw = ROOT / "raw"
    basin = gpd.read_file(raw / "geography/tennessee_huc2.geojson").to_crs(4326)
    lines = gpd.read_file(
        next((raw / "geography/nhdplus_v21/extracted").rglob("NHDFlowline.shp")),
        columns=["COMID", "geometry"],
    ).rename(columns={"COMID": "comid"})
    lines = lines.merge(reach[["comid"]].assign(position=np.arange(len(reach))), on="comid").to_crs(4326)
    segments = [np.asarray(g.geoms[0].coords if g.geom_type == "MultiLineString" else g.coords)[:, :2] for g in lines.geometry]
    for lead in (24, 72):
        i = lead // 6 - 1
        value = curve(d, "reach", "gnn+aorc", "nse")[i][lines.position.to_numpy()]
        value = np.clip(value, -1, 1)
        order = np.argsort(np.where(np.isfinite(value), reach.totdasqkm.to_numpy()[lines.position.to_numpy()], -1))
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.set_aspect(1 / np.cos(np.radians(36)))
        ax.grid(False)
        basin.boundary.plot(ax=ax, color=INK, lw=1.4, zorder=3)
        ok = np.isfinite(value)
        gray = LineCollection([segments[k] for k in np.flatnonzero(~ok)], colors="#d9d8d3", linewidths=0.2)
        ax.add_collection(gray)
        idx = [k for k in order if ok[k]]
        width = 0.3 + 1.6 * np.clip(np.log10(np.maximum(reach.totdasqkm.to_numpy()[lines.position.to_numpy()][idx], 1)) / 4, 0, 1)
        lc = LineCollection([segments[k] for k in idx], array=value[idx], cmap="RdYlBu", norm=TwoSlopeNorm(0.5, -1, 1), linewidths=width)
        ax.add_collection(lc)
        ax.autoscale()
        fig.colorbar(lc, ax=ax, shrink=0.7, label="NSE of the graph model against the retrospective (clipped to ±1)")
        ax.set_title(f"{lead} h lead, realized rain; grey: no usable retrospective series", loc="left", color=INK)
        fig.tight_layout()
        fig.savefig(FIGURES / f"4_map_{lead}h.png", dpi=170)
        plt.close(fig)

    # 5. summary table ----------------------------------------------------------------
    out = []
    for lead in (6, 24, 72, 120, 240):
        i = lead // 6 - 1
        for scope, mask in (("all reaches", np.ones(len(reach), bool)), ("gauge reaches", reach.gauged.to_numpy()), *((f"area {c}", m) for c, m in classes)):
            row = dict(lead_h=lead, scope=scope)
            for name in ("gnn+aorc", "gnn+nwm", "retro_persistence"):
                v = curve(d, "reach", name)[i][mask]
                row[f"{name} median NSE"] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
                row[f"{name} reaches"] = int(np.isfinite(v).sum())
            r = curve(d, "reach", "gnn+aorc", "r")[i][mask]
            row["gnn+aorc median r"] = float(np.nanmedian(r)) if np.isfinite(r).any() else np.nan
            out.append(row)
    table = pd.DataFrame(out)
    table.to_csv(FIGURES / "summary.csv", index=False)
    lines_md = ["# Graph model vs NWM retrospective, all reaches", "", markdown(table.round(3))]
    (FIGURES / "report.md").write_text("\n".join(lines_md) + "\n")
    print(f"Wrote {FIGURES}")


def ungauged(a):
    """Fold models (ungauged mode) and NWM forecasts against the retrospective at every reach.

    A reach belongs to the smallest catalog-gauge basin containing it. Reaches of a basin
    whose gauge is held out in fold k take the fold-k model: forecasts made without that
    gauge's flow, the ungauged setting. Reaches outside every gauge basin are ungauged for
    any model and take the all-gauge model. Written to figures/retrospective_ungauged.
    """
    import geopandas as gpd
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    from comparison_figures import INK, LEGEND, LSTM, MUTED, NWM, PERSIST, SCRATCH, plt
    from tva_flood.data.folds import holdout_folds
    from tva_flood.data.gauges import Catalog
    from tva_flood.data.tva import basin_owner

    figures = ROOT / "figures/retrospective_ungauged"
    figures.mkdir(parents=True, exist_ok=True)
    load = lambda tag: np.load(MODELS / f"retrospective_comparison{tag}/metrics.npz", allow_pickle=True)
    gauged, nwm = load(""), load("_nwm")
    folds = {k: load(f"_heldout_f{k}") for k in range(1, a.folds + 1)}
    catalog = Catalog(GAUGES)
    with np.load(TEST / "graph.npz") as g:
        ids, edges = g["feature_id"].copy(), np.column_stack([g["source"], g["target"]])
    outlet = pd.Index(ids).get_indexer(catalog.comid)
    owner, _ = basin_owner(edges, len(ids), outlet)
    fold_of_gauge = holdout_folds(ROOT, catalog, a.folds)
    fold = np.where(owner >= 0, fold_of_gauge[np.maximum(owner, 0)], -1)  # -1: no gauge basin
    held_basin = fold > 0
    nodes = pd.read_parquet(
        ROOT / "processed/graph_candidate/nodes.parquet", columns=["comid", "totdasqkm"]
    ).set_index("comid")
    area = nodes.totdasqkm.reindex(ids).to_numpy()
    leads = np.arange(1, 41) * 6

    def metric(d, key, metric="nse"):
        return d[f"reach/{key}/{metric}"]

    def composite(key, metric_name="nse"):
        out = metric(gauged, key, metric_name).copy()
        for k, d in folds.items():
            out[:, fold == k] = metric(d, key, metric_name)[:, fold == k]
        return out

    series = {
        "GNN, ungauged (fold model), NWM rain": (composite("gnn+nwm"), SCRATCH, "-"),
        "GNN, ungauged (fold model), realized rain": (composite("gnn+aorc"), SCRATCH, ":"),
        "GNN, gauged (all-gauge model), NWM rain": (metric(gauged, "gnn+nwm"), LSTM, "-"),
        "NWM operational forecast": (metric(nwm, "nwm_forecast"), NWM, "-"),
        "Retrospective persistence (oracle)": (metric(gauged, "retro_persistence"), PERSIST, "--"),
    }
    scopes = [
        ("held-out basins", held_basin),
        ("all reaches", np.ones(len(ids), bool)),
        ("outside every gauge basin", fold == -1),
    ]
    classes = [("< 50 km²", area < 50), ("50-500", (area >= 50) & (area < 500)),
               ("500-5,000", (area >= 500) & (area < 5000)), ("> 5,000", area >= 5000)]

    def med(x, mask):
        v = x[..., mask]
        return np.nanmedian(np.where(np.isfinite(v), v, np.nan), axis=-1)

    # 1. NSE against the retrospective by lead -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (scope, mask) in zip(axes, scopes[:2]):
        for name, (x, color, style) in series.items():
            ax.plot(leads, med(x, mask), color=color, ls=style, label=name)
        ax.set_title(f"{scope} ({mask.sum():,} reaches)")
        ax.set_xlabel("Lead (hours)")
        ax.set_ylim(-1, 1)
    axes[0].set_ylabel("Median NSE against the NWM retrospective")
    axes[0].legend(loc="lower left", **{**LEGEND, "fontsize": 9})
    fig.tight_layout()
    fig.savefig(figures / "1_agreement_by_lead.png", dpi=170)
    plt.close(fig)

    # 2. by drainage area, held-out basins, 24 and 72 h ----------------------------------
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, lead in zip(axes, (24, 72)):
        i = lead // 6 - 1
        for name, (x, color, style) in series.items():
            values = [med(x[i], held_basin & m) for _, m in classes]
            ax.plot(range(len(classes)), values, "o", color=color, ls=style, label=name)
            for (label, m), v in zip(classes, values):
                rows.append(dict(lead_h=lead, size=label, model=name, reaches=int((held_basin & m).sum()), median_nse=float(v)))
        ax.set_xticks(range(len(classes)), [c for c, _ in classes])
        ax.set_xlabel("Drainage area (held-out basins)")
        ax.set_title(f"{lead} h")
        ax.set_ylim(-1.5, 1)
    axes[0].set_ylabel("Median NSE against the retrospective")
    axes[0].legend(loc="lower right", **{**LEGEND, "fontsize": 8})
    fig.tight_layout()
    fig.savefig(figures / "2_agreement_by_area.png", dpi=170)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(figures / "agreement_by_area.csv", index=False)

    # 3. held-out gauges: skill against USGS, and does agreement track it? -------------------
    held_gauge = fold_of_gauge > 0
    usgs = np.full((40, len(catalog.comid)), np.nan)
    retro_agree = np.full((40, len(catalog.comid)), np.nan)
    for k, d in folds.items():
        g = fold_of_gauge == k
        usgs[:, g] = d["gauge/gnn+nwm/nse"][:, g]
        retro_agree[:, g] = d["reach/gnn+nwm/nse"][:, outlet[g]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    for name, x, color, style in (
        ("GNN, ungauged, NWM rain", usgs, SCRATCH, "-"),
        ("NWM operational forecast", nwm["gauge/nwm_forecast/nse"], NWM, "-"),
        ("NWM retrospective (realized rain)", gauged["gauge/retro/nse"], MUTED, "--"),
    ):
        ax.plot(leads, np.nanmedian(x[:, held_gauge], axis=1), color=color, ls=style, label=name)
    ax.set_title(f"Held-out gauges against USGS ({held_gauge.sum()} gauges)")
    ax.set_xlabel("Lead (hours)")
    ax.set_ylabel("Median NSE against USGS")
    ax.set_ylim(-0.6, 1)
    ax.legend(**{**LEGEND, "fontsize": 9})
    ax = axes[1]
    i = 72 // 6 - 1
    ok = held_gauge & np.isfinite(usgs[i]) & np.isfinite(retro_agree[i])
    x, y = usgs[i][ok], retro_agree[i][ok]
    ax.scatter(np.clip(x, -1, 1), np.clip(y, -1, 1), s=26, color=SCRATCH, edgecolor=INK, lw=0.4)
    ax.plot([-1, 1], [-1, 1], color=MUTED, lw=0.8)
    rho = pd.Series(x).corr(pd.Series(y), method="spearman")
    ax.set_title(f"Ungauged GNN at held-out gauges, 72 h: Spearman {rho:.2f}")
    ax.set_xlabel("NSE against USGS (truth)")
    ax.set_ylabel("NSE against the retrospective")
    fig.tight_layout()
    fig.savefig(figures / "3_heldout_gauges.png", dpi=170)
    plt.close(fig)

    # 4. maps: ungauged GNN, NWM forecast, and their difference, 72 h --------------------
    raw = ROOT / "raw"
    basin = gpd.read_file(raw / "geography/tennessee_huc2.geojson").to_crs(4326)
    lines = gpd.read_file(
        next((raw / "geography/nhdplus_v21/extracted").rglob("NHDFlowline.shp")), columns=["COMID", "geometry"]
    ).rename(columns={"COMID": "comid"})
    lines = lines.merge(pd.DataFrame(dict(comid=ids, position=np.arange(len(ids)))), on="comid").to_crs(4326)
    segments = [np.asarray(g.geoms[0].coords if g.geom_type == "MultiLineString" else g.coords)[:, :2] for g in lines.geometry]
    position = lines.position.to_numpy()
    width = 0.25 + 1.6 * np.clip(np.log10(np.maximum(area[position], 1)) / 4, 0, 1)
    i = 72 // 6 - 1
    gnn72 = series["GNN, ungauged (fold model), NWM rain"][0][i][position]
    nwm72 = series["NWM operational forecast"][0][i][position]
    panels = [
        ("Ungauged GNN (NWM rain) vs retrospective, NSE", np.clip(gnn72, -1, 1), "RdYlBu", TwoSlopeNorm(0, -1, 1)),
        ("NWM operational forecast vs retrospective, NSE", np.clip(nwm72, -1, 1), "RdYlBu", TwoSlopeNorm(0, -1, 1)),
        ("Difference (GNN minus NWM), clipped to ±1", np.clip(gnn72 - nwm72, -1, 1), "PuOr", TwoSlopeNorm(0, -1, 1)),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 15))
    for ax, (title, value, cmap, norm) in zip(axes, panels):
        ax.set_aspect(1 / np.cos(np.radians(36)))
        ax.grid(False)
        ok = np.isfinite(value)
        ax.add_collection(LineCollection([segments[k] for k in np.flatnonzero(~ok)], colors="#d9d8d3", linewidths=0.2))
        order = np.flatnonzero(ok)[np.argsort(area[position][ok])]
        lc = LineCollection([segments[k] for k in order], array=value[order], cmap=cmap, norm=norm, linewidths=width[order])
        ax.add_collection(lc)
        basin.boundary.plot(ax=ax, color=INK, lw=1.2, zorder=3)
        ax.autoscale()
        fig.colorbar(lc, ax=ax, shrink=0.75)
        ax.set_title(f"{title}, 72 h")
    fig.tight_layout()
    fig.savefig(figures / "4_maps_72h.png", dpi=150)
    plt.close(fig)

    # 5. report ------------------------------------------------------------------------
    table = []
    for lead in (6, 24, 72, 120, 240):
        i = lead // 6 - 1
        for scope, mask in scopes + [(f"held-out basins, area {c}", held_basin & m) for c, m in classes]:
            row = dict(lead_h=lead, scope=scope, reaches=int(mask.sum()))
            for name, (x, _, _) in series.items():
                row[name] = round(float(med(x[i], mask)), 3)
            table.append(row)
    table = pd.DataFrame(table)
    table.to_csv(figures / "summary.csv", index=False)
    gauge_rows = pd.DataFrame(
        dict(
            lead_h=leads,
            gnn_ungauged_vs_usgs=np.nanmedian(usgs[:, held_gauge], axis=1),
            nwm_forecast_vs_usgs=np.nanmedian(nwm["gauge/nwm_forecast/nse"][:, held_gauge], axis=1),
            retrospective_vs_usgs=np.nanmedian(gauged["gauge/retro/nse"][:, held_gauge], axis=1),
        )
    ).round(3)
    gauge_rows.to_csv(figures / "heldout_gauges_vs_usgs.csv", index=False)
    text = [
        "# Ungauged GNN and NWM forecasts against the NWM retrospective, every reach",
        "",
        f"Issues: {int(gauged['issues'])} 00Z days 2020-2022 (NWM: {int(nwm['issues'])}). Reaches in a held-out",
        "gauge's basin take that fold's model (no flow from that gauge); other reaches the all-gauge",
        "model. The retrospective is NWM itself (v3.0, realized rain), so the NWM forecast shares its",
        "physics: agreement favours NWM by construction. Truth is only at gauges (last table).",
        "",
        "## Median NSE against the retrospective",
        "",
        markdown(table[table.lead_h.isin([24, 72, 120])]),
        "",
        "## Held-out gauges against USGS (median NSE)",
        "",
        markdown(gauge_rows[gauge_rows.lead_h.isin([6, 24, 72, 120, 240])]),
        "",
        f"Spearman correlation over held-out gauges at 72 h between NSE against USGS and NSE against the retrospective: {rho:.2f}",
    ]
    (figures / "report.md").write_text("\n".join(text) + "\n")
    print(f"Wrote {figures}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    c.add_argument("--every", type=int, default=3, help="Use every n-th 00Z issue")
    c.add_argument("--limit", type=int, default=0, help="Only the first n issues (a probe)")
    c.add_argument("--min-count", type=int, default=100)
    c.add_argument("--amp", action="store_true", help="bf16 autocast (less memory)")
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--holdout-fold", type=int, default=0, help="Hide this fold's gauges (fold model)")
    c.add_argument("--folds", type=int, default=5)
    c.add_argument("--reference-only", action="store_true", help="Score NWM forecasts, no model")
    c.add_argument("--nwm-reaches", type=Path, default=ROOT / "processed/benchmarks/nwm_medium_range_reaches")
    c.add_argument("--rain", choices=["nwm", "gefs"], default="nwm", help="Forecast rain")
    c.add_argument("--gefs", type=Path, default=ROOT / "processed/rain_forecasts/gefs_operational_240")
    c.add_argument("--shard", type=int, default=0, help="This process takes issues shard::shards")
    c.add_argument("--shards", type=int, default=1, help="Then run merge")
    sub.add_parser("merge").add_argument("--min-count", type=int, default=100)
    sub.add_parser("plots")
    sub.add_parser("ungauged").add_argument("--folds", type=int, default=5)
    for q in sub.choices.values():
        q.add_argument("--tag", default="", help="Output suffix, for another checkpoint")
    a = p.parse_args()
    global OUT, FIGURES
    if a.tag:
        OUT, FIGURES = OUT.with_name(f"{OUT.name}_{a.tag}"), FIGURES.with_name(f"{FIGURES.name}_{a.tag}")
    if a.command in ("plots", "ungauged"):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
    {"compute": compute, "merge": merge, "plots": plots, "ungauged": ungauged}[a.command](a)


if __name__ == "__main__":
    main()
