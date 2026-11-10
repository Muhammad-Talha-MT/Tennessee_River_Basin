"""Attribution report for the manuscript experiments (docs/ATTRIBUTION.md): what carries the held-out skill?

Reads the saved test forecasts of every evaluated model (``tva_flood.hydro.evaluate`` writes
forecasts_gauged.npz and forecasts_ungauged.npz; ``evaluate-benchmark`` writes the LSTM's forecasts.npz) and
scores them on exactly the same (issue, lead, gauge) values: the reference model's mask (A0, the current GNN), and
for each comparison only values every compared model forecast.

Per gauge: NSE at every lead, NSE and KGE over days 1-5 (mean over 24, 48, 72, 96, 120 h), and the critical
success index of flows above the gauge's training 95th percentile over days 1-5. Across seeds a model's
per-gauge score is the seed mean. Paired differences: median and mean over gauges of the per-gauge difference,
with 95% bootstrap intervals (gauges resampled; seeds resampled too when a model has several) and the Wilcoxon
signed-rank p-value. The 2x2 graph/NWM interaction is (A0 - A5) - (A3 - A6).

  python scripts/attribution_report.py phase1     # T1, F4, the 2x2 and the LSTM checks
  python scripts/attribution_report.py blocked    # S1: random vs blocked held-out gauges
  python scripts/attribution_report.py coverage   # S2: skill against the share of training gauges

Outputs: /data/tva_data/figures/attribution/<section>/ (report.md, CSV tables, PNG figures).
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tva_flood.hydro import paths as P  # noqa: E402
from tva_flood.hydro import splits  # noqa: E402

ATTR = Path(os.environ.get("ATTR_DIR", P.MODELS / "attr"))
OUT = Path(os.environ.get("ATTR_FIGURES", P.FIGURES / "attribution"))
DAYS = [3, 7, 11, 15, 19]  # lead indices of 24-120 h
MIN_COUNT = 100
BOOT = 2000
GAUGED_DEFAULT = ["hydro+gefs", "hydro+aorc", "nwm", "persistence", "climatology"]
UNGAUGED_DEFAULT = ["ungauged+gefs", "ungauged+aorc"]

# id: (label, graph, routing, NWM pretraining, NWM prior in stage B, what it answers)
VARIANTS = {
    "A0": ("Full GNN", "yes", "learned", "yes", "yes", "reference"),
    "A1": ("GNN, no exchange between reaches", "no", "learned", "yes", "yes", "message passing + gauge broadcast"),
    "A2": ("GNN, instantaneous routing", "yes", "instant", "yes", "yes", "learned travel times"),
    "A3": ("GNN without NWM", "yes", "learned", "no", "no", "graph without the simulation"),
    "A4": ("GNN, no stage-B NWM prior", "yes", "learned", "yes", "no", "prior as regularizer"),
    "A5": ("Lumped reach model, same NWM training", "no", "none", "yes", "yes", "graph vs NWM supervision (key control)"),
    "A6": ("Lumped reach model, no NWM", "no", "none", "no", "no", "observation-only baseline"),
}
COMPARISONS = [  # (name, better, worse): difference better - worse
    ("graph | NWM (A0 - A5)", "A0", "A5"),
    ("graph | no NWM (A3 - A6)", "A3", "A6"),
    ("NWM | graph (A0 - A3)", "A0", "A3"),
    ("NWM | lumped (A5 - A6)", "A5", "A6"),
    ("exchange (A0 - A1)", "A0", "A1"),
    ("learned routing (A0 - A2)", "A0", "A2"),
    ("stage-B prior (A0 - A4)", "A0", "A4"),
    ("GNN - LSTM (original)", "A0", "LSTM"),
    ("GNN - LSTM (matched masking)", "A0", "LSTM_matched"),
    ("LSTM matched - original", "LSTM_matched", "LSTM"),
    ("lumped GNN-head - LSTM matched (A6 - LSTM_matched)", "A6", "LSTM_matched"),
]


# ---- loading -------------------------------------------------------------------------------------------------
def gauge_info():
    """Catalog names, CAMELS flag, training 95th percentiles, random folds and the dam groups."""
    from tva_flood.hydro.data import Arrays, Gauges, Network
    from tva_flood.hydro.evaluate import regulation

    network = Network()
    times = Arrays().times
    gauges = Gauges(network, times)
    with np.errstate(all="ignore"):
        p95 = np.nanquantile(gauges.observed[times < P.TRAIN_UNTIL], 0.95, axis=0)
    _, groups = regulation(network, gauges)
    return dict(names=list(gauges.names), p95=p95, groups=dict(groups), gauges=gauges, network=network)


def load_hydro(folder, kind):
    """{method: values [issue, lead, gauge]}, truth and times of a ``hydro.evaluate`` output."""
    d = np.load(Path(folder) / f"forecasts_{kind}.npz", allow_pickle=True)
    methods = list(d["methods"].astype(str)) if "methods" in d.files else (GAUGED_DEFAULT if kind == "gauged" else UNGAUGED_DEFAULT)
    methods = methods[: d["values"].shape[1]]
    values = {m: d["values"][:, i].astype(np.float32) for i, m in enumerate(methods)}
    return dict(time=pd.DatetimeIndex(d["time"]), truth=d["truth"].astype(np.float32), values=values, stations=list(d["stations"].astype(str)))


def load_lstm(template, kind, names, folds=5):
    """LSTM forecasts (``evaluate-benchmark``) on the catalog order: gauged from {suffix}='', held-out from the
    five fold runs. None if missing."""
    parts = [""] if kind == "gauged" else [f"_f{k}" for k in range(1, folds + 1)]
    out = None
    for suffix in parts:
        path = Path(template.format(suffix=suffix)) / "forecasts.npz"
        if not path.exists():
            return None
        d = np.load(path, allow_pickle=True)
        methods = list(d["methods"].astype(str))
        if "lstm+gefs" not in methods:
            return None
        v = d["values"][:, methods.index("lstm+gefs")]  # [issue, lead, station]
        index = pd.Index(names).get_indexer(d["stations"].astype(str))
        if out is None:
            out = dict(time=pd.DatetimeIndex(d["time"]), value=np.full((len(d["time"]), v.shape[1], len(names)), np.nan, np.float32))
        elif not out["time"].equals(pd.DatetimeIndex(d["time"])):
            raise ValueError(f"{path}: other issues than the first fold")
        out["value"][:, :, index] = np.where(np.isfinite(v), v, out["value"][:, :, index])
    return out


def on_times(values, times, wanted):
    """Values [issue, ...] reindexed to ``wanted`` times (NaN where missing)."""
    index = pd.DatetimeIndex(times).get_indexer(wanted)
    out = np.full((len(wanted),) + values.shape[1:], np.nan, np.float32)
    out[index >= 0] = values[index[index >= 0]]
    return out


def collect(kind, info, entries):
    """Reference truth (the first A0 evaluation) and every model's forecasts [issue, lead, gauge] on its issues.

    ``entries``: {model id: list of (source, location)} with source "hydro" (an evaluation folder) or "lstm" (a
    benchmark template); several entries are seeds."""
    reference = None
    for source, where in entries["A0"]:
        if source == "hydro" and (Path(where) / f"forecasts_{kind}.npz").exists():
            reference = load_hydro(where, kind)
            break
    if reference is None:
        raise SystemExit(f"No {kind} evaluation of the reference model A0 found")
    forecasts = {}
    for model, seeds in entries.items():
        for source, where in seeds:
            if source == "hydro":
                if not (Path(where) / f"forecasts_{kind}.npz").exists():
                    continue
                d = load_hydro(where, kind)
                if d["stations"] != info["names"]:
                    raise ValueError(f"{where}: other gauges")
                key = "hydro+gefs" if kind == "gauged" else "ungauged+gefs"
                forecasts.setdefault(model, []).append(on_times(d["values"][key], d["time"], reference["time"]))
            else:
                d = load_lstm(where, kind, info["names"])
                if d is not None:
                    forecasts.setdefault(model, []).append(on_times(d["value"], d["time"], reference["time"]))
    return reference["truth"], forecasts


# ---- scores --------------------------------------------------------------------------------------------------
def scores(pred, truth, p95):
    """Per-gauge scores of ``pred`` [issue, lead, gauge] against ``truth``: dict of nse [lead, gauge], nse_d15, kge_d15,
    csi_d15 [gauge]; NaN where fewer than ``MIN_COUNT`` values or no variance."""
    ok = np.isfinite(pred) & np.isfinite(truth)
    n = ok.sum(0)
    o = np.where(ok, truth, 0.0).astype(np.float64)
    f = np.where(ok, pred, 0.0).astype(np.float64)
    with np.errstate(all="ignore"):
        mo, mf = o.sum(0) / n, f.sum(0) / n
        vo = (o * o).sum(0) / n - mo**2
        vf = (f * f).sum(0) / n - mf**2
        cov = (o * f).sum(0) / n - mo * mf
        mse = ((f - o) ** 2).sum(0) / n
        good = (n >= MIN_COUNT) & (vo > 1e-6) & (mo > 0)
        nse = np.where(good, 1 - mse / vo, np.nan)
        r = cov / np.sqrt(vo * vf)
        alpha, beta = np.sqrt(vf / vo), mf / mo
        kge = np.where(good, 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2), np.nan)
    d = np.zeros(truth.shape[1], bool)
    d[DAYS[0] : DAYS[-1] + 1] = True  # leads 24-120 h for the event score
    event_o = (truth[:, d] > p95[None, None]) & ok[:, d]
    event_f = (pred[:, d] > p95[None, None]) & ok[:, d]
    hits = (event_o & event_f).sum((0, 1))
    misses = (event_o & ~event_f).sum((0, 1))
    false = (~event_o & event_f).sum((0, 1))
    with np.errstate(all="ignore"):
        csi = np.where(hits + misses >= 5, hits / (hits + misses + false), np.nan)
    with np.errstate(all="ignore"):
        return dict(nse=nse, nse_d15=np.nanmean(nse[DAYS], 0) if np.isfinite(nse[DAYS]).any() else np.full(nse.shape[1], np.nan),
                    kge_d15=np.mean(kge[DAYS], 0), csi_d15=csi)


def model_scores(forecasts, truth, p95):
    """{model: [seed scores]}: every model scored on the reference mask."""
    return {m: [scores(p, truth, p95) for p in seeds] for m, seeds in forecasts.items()}


def seed_mean(seeds, key):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack([s[key] for s in seeds]), axis=0)


def paired(better, worse, key, gauges, rng):
    """Paired difference of per-gauge ``key`` (seed means), over ``gauges`` [bool] scored by both: median and mean with
    95% bootstrap intervals (gauges and seeds resampled), Wilcoxon p, gauge count, share of gauges improved."""
    from scipy.stats import wilcoxon

    a, b = np.stack([s[key] for s in better]), np.stack([s[key] for s in worse])  # [seed, ...gauge]
    if a.ndim == 3:  # by lead: [seed, lead, gauge]
        raise ValueError("paired() takes per-gauge scores")
    both = gauges & np.isfinite(a).all(0) & np.isfinite(b).all(0)
    if both.sum() < 5:
        return None
    a, b = a[:, both], b[:, both]
    diff = a.mean(0) - b.mean(0)
    reps_med, reps_mean = [], []
    g = diff.size
    for _ in range(BOOT):
        pick = rng.integers(0, g, g)
        sa = a[rng.integers(0, len(a))] if len(a) > 1 else a[0]
        sb = b[rng.integers(0, len(b))] if len(b) > 1 else b[0]
        d = sa[pick] - sb[pick]
        reps_med.append(np.median(d))
        reps_mean.append(d.mean())
    try:
        p = float(wilcoxon(diff).pvalue)
    except ValueError:
        p = np.nan
    return dict(
        gauges=int(g), median=float(np.median(diff)), median_lo=float(np.quantile(reps_med, 0.025)), median_hi=float(np.quantile(reps_med, 0.975)),
        mean=float(diff.mean()), mean_lo=float(np.quantile(reps_mean, 0.025)), mean_hi=float(np.quantile(reps_mean, 0.975)),
        wilcoxon_p=p, improved=float((diff > 0).mean()), seeds_better=len(a), seeds_worse=len(b),
    )


def lead_curve(better, worse, gauges, rng, boots=500):
    """Median over gauges of the per-gauge NSE difference at every lead, with a 95% gauge-bootstrap band."""
    a, b = seed_mean(better, "nse"), seed_mean(worse, "nse")  # [lead, gauge]
    mid, lo, hi = [], [], []
    for lead in range(a.shape[0]):
        ok = gauges & np.isfinite(a[lead]) & np.isfinite(b[lead])
        d = a[lead, ok] - b[lead, ok]
        if d.size < 5:
            mid.append(np.nan), lo.append(np.nan), hi.append(np.nan)
            continue
        reps = [np.median(d[rng.integers(0, d.size, d.size)]) for _ in range(boots)]
        mid.append(np.median(d)), lo.append(np.quantile(reps, 0.025)), hi.append(np.quantile(reps, 0.975))
    return np.array(mid), np.array(lo), np.array(hi)


def median_row(seeds, key, gauges):
    v = seed_mean(seeds, key)
    with np.errstate(all="ignore"):
        return float(np.nanmedian(v[gauges])) if np.isfinite(v[gauges]).any() else np.nan


# ---- cost ----------------------------------------------------------------------------------------------------
def run_cost(folder):
    """(trainable parameters, GPU-hours) of a training run from its last checkpoint and validation log."""
    folder = Path(folder)
    if not (folder / "validation.json").exists():
        return np.nan, np.nan
    log = json.loads((folder / "validation.json").read_text())
    import torch

    state = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
    gpus = state.get("gpus", 4)  # runs before the attribution code: 4 GPUs (scripts/gnn_pipeline.sh)
    params = state.get("parameters") or sum(v.numel() for v in state["model"].values())
    return int(params), float(log[-1]["minutes"]) * gpus / 60


# ---- sections ------------------------------------------------------------------------------------------------
def entries_for(model, seeds):
    """Evaluation sources of a model over ``seeds``."""
    if model == "LSTM":
        return [("lstm", str(P.MODELS / "gefs_benchmark{suffix}"))]
    if model == "LSTM_matched":
        return [("lstm", str(ATTR / f"lstm_matched_s{s}_bench{{suffix}}")) for s in seeds]
    return [("hydro", str(ATTR / f"eval_{model}_s{s}")) for s in seeds]


def fmt(x, digits=3):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def phase1(a, info):
    out = OUT / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    models = list(VARIANTS) + ["LSTM", "LSTM_matched"]
    entries = {m: entries_for(m, a.seeds) for m in models}
    entries = {"A0": entries.pop("A0"), **entries}  # the reference first
    names = info["names"]
    held = splits.gauge_folds("random", info["gauges"]) > 0
    subsets = {"all": np.ones(len(names), bool), "headwater, no dam": info["groups"]["headwater, no dam"]}
    lines = [
        "# Attribution experiments, phase 1 (Exp.1 and the LSTM checks)",
        "",
        "Test: 00Z issues 2020-09-23 to 2025-12, GEFSv12 operational rain; every model scored on the same (issue, lead, gauge)",
        "values as the full GNN (A0). Per-gauge scores are seed means; intervals are 95% bootstrap over gauges (and seeds).",
        "",
    ]
    table, curves = [], {}
    for kind, label in (("ungauged", "held-out gauges"), ("gauged", "gauges with their flow history")):
        truth, forecasts = collect(kind, info, entries)
        sc = model_scores(forecasts, truth, info["p95"])
        base = held if kind == "ungauged" else np.ones(len(names), bool)
        for m in models:
            if m not in sc:
                continue
            for subset, mask in subsets.items():
                g = base & mask
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    per_lead = float(np.mean([np.nanmedian(seed_mean(sc[m], "nse")[lead][g]) for lead in DAYS]))
                table.append(dict(protocol=kind, subset=subset, model=m, seeds=len(sc[m]), nse_d15=median_row(sc[m], "nse_d15", g),
                                  nse_d15_as_report=per_lead,
                                  kge_d15=median_row(sc[m], "kge_d15", g), csi_d15=median_row(sc[m], "csi_d15", g),
                                  nse_24h=float(np.nanmedian(seed_mean(sc[m], "nse")[3][g])), nse_120h=float(np.nanmedian(seed_mean(sc[m], "nse")[19][g])),
                                  nse_240h=float(np.nanmedian(seed_mean(sc[m], "nse")[39][g]))))
        lines += [f"## Paired differences, {label}", ""]
        lines += ["| comparison | subset | metric | gauges | median diff [95% CI] | mean diff [95% CI] | gauges improved | Wilcoxon p |",
                  "|---|---|---|---:|---|---|---:|---:|"]
        diffs = []
        for name, better, worse in COMPARISONS:
            if better not in sc or worse not in sc:
                continue
            for subset, mask in subsets.items():
                for key in ("nse_d15", "kge_d15", "csi_d15"):
                    r = paired(sc[better], sc[worse], key, base & mask, rng)
                    if r is None:
                        continue
                    diffs.append(dict(protocol=kind, comparison=name, subset=subset, metric=key, **r))
                    lines.append(f"| {name} | {subset} | {key} | {r['gauges']} | {r['median']:+.3f} [{r['median_lo']:+.3f}, {r['median_hi']:+.3f}] | "
                                 f"{r['mean']:+.3f} [{r['mean_lo']:+.3f}, {r['mean_hi']:+.3f}] | {100 * r['improved']:.0f}% | {r['wilcoxon_p']:.3g} |")
            curves[(kind, name)] = lead_curve(sc[better], sc[worse], base, rng)
        if all(m in sc for m in ("A0", "A3", "A5", "A6")):  # 2x2 interaction
            lines += ["", "2x2 interaction (graph effect with NWM minus graph effect without NWM), NSE days 1-5:", ""]
            for subset, mask in subsets.items():
                both = base & mask
                v = {m: np.stack([s["nse_d15"] for s in sc[m]]) for m in ("A0", "A3", "A5", "A6")}
                ok = both & np.all([np.isfinite(x).all(0) for x in v.values()], axis=0)
                inter = (v["A0"].mean(0) - v["A5"].mean(0)) - (v["A3"].mean(0) - v["A6"].mean(0))
                reps = []
                idx = np.flatnonzero(ok)
                for _ in range(BOOT):
                    pick = idx[rng.integers(0, idx.size, idx.size)]
                    pickm = {m: x[rng.integers(0, len(x))] for m, x in v.items()}
                    reps.append(np.median((pickm["A0"] - pickm["A5"] - pickm["A3"] + pickm["A6"])[pick]))
                lines.append(f"- {subset} ({idx.size} gauges): median {np.median(inter[ok]):+.3f} [{np.quantile(reps, 0.025):+.3f}, {np.quantile(reps, 0.975):+.3f}]")
        pd.DataFrame(diffs).to_csv(out / f"paired_{kind}.csv", index=False)
        lines.append("")
    table = pd.DataFrame(table)
    table.to_csv(out / "scores.csv", index=False)
    # T1: the ablation table
    cost_rows = []
    for m in VARIANTS:
        params, hours = [], 0.0
        for s in a.seeds:
            for k in range(6):
                folder = ATTR / (f"{m}_s{s}" + (f"_f{k}" if k else ""))
                p, h = run_cost(folder)
                if np.isfinite(p):
                    params.append(p)
                    hours += h
        stage_a = {"A0": P.MODELS / "gefs_hydro_A", "A1": P.MODELS / "gefs_hydro_A", "A2": P.MODELS / "gefs_hydro_A", "A4": P.MODELS / "gefs_hydro_A", "A5": ATTR / "lumped_A"}.get(m)
        cost_rows.append(dict(model=m, parameters=params[0] if params else np.nan, gpu_hours_stage_b=hours,
                              gpu_hours_stage_a=run_cost(stage_a)[1] if stage_a else 0.0))
    cost = pd.DataFrame(cost_rows).set_index("model")
    lines += ["## T1. Ablation table (median over gauges of seed-mean scores, days 1-5)", "",
              "| id | model | graph | routing | NWM pretrain | NWM prior | parameters | GPU-h (B / A) | gauged NSE | held-out NSE | held-out headwater NSE | held-out KGE | held-out CSI (> p95) |",
              "|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|"]
    pick = lambda prot, sub, m, k: table.query("protocol == @prot and subset == @sub and model == @m")[k].squeeze() if len(table.query("protocol == @prot and subset == @sub and model == @m")) else np.nan  # noqa: E731
    for m in models:
        label, graph, routing, pre, prior, _ = VARIANTS.get(m, (m.replace("_", " "), "no", "none", "no", "no", ""))
        params = cost.parameters.get(m, np.nan) if m in cost.index else np.nan
        hours = f"{cost.gpu_hours_stage_b[m]:.0f} / {cost.gpu_hours_stage_a[m]:.0f}" if m in cost.index else "-"
        lines.append(f"| {m} | {label} | {graph} | {routing} | {pre} | {prior} | {'-' if not np.isfinite(params) else f'{int(params):,}'} | {hours} | "
                     f"{fmt(pick('gauged', 'all', m, 'nse_d15'))} | {fmt(pick('ungauged', 'all', m, 'nse_d15'))} | {fmt(pick('ungauged', 'headwater, no dam', m, 'nse_d15'))} | "
                     f"{fmt(pick('ungauged', 'all', m, 'kge_d15'))} | {fmt(pick('ungauged', 'all', m, 'csi_d15'))} |")
    lines += ["", "LSTM: the tuned LSTM baseline (models/gefs_benchmark*); LSTM_matched: retrained with the GNN's history masking and selection.",
              "GPU-hours: stage B summed over the six models (all gauges and five folds) and seeds; stage A once per model family."]
    figure_f4(curves, out)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"written: {out}")


def figure_f4(curves, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hours = 6 * np.arange(1, 41)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, kind, title in zip(axes, ("ungauged", "gauged"), ("Held-out gauges", "Gauges with flow history")):
        for (k, name), (mid, lo, hi) in curves.items():
            if k != kind or "LSTM" in name:
                continue
            (line,) = ax.plot(hours, mid, label=name)
            ax.fill_between(hours, lo, hi, color=line.get_color(), alpha=0.15)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set(title=title, xlabel="lead (h)", xticks=[24, 72, 120, 168, 240])
    axes[0].set_ylabel("median per-gauge NSE difference")
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / "f4_ablation_by_lead.png", dpi=160)
    plt.close(fig)


def blocked(a, info):
    """S1: the same models on random-fold and blocked held-out gauges."""
    out = OUT / "blocked"
    out.mkdir(parents=True, exist_ok=True)
    gauge_fold = splits.gauge_folds("blocked", info["gauges"], info["network"])
    lines = ["# S1. Blocked spatial transfer", "", "Blocks (complete upstream basins; every catalog gauge inside withheld):", ""]
    for k, (g, basin) in enumerate(splits.blocks(info["gauges"], info["network"]), start=1):
        lines.append(f"- block {k}: outlet {info['names'][g]}, {info['gauges'].area[g]:,.0f} km2, {basin.size:,} reaches, "
                     f"{int((gauge_fold == k).sum())} scored gauges")
    lines += ["", "| model | split | gauges | median NSE days 1-5 | KGE | median NSE, same gauges, random folds |", "|---|---|---:|---:|---:|---:|"]
    rows = []
    for m in a.models:
        for scheme, folder in (("blocked", ATTR / f"eval_S1_{m}_s1"), ("random", ATTR / f"eval_{m}_s1")):
            if not (folder / "forecasts_ungauged.npz").exists():
                continue
            d = load_hydro(folder, "ungauged")
            sc = scores(d["values"]["ungauged+gefs"], d["truth"], info["p95"])
            rows.append(dict(model=m, scheme=scheme, sc=sc))
    for m in a.models:
        got = {r["scheme"]: r["sc"] for r in rows if r["model"] == m}
        if "blocked" not in got:
            continue
        g = gauge_fold > 0
        same = got.get("random")
        lines.append(f"| {m} | blocked | {int(np.isfinite(got['blocked']['nse_d15'][g]).sum())} | {fmt(np.nanmedian(got['blocked']['nse_d15'][g]))} | "
                     f"{fmt(np.nanmedian(got['blocked']['kge_d15'][g]))} | {fmt(np.nanmedian(same['nse_d15'][g])) if same else '-'} |")
    lines += ["", "The last column scores the same blocked gauges under the random folds, where their neighbours were in training:",
              "the drop from it to the blocked score is the value of nearby gauges."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def coverage(a, info):
    """S2: held-out skill against the share of training gauges (0 = the stage-A model, 1 = phase 1)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = OUT / "coverage"
    out.mkdir(parents=True, exist_ok=True)
    held = splits.gauge_folds("random", info["gauges"]) > 0
    rows = []
    for m in a.models:
        for keep in (0.0, 0.25, 0.5, 1.0):
            folder = ATTR / (f"eval_{m}_s1" if keep == 1.0 else f"eval_S2_{m}_k{int(100 * keep):03d}")
            if not (folder / "forecasts_ungauged.npz").exists():
                continue
            d = load_hydro(folder, "ungauged")
            sc = scores(d["values"]["ungauged+gefs"], d["truth"], info["p95"])
            v = sc["nse_d15"][held]
            rng = np.random.default_rng(0)
            v = v[np.isfinite(v)]
            reps = [np.median(v[rng.integers(0, v.size, v.size)]) for _ in range(BOOT)]
            rows.append(dict(model=m, keep=keep, gauges=v.size, nse_d15=np.median(v), lo=np.quantile(reps, 0.025), hi=np.quantile(reps, 0.975)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "coverage.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, part in df.groupby("model"):
        ax.errorbar(100 * part.keep, part.nse_d15, yerr=[part.nse_d15 - part.lo, part.hi - part.nse_d15], marker="o", capsize=3, label=m)
    ax.set(xlabel="training gauges kept (%)", ylabel="held-out median NSE, days 1-5", xticks=[0, 25, 50, 100])
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "f5_coverage.png", dpi=160)
    lines = ["# S2. Gauge coverage", "", df.to_markdown(index=False, floatfmt=".3f"), "",
             "0%: the stage-A model (NWM only, no gauge ever seen); 100%: the phase-1 fold models (seed 1)."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("section", choices=["phase1", "blocked", "coverage"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="Seeds to include where they exist")
    p.add_argument("--models", nargs="+", default=["A0", "A5"], help="blocked/coverage: models")
    a = p.parse_args()
    info = gauge_info()
    dict(phase1=phase1, blocked=blocked, coverage=coverage)[a.section](a, info)


if __name__ == "__main__":
    main()
