"""Analyses of the attribution experiments that need no training (docs/ATTRIBUTION.md).

  leakage       P2: how well the NWM v3.0 retrospective already matches each gauge (six-hour NSE against USGS,
                2000-2018 and 2020-09-23 to 2022-12-31). The retrospective is the stage-A target; where it is
                good, a "held-out" gauge is partly known to any model trained on it. Splits held-out skill by it
                (and by an optional list of NWM calibration gauges, --calibration CSV with a usgs_id column).
  transfer      S3: held-out skill and the graph gain (A0 - A5) against river distance to the nearest training
                gauge (upstream, downstream, either), the share of the basin drained through training gauges,
                drainage area, distance to the outlet and dams.
  conservation  Exp.3 (GPU): routing mass balance at every reach, basin continuity at confluences, and whether the
                gauge head (which replaces the routed flow at visible gauges) breaks continuity with the reaches
                below it.
  zero-coverage S2 0% level: the stage-A models (NWM only, no gauge) written as fold checkpoints that see no
                gauge, for tva_flood.hydro.evaluate.

Outputs: /data/tva_data/figures/attribution/<command>/.
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
DAYS = [3, 7, 11, 15, 19]


def context():
    from tva_flood.hydro.data import Arrays, Gauges, Network
    from tva_flood.hydro.evaluate import regulation

    network = Network()
    arrays = Arrays()
    gauges = Gauges(network, arrays.times)
    regulated, groups = regulation(network, gauges)
    return dict(network=network, arrays=arrays, gauges=gauges, groups=dict(groups), regulated=regulated)


def heldout_nse(folder, gauges):
    """Per-gauge held-out NSE over days 1-5 (mean of the per-lead NSE) from an evaluation folder, NaN if missing."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from attribution_report import load_hydro, scores

    path = Path(folder)
    if not (path / "forecasts_ungauged.npz").exists():
        return None
    d = load_hydro(path, "ungauged")
    p95 = np.full(gauges.count, np.inf)  # only the NSE is used here
    return scores(d["values"]["ungauged+gefs"], d["truth"], p95)["nse_d15"]


def heldout_table(gauges):
    """Held-out NSE days 1-5 per gauge of the models evaluated so far (A0 seed 1 falls back to gnn_evaluation)."""
    out = {}
    for m in ("A0", "A5", "A1", "A3", "A6"):
        v = heldout_nse(ATTR / f"eval_{m}_s1", gauges)
        if v is None and m == "A0":
            v = heldout_nse(P.MODELS / "gnn_evaluation", gauges)
        if v is not None:
            out[m] = v
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from attribution_report import load_lstm, scores

    for name, template in (("LSTM", str(P.MODELS / "gefs_benchmark{suffix}")), ("LSTM_matched", str(ATTR / "lstm_matched_s1_bench{suffix}"))):
        d = load_lstm(template, "ungauged", list(gauges.names))
        if d is not None:
            truth = np.load(P.MODELS / "gnn_evaluation/forecasts_ungauged.npz")
            t = pd.DatetimeIndex(truth["time"])
            index = d["time"].get_indexer(t)
            pred = np.full(truth["truth"].shape, np.nan, np.float32)
            pred[index >= 0] = d["value"][index[index >= 0]]
            with np.errstate(all="ignore"):
                p95 = np.nanquantile(gauges.observed, 0.95, axis=0)
            out[name] = scores(pred, truth["truth"], p95)["nse_d15"]
    return pd.DataFrame(out, index=gauges.names)


# ---- P2 ------------------------------------------------------------------------------------------------------
def retrospective_at_gauges(arrays, gauges):
    """NWM retrospective flow [time, gauge] (m3/s, NaN invalid) at the gauge reaches, to the retrospective's end."""
    stop = min(arrays.steps, int((P.RETRO_UNTIL - P.START) / pd.Timedelta(hours=6)))
    flows = []
    for k, (first, steps, _) in enumerate(arrays.parts):
        if first >= stop:
            break
        month = arrays.month(k)
        z = np.asarray(month["flow"][:, gauges.reach], dtype=np.float32)
        ok = np.asarray(month["flow_valid"][:, gauges.reach]).astype(bool)
        flows.append(np.where(ok, arrays.flow_raw(z), np.nan))
    return np.concatenate(flows)[:stop]


def nse(sim, obs):
    ok = np.isfinite(sim) & np.isfinite(obs)
    out = np.full(sim.shape[1], np.nan)
    for g in range(sim.shape[1]):
        o, s = obs[ok[:, g], g], sim[ok[:, g], g]
        if o.size >= 400 and o.var() > 1e-6:
            out[g] = 1 - ((s - o) ** 2).mean() / o.var()
    return out


def leakage(a):
    from scipy.stats import spearmanr

    c = context()
    gauges, arrays = c["gauges"], c["arrays"]
    out = OUT / "leakage"
    out.mkdir(parents=True, exist_ok=True)
    retro = retrospective_at_gauges(arrays, gauges)
    times = arrays.times[: len(retro)]
    obs = gauges.observed[: len(retro)]
    train = (times >= pd.Timestamp("2000-01-01")) & (times < P.TRAIN_UNTIL)
    test = (times >= pd.Timestamp("2020-09-23")) & (times < P.RETRO_UNTIL)
    fold = splits.random_folds(gauges)
    table = pd.DataFrame(dict(fold=fold, retro_nse_train=nse(retro[train], obs[train]), retro_nse_test=nse(retro[test], obs[test]),
                              area_km2=gauges.area, headwater_no_dam=c["groups"]["headwater, no dam"]), index=gauges.names)
    if a.calibration:
        ids = set(pd.read_csv(a.calibration, dtype=str)["usgs_id"].str.strip().str.zfill(8))
        table["nwm_calibration"] = table.index.isin(ids)  # catalog names are USGS site numbers
    skill = heldout_table(gauges)
    table = table.join(skill)
    table.to_csv(out / "gauges.csv")
    held = table[table.fold > 0]
    lines = ["# P2. Is the held-out test leaky through the NWM retrospective?", "",
             "Retrospective NSE: the NWM v3.0 retrospective (stage-A target) against USGS, six-hour means. Held-out NSE: forecasts",
             "with GEFS rain, days 1-5. A strong link between the two means part of the held-out skill comes from the simulation",
             "the model was pre-trained on, which may itself be calibrated to these gauges.", "",
             f"Held-out gauges: {len(held)}; retrospective NSE 2000-2018 median {held.retro_nse_train.median():.3f}, "
             f"2020-09 to 2022 median {held.retro_nse_test.median():.3f}.", "",
             "| model | Spearman with retro NSE (2000-2018) | p | held-out NSE, retro NSE below median | above median |", "|---|---:|---:|---:|---:|"]
    cut = held.retro_nse_train.median()
    for m in skill.columns:
        ok = held[[m, "retro_nse_train"]].dropna()
        if len(ok) < 10:
            continue
        r = spearmanr(ok[m], ok.retro_nse_train)
        low, high = ok[ok.retro_nse_train < cut][m].median(), ok[ok.retro_nse_train >= cut][m].median()
        lines.append(f"| {m} | {r.statistic:+.3f} | {r.pvalue:.3g} | {low:.3f} | {high:.3f} |")
    if "A0" in skill and "A5" in skill:
        held = held.assign(graph_gain=held.A0 - held.A5)
        ok = held[["graph_gain", "retro_nse_train"]].dropna()
        r = spearmanr(ok.graph_gain, ok.retro_nse_train)
        lines += ["", f"Graph gain A0 - A5 against retro NSE: Spearman {r.statistic:+.3f} (p {r.pvalue:.3g}); "
                  f"median gain {ok[ok.retro_nse_train < cut].graph_gain.median():+.3f} where the retrospective is poor, "
                  f"{ok[ok.retro_nse_train >= cut].graph_gain.median():+.3f} where it is good."]
    if "nwm_calibration" in table:
        lines += ["", "| model | calibration gauges | median NSE | other gauges | median NSE |", "|---|---:|---:|---:|---:|"]
        for m in skill.columns:
            cal, other = held[held.nwm_calibration][m].dropna(), held[~held.nwm_calibration][m].dropna()
            lines.append(f"| {m} | {len(cal)} | {cal.median():.3f} | {len(other)} | {other.median():.3f} |")
    else:
        lines += ["", "No NWM calibration list given (--calibration): only the retrospective-skill split is reported."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- S3 ------------------------------------------------------------------------------------------------------
def transfer(a):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.stats import spearmanr

    c = context()
    network, gauges = c["network"], c["gauges"]
    out = OUT / "transfer"
    out.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_parquet(P.PROCESSED / "graph_candidate/nodes.parquet", columns=["comid", "lengthkm", "pathlength", "totdasqkm"]).set_index("comid").reindex(network.ids)
    length = np.nan_to_num(nodes.lengthkm.to_numpy(float), nan=0.5)
    s, t = network.edges[:, 0], network.edges[:, 1]
    w = 0.5 * (length[s] + length[t]) + 1e-6
    down = csr_matrix((w, (s, t)), shape=(network.n, network.n))  # source -> target, downstream
    fold = splits.random_folds(gauges)
    rows = []
    for k in range(1, splits.RANDOM_FOLDS + 1):
        train = np.flatnonzero((fold != k) & gauges.trained)
        sources = gauges.reach[train]
        from_up = dijkstra(down, directed=True, indices=sources, min_only=True)  # distance from the nearest training gauge upstream
        from_down = dijkstra(down.T.tocsr(), directed=True, indices=sources, min_only=True)  # ... downstream
        either = dijkstra(down, directed=False, indices=sources, min_only=True)
        covered = np.zeros(network.n, bool)
        for g in train:
            covered[gauges.basins[g]] = True
        for g in np.flatnonzero(fold == k):
            r = gauges.reach[g]
            basin = gauges.basins[g]
            rows.append(dict(gauge=gauges.names[g], fold=k, area_km2=float(gauges.area[g]),
                             km_to_training_gauge_upstream=float(from_up[r]), km_to_training_gauge_downstream=float(from_down[r]),
                             km_to_training_gauge=float(either[r]),
                             upstream_area_gauged=float(network.area[basin][covered[basin]].sum() / network.area[basin].sum()),
                             km_to_outlet=float(nodes.pathlength.iloc[r]), dam_in_basin=bool(c["groups"]["dam in the basin"][g]),
                             headwater_no_dam=bool(c["groups"]["headwater, no dam"][g])))
    table = pd.DataFrame(rows).set_index("gauge")
    table = table.join(heldout_table(gauges))
    if "A0" in table and "A5" in table:
        table["graph_gain"] = table.A0 - table.A5
    table.to_csv(out / "gauges.csv")
    lines = ["# S3. What predicts transfer to a held-out gauge?", "",
             "Per held-out gauge (random folds): river distances (km along the network) to the nearest gauge in training,",
             "the share of its basin area drained through training gauges, and its held-out NSE (days 1-5).", "",
             "| covariate | " + " | ".join(f"Spearman with {m}" for m in ("A0", "A5", "LSTM", "graph_gain") if m in table) + " |",
             "|---|" + "---:|" * sum(m in table for m in ("A0", "A5", "LSTM", "graph_gain"))]
    covariates = ["km_to_training_gauge", "km_to_training_gauge_upstream", "km_to_training_gauge_downstream", "upstream_area_gauged", "area_km2", "km_to_outlet"]
    for cov in covariates:
        cells = []
        for m in ("A0", "A5", "LSTM", "graph_gain"):
            if m not in table:
                continue
            ok = table[[cov, m]].replace([np.inf, -np.inf], np.nan).dropna()
            r = spearmanr(ok[cov], ok[m]) if len(ok) > 10 else None
            cells.append("-" if r is None else f"{r.statistic:+.2f} (p {r.pvalue:.2g}, n {len(ok)})")
        lines.append(f"| {cov} | " + " | ".join(cells) + " |")
    table["has_training_gauge_upstream"] = np.isfinite(table.km_to_training_gauge_upstream)
    lines += ["", "Median held-out NSE by group:", ""]
    groups = [("training gauge upstream", table.has_training_gauge_upstream), ("no training gauge upstream", ~table.has_training_gauge_upstream),
              ("dam in basin", table.dam_in_basin), ("headwater, no dam", table.headwater_no_dam)]
    models = [m for m in ("A0", "A5", "LSTM", "graph_gain") if m in table]
    lines += ["| group | gauges | " + " | ".join(models) + " |", "|---|---:|" + "---:|" * len(models)]
    for name, mask in groups:
        lines.append(f"| {name} | {int(mask.sum())} | " + " | ".join(f"{table[mask][m].median():.3f}" for m in models) + " |")
    if "graph_gain" in table:  # linear model of the graph gain
        x = table[["graph_gain", "km_to_training_gauge", "upstream_area_gauged", "area_km2", "dam_in_basin"]].replace([np.inf, -np.inf], np.nan).dropna()
        design = np.column_stack([np.ones(len(x)), np.log1p(x.km_to_training_gauge), x.upstream_area_gauged, np.log(x.area_km2), x.dam_in_basin.astype(float)])
        coef, *_ = np.linalg.lstsq(design, x.graph_gain.clip(-1, 1), rcond=None)
        lines += ["", f"Least squares, graph gain (clipped to +-1) on log(1 + km to a training gauge), gauged basin share, log area, dam ({len(x)} gauges):",
                  "", "| term | coefficient |", "|---|---:|"] + [f"| {n} | {v:+.3f} |" for n, v in zip(["intercept", "log1p km", "gauged share", "log area", "dam"], coef)]
    figure_transfer(table, out)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def figure_transfer(table, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = "graph_gain" if "graph_gain" in table else "A0"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, cov, log in zip(axes, ("km_to_training_gauge", "upstream_area_gauged", "area_km2"), (True, False, True)):
        x = table[cov].replace([np.inf, -np.inf], np.nan)
        ax.scatter(x, table[target].clip(-1, 1), s=14, c=np.where(table.dam_in_basin, "tab:red", "tab:blue"))
        ax.set(xlabel=cov, ylabel=f"{target} (clipped to +-1)", xscale="log" if log else "linear")
        ax.axhline(0, color="0.5", lw=0.8)
    axes[0].set_title("red: dam in basin")
    fig.tight_layout()
    fig.savefig(out / "f5_transfer_covariates.png", dpi=160)
    plt.close(fig)


# ---- Exp.3 ---------------------------------------------------------------------------------------------------
def conservation(a):
    import torch

    from tva_flood.hydro.evaluate import load_model
    from tva_flood.hydro.samples import Device, Samples

    out = OUT / "conservation"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False  # exact float32 sums for the balance
    s = Samples()
    network, gauges = s.network, s.gauges
    model, state = load_model(a.model, s, device)
    if state["config"].get("kind") == "lumped":
        raise SystemExit("A lumped model has no routing: nothing to audit")
    gpu = Device(network, gauges, s.arrays, device)
    gefs = sorted(s.rain.days("gefs"))
    issues = [d for d in gefs if pd.Timestamp("2020-10-01") <= d <= pd.Timestamp("2025-11-30")][:: a.every][: a.issues]
    router = model.router
    a_full = router.full[0]  # [group * n + i, member] shares
    n, groups = network.n, router.groups
    edges = network.edges
    parents_of = [[] for _ in range(n)]
    for src, dst in edges:
        parents_of[dst].append(src)
    downstream = np.full(n, -1)
    for src, dst in edges:
        downstream[src] = dst  # last wins at divergences (rare)
    head_of = np.full(n, -1)
    head_of[gauges.reach] = np.arange(gauges.count)
    records, masses, confluence = [], [], []
    from tva_flood.hydro.data import timeline_index

    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for day in issues:
            step = timeline_index(day)
            view = s.gauge_view(step, s.eligible(step, np.zeros(gauges.count, bool)))
            b = gpu.with_rain(gpu.with_gauges(gpu.batch(s.build(step, "aorc")), view), s.rain.get("gefs", day))
            bb = model.without_broadcast(b)
            h = model.trunk(bb["steps"], bb["reach"], model.graph)
            inflow = (0.5 * torch.nn.functional.softplus(model.simulation(h).float()[..., 0])) * b["area"][:, None] / 21.6
            q = inflow * torch.exp(2 * torch.tanh(model.observed(h).float()[..., 0]))
            # (a) routing mass balance: route q followed by ROUTE empty steps; every unit of inflow must arrive.
            with torch.autocast("cuda", enabled=False):  # float32 throughout, as the router itself
                q = q.float()
                q_ext = torch.cat([q, torch.zeros(n, P.ROUTE, device=device)], dim=1)
                v_out = router(q_ext).sum(1).double()
                v_basin = (a_full @ q.sum(1, keepdim=True)).view(groups, n).sum(0).double()
            rel = ((v_out - v_basin).abs() / v_basin.clamp_min(1e-6)).cpu().numpy()
            # (b) basin continuity: a reach's share-weighted basin inflow = its parents' + its own lateral inflow.
            v_local = q.sum(1).double().cpu().numpy()
            vb = v_basin.cpu().numpy()
            # (c) the final forecast: gauge head at visible gauges against the routed flow and the reach below.
            result = model(b, full=True)
            lead = slice(P.INNOVATION, P.SCORED)
            final = result["reach"][:, lead].double().sum(1).cpu().numpy() * 6 * 3600  # m3 over 10 days
            routed = result["observed"][:, lead].double().sum(1).cpu().numpy() * 6 * 3600
            visible = np.flatnonzero(view["visible"])
            for g in visible:
                r = gauges.reach[g]
                d = downstream[r]
                if d < 0:
                    continue
                records.append(dict(issue=day, gauge=gauges.names[g], head_over_routed=final[r] / max(routed[r], 1e-6),
                                    final_up_over_down=final[r] / max(final[d], 1e-6), routed_up_over_down=routed[r] / max(routed[d], 1e-6)))
            if day == issues[0]:  # a parent enters with its share in this reach's basin (divergences split flow)
                for i in range(n):
                    if parents_of[i]:
                        lo, hi = network.indptr[i], network.indptr[i + 1]
                        share = dict(zip(network.members[lo:hi].tolist(), network.share[lo:hi].tolist()))
                        expected = sum(share.get(p, 0.0) * vb[p] for p in parents_of[i]) + v_local[i]
                        confluence.append(abs(vb[i] - expected) / max(expected, 1e-6))
            masses.append(dict(issue=day, median=float(np.median(rel)), p99=float(np.quantile(rel, 0.99)), max=float(rel.max())))
    mass, gauge = pd.DataFrame(masses), pd.DataFrame(records)
    confluence = np.array(confluence)
    gauge.to_csv(out / "gauge_head_continuity.csv", index=False)
    mass.to_csv(out / "routing_mass_balance.csv", index=False)
    tol = 1.05
    lines = [
        f"# Exp.3. Physical consistency of {a.model}", "", f"{len(issues)} test issues (GEFS rain, every visible gauge's history shown).", "",
        "## (a) Routing mass balance", "",
        "Lateral inflow q routed over the window plus 12 empty days: arriving volume against the share-weighted inflow of the basin.",
        f"Relative error per reach: median {mass['median'].median():.2e}, 99th percentile {mass.p99.max():.2e}, worst {mass['max'].max():.2e}.", "",
        "## (b) Basin continuity at confluences (first issue)", "",
        f"|basin inflow - (own inflow + each parent's basin inflow x its share here)| / that sum over {confluence.size:,} reaches with parents: "
        f"median {np.median(confluence):.2e}, worst {confluence.max():.2e}, share above 1% {np.mean(confluence > 0.01):.2%}.", "",
        "## (c) The gauge head and continuity", "",
        "At visible gauges the forecast is the gauge head, which corrects the routed flow there but is not passed downstream.",
        f"10-day volume of the gauge-head forecast over the routed flow at the same gauge: median {gauge.head_over_routed.median():.3f}, "
        f"5-95% {gauge.head_over_routed.quantile(0.05):.3f}-{gauge.head_over_routed.quantile(0.95):.3f}.",
        f"Upstream gauge volume above the reach just below by more than 5%: final forecast {np.mean(gauge.final_up_over_down > tol):.1%} of (gauge, issue) pairs, "
        f"routed flow alone {np.mean(gauge.routed_up_over_down > tol):.1%}. (Within a 10-day window the routed flow can also differ between",
        "adjacent reaches because each upstream source reaches each of them through its own river-distance kernel group.)",
        "",
        "Reading: (a) and (b) test the routing as implemented (kernel normalization and basin shares, end to end); (c) measures how",
        "far the gauge head departs from the routed, mass-conserving flow. Conservation claims should be limited to the routed flow.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- S2 0% ---------------------------------------------------------------------------------------------------
def zero_coverage(a):
    import torch

    state = torch.load(a.stage_a, map_location="cpu", weights_only=False)
    for k in range(1, splits.RANDOM_FOLDS + 1):
        folder = Path(a.out_template.format(k=k))
        folder.mkdir(parents=True, exist_ok=True)
        copy = dict(state, fold=k, split=dict(scheme="random", keep=0.0, keep_seed=0), stage="A as a fold model (no gauge seen)")
        torch.save(copy, folder / "best.pt")
        (folder / "DONE").touch()
        print(f"wrote {folder / 'best.pt'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("leakage")
    q.add_argument("--calibration", type=Path, help="CSV with a usgs_id column: NWM v3.0 calibration gauges")
    sub.add_parser("transfer")
    q = sub.add_parser("conservation")
    q.add_argument("--model", type=Path, default=P.MODELS / "gnn_B/best.pt")
    q.add_argument("--issues", type=int, default=100)
    q.add_argument("--every", type=int, default=19)
    q = sub.add_parser("zero-coverage")
    q.add_argument("--stage-a", type=Path, required=True)
    q.add_argument("--out-template", required=True, help="Folder per fold, with {k}")
    a = p.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    dict(leakage=leakage, transfer=transfer, conservation=conservation, **{"zero-coverage": zero_coverage})[a.command](a)


if __name__ == "__main__":
    main()
