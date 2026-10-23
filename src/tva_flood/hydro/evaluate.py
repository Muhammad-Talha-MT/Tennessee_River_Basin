"""The three test protocols in one pass over the test issues (torchrun, all GPUs).

Every model forecasts with what a forecaster has at a 00Z issue: observed weather and flow
before it, and after it GEFSv12 operational control rain to 240 h (other weather unknown).
Issues: days from 2020-09-23 with GEFS rain and an NWM medium-range forecast (the
reference). ``+aorc`` rows put the realized rain in place of the forecast: a perfect-rain
diagnostic, not a forecast.

1. Gauged: the all-gauge model with every gauge's recent flow shown, at the CAMELS-H gauges
   against USGS. The tuned LSTM, trained under the same rain rule, comes from
   ``evaluate-benchmark --gefs`` (``models/gefs_benchmark``) on exactly the same values.
   References: NWM medium-range streamflow, persistence, climatology.
2. Ungauged: each fold model, with that fold's gauges hidden, at that fold's gauges
   (``models/gefs_benchmark_f*`` for the LSTM).
3. Every reach (every second issue up to 2023-01-20, the retrospective's end): against the
   NWM v3.0 retrospective. A reach in a gauge's basin (smallest catalog basin containing
   it) takes the model of that gauge's fold, so held-out basins are forecast without their
   gauge; other reaches take the all-gauge model. The NWM forecast at every reach is the
   reference (``models/retrospective_comparison_gefs_nwm``), the LSTM run at every reach
   (``scripts/lstm_every_reach.py``) another.

Results are also split by the TVA dams (``hydro-dams``; the model never reads them): gauges
in headwater basins without a dam, below another gauge without a dam, and with a dam in the
basin; reaches with more than half their basin behind a dam, and free-flowing large rivers.

Outputs in --out: report.md, gauged.csv, ungauged.csv, reaches.csv (medians by scope and
drainage area), reach_metrics.npz, forecasts_gauged.npz and forecasts_ungauged.npz.
"""

import argparse
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from tva_flood.common import atomic_json
from tva_flood.data.folds import holdout_folds
from tva_flood.data.tva import basin_owner
from tva_flood.evaluation.benchmark import climatology, load_nwm
from tva_flood.evaluation.hydrology import Accumulator, reach_metrics

from . import paths as P
from .data import timeline_index
from .legacy import HydroModelV1
from .model import HydroModel
from .samples import Device, Samples

FUTURE = slice(P.INNOVATION, P.SCORED)
LEADS = (6, 24, 72, 120, 240)
DAYS_1_5 = (24, 48, 72, 96, 120)
FOLDS = 5
BASELINES = (("LSTM, tuned", "lstm"),)
AREA_CLASSES = (("< 50 km2", 0, 50), ("50-500", 50, 500), ("500-5,000", 500, 5000), ("> 5,000", 5000, np.inf))
METRICS = ("nse", "kge", "r", "alpha", "beta", "rmse", "peak_error", "peak_error_at_target_time")  # saved per reach


def load_model(path, s, device, legacy=False):
    state = torch.load(path, map_location="cpu", weights_only=False)
    c = state["config"]
    model = (HydroModelV1 if legacy else HydroModel)(c["groups"], c["delay"], c["hidden"], c["channels"], c["graph_layers"], **{k: c[k] for k in ("quantiles", "head") if k in c})
    model.load_state_dict(state["model"])
    model.to(device).eval()
    model.attach(s.network, s.network.edges, s.gauges.reach, device)
    return model, state


class NWMReaches:
    """NWM medium-range six-hour flow [lead, reach] by issue day, one month at a time."""

    def __init__(self):
        self.paths = {p.stem: p for p in P.NWM_REACHES.glob("????-??.npz")}
        self.month, self.flows = None, {}

    def days(self):
        out = set()
        for path in self.paths.values():
            with np.load(path) as d:
                out.update(pd.Timestamp(str(x)) for x in d["issue"])
        return out

    def get(self, day):
        key = f"{day:%Y-%m}"
        if key != self.month:
            with np.load(self.paths[key]) as d:
                self.flows = {pd.Timestamp(str(x)): d["flow"][i] for i, x in enumerate(d["issue"])}
            self.month = key
        return self.flows[day]


def test_issues(s, nwm, reach_days, limit=0):
    """Test issues (GEFS days with an NWM forecast) and the every-reach subset: every second
    issue up to the retrospective's end with an NWM forecast at every reach (as
    ``scripts/retrospective_comparison.py --rain gefs --every 2`` selects them)."""
    times = s.arrays.times
    last = s.arrays.steps - P.HORIZON
    gefs_days = s.rain.days("gefs")
    issues = [
        timeline_index(day)
        for day in sorted(nwm)
        if day >= P.TEST_FROM and day in gefs_days and timeline_index(day) <= last
    ]
    every_reach = set(
        [i for i in issues if times[i] <= P.RETRO_UNTIL + pd.Timedelta(days=19) and times[i] in reach_days][::2]
    )
    if limit:
        issues = issues[:limit]
        every_reach &= set(issues)
    return issues, every_reach


def reach_folds(s):
    """Fold of the gauge whose basin holds each reach (-1: none) and of each gauge."""
    gauge_fold = holdout_folds(P.ROOT, s.gauges.catalog, FOLDS)
    owner, _ = basin_owner(s.network.edges, s.network.n, s.gauges.reach)
    return np.where(owner >= 0, gauge_fold[np.maximum(owner, 0)], -1), gauge_fold


def regulation(network, gauges):
    """Share of each reach's basin whose water passes a TVA dam first, and the gauge groups:
    headwater without a dam, below another gauge without a dam, a dam in the basin."""
    with np.load(P.HYDRO / "dams.npz") as d:
        regulated, dam_reach = d["regulated"], d["reach"]
    gauge_at, dam_at = np.zeros(network.n, bool), np.zeros(network.n, bool)
    gauge_at[gauges.reach], dam_at[dam_reach] = True, True
    dammed = np.array([dam_at[b].any() for b in gauges.basins]) | (regulated[gauges.reach] > 0)
    nested = np.array([gauge_at[b].sum() > 1 for b in gauges.basins])
    groups = (("headwater, no dam", ~nested & ~dammed), ("below a gauge, no dam", nested & ~dammed), ("dam in the basin", dammed))
    return regulated, groups


def reach_table(series, reach_fold, network, regulated):
    """Median NSE by lead and scope (all reaches, held-out basins, by size, behind dams,
    free-flowing large rivers) for each series [lead, reach]: a DataFrame, its markdown lines."""
    area = pd.read_parquet(P.PROCESSED / "graph_candidate/nodes.parquet", columns=["comid", "totdasqkm"]).set_index("comid").totdasqkm.reindex(network.ids).to_numpy()
    held_basin = reach_fold > 0
    scopes = [("all reaches", np.ones(network.n, bool)), ("held-out basins", held_basin), ("outside every gauge basin", reach_fold == -1)]
    scopes += [(f"held-out basins, {c}", held_basin & (area >= lo) & (area < hi)) for c, lo, hi in AREA_CLASSES]
    scopes += [("behind dams (> 50% of basin)", regulated > 0.5), ("free-flowing, > 500 km2", (regulated <= 0) & (area >= 500))]
    table = []
    for lead in (24, 72, 120):
        for scope, m in scopes:
            entry = dict(lead_h=lead, scope=scope, reaches=int(m.sum()))
            for label, nse in series.items():
                entry[label] = round(float(medians(nse[lead // 6 - 1][m][None])[0]), 3)
            table.append(entry)
    table = pd.DataFrame(table)
    lines = ["| lead | scope | reaches | " + " | ".join(series) + " |", "|---|---|---:|" + "---:|" * len(series)]
    for r in table.itertuples(index=False):
        lines.append(f"| {r[0]} h | {r[1]} | {r[2]:,} | " + " | ".join(f"{x:.3f}" for x in r[3:]) + " |")
    return table, lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=P.MODELS / "gefs_hydro_B/best.pt")
    p.add_argument("--fold-models", default=str(P.MODELS / "gefs_hydro_B_f{k}/best.pt"))
    p.add_argument("--legacy", action="store_true", help="The checkpoints are V1 (models/hydro_v1): residual head")
    p.add_argument("--rain-steps", type=int, default=P.HORIZON, help="Forecast rain known to this lead (V1: 16)")
    p.add_argument("--label", default="new model", help="Name of the scored model in the report")
    p.add_argument("--min-count", type=int, default=100)
    p.add_argument("--limit", type=int, default=0, help="First n issues only (testing)")
    a = p.parse_args()
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=120), device_id=device)
    try:
        evaluate(a, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def evaluate(a, rank, world, device):
    s = Samples()
    arrays, gauges, network = s.arrays, s.gauges, s.network
    out = a.out.resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    gpu = Device(network, gauges, arrays, device)
    models = {0: load_model(a.model, s, device, a.legacy)[0]}
    for k in range(1, FOLDS + 1):
        model, state = load_model(Path(a.fold_models.format(k=k)), s, device, a.legacy)
        if state["fold"] != k:
            raise ValueError(f"Fold model {k} was trained with fold {state['fold']} withheld")
        models[k] = model
    reach_fold, gauge_fold = reach_folds(s)
    withheld = {k: gauge_fold == k if k else np.zeros(gauges.count, bool) for k in models}
    reach_model = torch.as_tensor(np.maximum(reach_fold, 0), device=device)
    sites = torch.as_tensor(gauges.camelsh, device=device)

    nwm, nwm_names = load_nwm(P.NWM_GAUGES)
    if list(nwm_names) != list(gauges.names):
        raise ValueError("NWM gauge forecasts belong to another gauge list")
    reach_nwm = NWMReaches()
    clim = climatology(gauges.catalog, P.TRAIN_UNTIL)
    times = arrays.times
    issues, every_reach = test_issues(s, nwm, reach_nwm.days(), a.limit)
    mine = issues[rank::world]

    g = gauges.count
    gauged_names = ["hydro+gefs", "hydro+aorc", "nwm", "persistence", "climatology"]
    ungauged_names = ["hydro+gefs", "hydro+aorc", "nwm", "persistence", "climatology"]
    acc = dict(
        gauged={m: Accumulator(g, device) for m in gauged_names},
        ungauged={m: Accumulator(g, device) for m in ungauged_names},
        reach={m: Accumulator(network.n, device) for m in ["hydro+gefs", "hydro+aorc", "retro_persistence"]},
    )
    kept = dict(gauged=[], ungauged=[])
    began = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for position, step in enumerate(mine):
            day = times[step].normalize()
            nwm_flow, _ = nwm[day]
            truth_np = gauges.observed[step : step + P.HORIZON]
            truth = torch.as_tensor(truth_np, device=device).double()
            latest = torch.as_tensor(gauges.observed[step - 1], device=device).double()
            nwm_gauge = torch.as_tensor(nwm_flow, device=device).double()
            mask = torch.isfinite(truth) & torch.isfinite(latest)[None] & torch.isfinite(nwm_gauge) & sites[None]
            truth0 = torch.nan_to_num(truth)
            ahead = times[step + np.arange(P.HORIZON)]
            refs = dict(
                nwm=torch.nan_to_num(nwm_gauge),
                persistence=torch.nan_to_num(latest)[None].expand(P.HORIZON, -1),
                climatology=torch.as_tensor(np.nan_to_num(clim[ahead.dayofyear, ahead.hour // 6]), device=device),
            )
            base = gpu.limit_rain(gpu.batch(s.build(step, "aorc")), a.rain_steps)
            rains = {"gefs": s.rain.get("gefs", day), "aorc": "aorc"}
            full = step in every_reach
            if full:
                retro = torch.as_tensor(arrays.flow_raw(base["nwm"][FUTURE].cpu().numpy()), device=device)
                retro_ok = base["nwm_valid"][FUTURE].bool()
                latest_ok = base["nwm_valid"][P.INNOVATION - 1].bool()
                nwm_reach = torch.as_tensor(reach_nwm.get(day), device=device).double()
                reach_mask = retro_ok & latest_ok[None] & torch.isfinite(nwm_reach)
                composite = {m: torch.zeros(P.HORIZON, network.n, dtype=torch.float64, device=device) for m in rains}
            values = {}
            for k, model in models.items():
                view = gpu.with_gauges(base, s.gauge_view(step, s.eligible(step, withheld[k]), withheld[k]))
                for tag, rain in rains.items():
                    batch = view if isinstance(rain, str) else gpu.with_rain(view, rain)
                    result = model(batch, full=full)
                    flow = result["gauge"][:, FUTURE].T.double()
                    if k == 0:
                        values[f"hydro+{tag}"] = flow
                    else:
                        held = torch.as_tensor(withheld[k], device=device)
                        prior = values.get(f"ungauged+{tag}", torch.zeros_like(flow))
                        values[f"ungauged+{tag}"] = torch.where(held[None], flow, prior)
                    if full:
                        reach_flow = result["reach"][:, FUTURE].T.double()
                        mine_reaches = (reach_model == k)[None]
                        composite[tag] = torch.where(mine_reaches, reach_flow, composite[tag])
            for name in gauged_names:
                value = values[name] if name.startswith("hydro") else refs[name]
                acc["gauged"][name].add(value, truth0, mask)
            fold_mask = mask & torch.as_tensor(gauge_fold > 0, device=device)[None]
            for name in ungauged_names:
                value = values[name.replace("hydro", "ungauged")] if name.startswith("hydro") else refs[name]
                acc["ungauged"][name].add(value, truth0, fold_mask)
            kept["gauged"].append(_keep(step, truth, mask, [values.get(n, refs.get(n)) for n in gauged_names]))
            kept["ungauged"].append(
                _keep(step, truth, fold_mask, [values["ungauged+gefs"], values["ungauged+aorc"]])
            )
            if full:
                retro0 = torch.nan_to_num(retro).double()
                acc["reach"]["hydro+gefs"].add(composite["gefs"], retro0, reach_mask)
                acc["reach"]["hydro+aorc"].add(composite["aorc"], retro0, reach_mask)
                last_retro = torch.as_tensor(arrays.flow_raw(base["nwm"][P.INNOVATION - 1].cpu().numpy()), device=device)
                acc["reach"]["retro_persistence"].add(last_retro[None].expand(P.HORIZON, -1).double(), retro0, reach_mask)
            if rank == 0 and position % 25 == 0:
                print(f"issue {position + 1}/{len(mine)} per process; {(time.perf_counter() - began) / 60:.1f} min", flush=True)
    for group in acc.values():
        for accumulator in group.values():
            accumulator.reduce(world)
    for name, rows in kept.items():
        np.savez(out / f".{name}_{rank}.npz", **_stack(rows))
    if world > 1:
        dist.barrier()
    if rank == 0:
        write(a, out, s, acc, issues, every_reach, reach_fold, world)


def _keep(step, truth, mask, values):
    nan = torch.tensor(float("nan"), device=truth.device, dtype=torch.float64)
    return dict(
        step=step,
        truth=torch.where(mask, truth, nan).float().cpu().numpy(),
        values=np.stack([torch.where(mask, v.double(), nan).float().cpu().numpy() for v in values]),
    )


def _stack(rows):
    if not rows:
        return dict(step=np.zeros(0, np.int64))
    return dict(
        step=np.array([r["step"] for r in rows]),
        truth=np.stack([r["truth"] for r in rows]),
        values=np.stack([r["values"] for r in rows]),
    )


def gauge_table(accumulators, min_count):
    """Median NSE over scored gauges by lead, per method."""
    rows = {}
    for name, acc in accumulators.items():
        m = reach_metrics(**acc.numpy(), min_count=min_count)
        rows[name] = (m["nse"], m["scored"])
    return rows


def medians(nse, scored=None):
    v = np.where(scored, nse, np.nan) if scored is not None else nse
    with np.errstate(all="ignore"):
        return np.nanmedian(v, axis=-1)


def days_1_5(nse, scored, gauges):
    """Mean over leads 24-120 h of the median NSE [lead, gauge] over the selected gauges."""
    return float(np.mean(medians(nse[:, gauges], None if scored is None else scored[:, gauges])[[h // 6 - 1 for h in DAYS_1_5]]))


def _by_station(saved, names):
    """Saved NSE (``saved_nse`` value: nse [lead, station], stations) on the catalog order."""
    nse, stations = saved
    full = np.full((nse.shape[0], len(names)), np.nan)
    full[:, pd.Index(names).get_indexer(stations)] = nse
    return full


def saved_nse(path, methods, steps, times, min_count):
    """Per-gauge NSE [lead, station] of saved benchmark forecasts on the given issues."""
    d = np.load(path, allow_pickle=True)
    keep = pd.DatetimeIndex(d["time"]).isin(times[np.array(sorted(steps), dtype=np.int64)])
    names = list(d["methods"].astype(str))
    out = {}
    for method in methods:
        if method not in names:
            continue
        value = d["values"][keep, names.index(method)].astype(np.float64)
        truth = d["truth"][keep].astype(np.float64)
        ok = np.isfinite(value) & np.isfinite(truth)
        n = ok.sum(0)
        o = np.where(ok, truth, 0)
        f = np.where(ok, value, 0)
        mean_o = o.sum(0) / np.maximum(n, 1)
        var_o = (o * o).sum(0) / np.maximum(n, 1) - mean_o**2
        mse = ((f - o) ** 2).sum(0) / np.maximum(n, 1)
        scored = (n >= min_count) & (mean_o > 0) & (np.sqrt(np.maximum(var_o, 0)) >= 1e-3)
        out[method] = (np.where(scored, 1 - mse / var_o, np.nan), d["stations"].astype(str))
    return out


def row(label, nse_by_lead):
    cells = [f"{nse_by_lead[h // 6 - 1]:.3f}" for h in LEADS]
    days = np.mean([nse_by_lead[h // 6 - 1] for h in DAYS_1_5])
    return f"| {label} | " + " | ".join(cells) + f" | {days:.3f} |"


def header(first):
    return [f"| {first} | " + " | ".join(f"{h} h" for h in LEADS) + " | days 1-5 |", "|---|" + "---:|" * (len(LEADS) + 1)]


def write(a, out, s, acc, issues, every_reach, reach_fold, world):
    gauges, network, times = s.gauges, s.network, s.arrays.times
    for name in ("gauged", "ungauged"):
        parts = [np.load(out / f".{name}_{r}.npz") for r in range(world)]
        parts = [q for q in parts if len(q["step"])]
        step = np.concatenate([q["step"] for q in parts])
        order = np.argsort(step)
        np.savez_compressed(
            out / f"forecasts_{name}.npz",
            step=step[order],
            time=times[step[order]].to_numpy(),
            truth=np.concatenate([q["truth"] for q in parts])[order],
            values=np.concatenate([q["values"] for q in parts])[order],
            stations=gauges.names,
        )
        for r in range(world):
            (out / f".{name}_{r}.npz").unlink()
    first, final = times[issues[0]], times[issues[-1]]
    lines = [
        f"# Distributed model: three test protocols ({len(issues)} 00Z issues, {first:%Y-%m-%d} to {final:%Y-%m-%d})",
        "",
        f"Every forecast uses GEFSv12 operational rain to {a.rain_steps * 6} h; +aorc is the realized-rain diagnostic.",
        "",
    ]
    # 1. Gauged --------------------------------------------------------------------------
    mine = gauge_table(acc["gauged"], a.min_count)
    old = saved_nse(
        P.MODELS / "gefs_benchmark/forecasts.npz",
        [f"lstm+{r}" for r in ("gefs", "aorc")],
        issues,
        times,
        a.min_count,
    )
    names = list(gauges.names)
    rows = header("method (gauged, median NSE over gauges)")
    for rain in ("gefs", "aorc"):
        nse, scored = mine[f"hydro+{rain}"]
        rows.append(row(f"**{a.label}**+{rain}", medians(nse, scored)))
        for label, prefix in BASELINES:
            if f"{prefix}+{rain}" in old:
                rows.append(row(f"{label}+{rain}", medians(old[f"{prefix}+{rain}"][0])))
    for name in ("nwm", "persistence", "climatology"):
        nse, scored = mine[name]
        rows.append(row(name, medians(nse, scored)))
    lines += ["## 1. Gauged: CAMELS-H gauges against USGS", ""] + rows
    per_station = []
    for name, (nse, scored) in mine.items():
        for lead in LEADS:
            for gi, st in enumerate(names):
                per_station.append(dict(method=name, lead_h=lead, station=st, nse=nse[lead // 6 - 1, gi] if scored[lead // 6 - 1, gi] else np.nan))
    pd.DataFrame(per_station).to_csv(out / "gauged.csv", index=False)
    # 2. Ungauged ------------------------------------------------------------------------
    mine_u = gauge_table(acc["ungauged"], a.min_count)
    rows = header("method (ungauged, held-out gauges)")
    for name in ("hydro+gefs", "hydro+aorc"):
        nse, scored = mine_u[name]
        rows.append(row(f"**{a.label}**{name[5:]}", medians(nse, scored)))
    old_u = {}
    for k in range(1, FOLDS + 1):
        part = saved_nse(
            P.MODELS / f"gefs_benchmark_f{k}/forecasts.npz",
            [f"lstm+{r}" for r in ("gefs", "aorc")],
            issues,
            times,
            a.min_count,
        )
        for method, (nse, stations) in part.items():
            full = old_u.setdefault(method, np.full((P.HORIZON, gauges.count), np.nan))
            index = pd.Index(names).get_indexer(stations)
            full[:, index] = np.where(np.isfinite(nse), nse, full[:, index])
    held = holdout_folds(P.ROOT, gauges.catalog, FOLDS) > 0
    for label, prefix in BASELINES:
        for rain in ("gefs", "aorc"):
            if f"{prefix}+{rain}" in old_u:
                rows.append(row(f"{label}, ungauged+{rain}", medians(old_u[f"{prefix}+{rain}"][:, held])))
    for name in ("nwm", "persistence", "climatology"):
        nse, scored = mine_u[name]
        rows.append(row(f"{name} (reference: may use these gauges)", medians(nse, scored)))
    count = int(np.isfinite(np.where(mine_u["hydro+gefs"][1], mine_u["hydro+gefs"][0], np.nan)[3]).sum())
    lines += ["", f"## 2. Ungauged: {count} held-out gauges (5 folds) against USGS", ""] + rows
    # By dams ----------------------------------------------------------------------------
    regulated, groups = regulation(network, gauges)
    lines += ["", "By dams (days 1-5, GEFS rain):", "", f"| protocol | gauges | {a.label} | LSTM | NWM |", "|---|---|---:|---:|---:|"]
    for protocol, table_, lstm_nse, keep in (
        ("gauged", mine, old.get("lstm+gefs"), np.ones(gauges.count, bool)),
        ("ungauged", mine_u, old_u.get("lstm+gefs"), held),
    ):
        if protocol == "gauged" and lstm_nse is not None:
            lstm_nse = _by_station(lstm_nse, names)
        for scope, m in groups:
            m = m & keep
            nse, scored = table_["hydro+gefs"]
            cells = [days_1_5(nse, scored, m), days_1_5(*table_["nwm"], m)]
            lstm_cell = "-" if lstm_nse is None else f"{days_1_5(lstm_nse, None, m):.3f}"
            count = int(scored[DAYS_1_5[0] // 6 - 1][m].sum())  # gauges scored at 24 h
            lines.append(f"| {protocol} | {scope} ({count}) | {cells[0]:.3f} | {lstm_cell} | {cells[1]:.3f} |")
    pd.DataFrame(
        [dict(method=n, lead_h=h, station=st, nse=nse[h // 6 - 1, i]) for n, (nse, _) in mine_u.items() for h in LEADS for i, st in enumerate(names)]
    ).to_csv(out / "ungauged.csv", index=False)
    # 3. Every reach ---------------------------------------------------------------------
    metrics = {n: reach_metrics(**acc_.numpy(), min_count=a.min_count) for n, acc_ in acc["reach"].items()}
    nwm_ref = np.load(P.MODELS / "retrospective_comparison_gefs_nwm/metrics.npz")
    if not a.limit and int(nwm_ref["issues"]) != len(every_reach):
        raise ValueError(f"NWM every-reach reference: {int(nwm_ref['issues'])} issues, not {len(every_reach)}")
    series = {
        f"{a.label}+gefs": metrics["hydro+gefs"]["nse"],
        f"{a.label}+aorc": metrics["hydro+aorc"]["nse"],
    }
    lstm = P.MODELS / "gefs_lstm_reaches/metrics.npz"  # lstm_every_reach.py
    if lstm.exists():
        saved = np.load(lstm)
        if not a.limit and int(saved["issues"]) != len(every_reach):
            raise ValueError(f"LSTM every-reach run: {int(saved['issues'])} issues, not {len(every_reach)}")
        series.update({"LSTM, tuned+gefs": saved["nse/gefs"], "LSTM, tuned+aorc": saved["nse/aorc"]})
    series.update({
        "NWM forecast": nwm_ref["reach/nwm_forecast/nse"],
        "retrospective persistence (oracle)": metrics["retro_persistence"]["nse"],
    })
    table, rows = reach_table(series, reach_fold, network, regulated)
    table.to_csv(out / "reaches.csv", index=False)
    lines += ["", f"## 3. Every reach against the NWM v3.0 retrospective ({len(every_reach)} issues to 2023-01-20)", ""] + rows
    full = {f"{a.label}+{t}/{k}": metrics[f"hydro+{t}"][k] for t in ("gefs", "aorc") for k in METRICS}
    if lstm.exists():
        full.update({f"LSTM, tuned+{t}/{k}": saved[f"{k}/{t}"] for t in ("gefs", "aorc") for k in METRICS if f"{k}/{t}" in saved})
    full.update({f"NWM forecast/{k}": nwm_ref[f"reach/nwm_forecast/{k}"] for k in METRICS if f"reach/nwm_forecast/{k}" in nwm_ref})
    full.update({f"{k}/nse": v for k, v in series.items()})
    np.savez_compressed(out / "reach_metrics.npz", comid=network.ids, reach_fold=reach_fold, **full)
    lines += ["", "Median NSE per lead (six-hour means). Days 1-5: mean of the medians at 24-120 h."]
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report)
    atomic_json(out / "manifest.json", dict(model=str(a.model), fold_models=a.fold_models, issues=len(issues), every_reach_issues=len(every_reach), min_count=a.min_count))
    print(report, flush=True)
    print(f"EVALUATION COMPLETE: {out}", flush=True)


if __name__ == "__main__":
    main()
