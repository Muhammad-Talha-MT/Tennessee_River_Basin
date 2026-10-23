"""Public benchmark: graph models, the LSTM baseline and NOAA NWM against USGS, 2020-2025.

No TVA data is used. Every method is scored on the same issue times (00Z, when an
NWM medium-range forecast exists), the same gauges (CAMELS-H sites of the catalog
by default) and the same six-hour USGS observations (``download-usgs``).

Methods (each graph or LSTM checkpoint gets two or three rain settings):

- ``NAME+nwm``: future rain = the rain NWM's medium-range forecast was forced with
  (``download-nwm-forecasts``), up to the checkpoint's rain window: the
  operational comparison with NWM;
- ``NAME+gefs`` (with ``--gefs``): GEFSv12 operational control rain, the system the
  graph model's reforecast training rain comes from;
- ``NAME+aorc``: realized AORC rain to the same window (a perfect rain forecast,
  an upper bound);
- ``nwm``: NWM medium-range streamflow (deterministic member);
- ``persistence``: the latest observed six-hour flow; ``climatology``: the
  1979-2018 mean flow of that gauge, day of year and time of day.

History at issue time: AORC weather and observed USGS flow at every catalog gauge
(the graph models), or their basin means (the LSTM). Leads follow the archive
convention: model step k is the six-hour mean over [t+6k, t+6k+6).

Ungauged test (``--holdout-fold k --folds n``, ``data.folds``): only that fold's
gauges are scored and their flow history is hidden from every model, which must be
trained with the same fold withheld. Persistence and climatology still use those
gauges' own records, and NWM's analysis may assimilate them: references only.

Outputs: ``per_station.csv``, ``summary_by_lead.csv``, ``report.md`` and
``forecasts.npz`` (for ``evaluate-events --reference nwm``).
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
import xarray as xr

from tva_flood.common import atomic_json, sha
from tva_flood.data.basin_series import basin_forecast, basin_rows, covered_reaches
from tva_flood.data.folds import held_out
from tva_flood.data.gauges import Catalog
from tva_flood.data.inputs import ForecastInputs
from tva_flood.data.river import River
from tva_flood.data.schema import HORIZON
from tva_flood.data.ssl import EncoderWindows
from tva_flood.downloads.nwm_forecasts import six_hour_flow
from tva_flood.models.lstm import ForecastLSTM
from tva_flood.models.sparse import sparse_supports
from tva_flood.training.lstm_baseline import Series

from .hydrology import Accumulator, load_forecast, raw_flow, reach_metrics, read_forecast

REPORT_LEADS = (6, 24, 72, 120, 240)


def pairs(values, p, flag):
    try:
        return dict(item.split("=", 1) for item in values or [])
    except ValueError:
        p.error(f"{flag} must be NAME=PATH")


def load_nwm(folder):
    """{issue date: (six-hour flow [lead, gauge], rain [step, reach])} from monthly files."""
    result, names = {}, None
    for path in sorted(Path(folder).glob("*.npz")):
        if path.name.endswith(".part.npz"):
            continue
        with np.load(path) as data:
            names = data["gauge"].astype(str)
            flow = six_hour_flow(data["flow"])
            for d, day in enumerate(data["issue"]):
                result[pd.Timestamp(str(day))] = (flow[d], data["rain"][d].astype(np.float32))
    return result, names


def load_rain(folder):
    """{issue date: six-hour rain [step, reach] (mm)} from monthly rain-forecast files."""
    result = {}
    for path in sorted(Path(folder).glob("*.npz")):
        if path.name.endswith(".part.npz"):
            continue
        with np.load(path) as data:
            for d, day in enumerate(data["issue"]):
                result[pd.Timestamp(str(day))] = data["rain"][d].astype(np.float32)
    return result


def climatology(catalog, end):
    """Mean flow [day of year 1-366, six-hour slot 0-3, gauge] over training years."""
    times = catalog.times
    keep = times < end
    frame = pd.DataFrame(catalog.observed[keep].astype(np.float64))
    frame["doy"], frame["slot"] = times[keep].dayofyear, times[keep].hour // 6
    mean = frame.groupby(["doy", "slot"]).mean()
    table = np.full((367, 4, len(catalog)), np.nan)
    for (doy, slot), row in mean.iterrows():
        table[doy, slot] = row.to_numpy()
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--preparation", type=Path, required=True, help="Test preparation")
    p.add_argument("--array-cache", type=Path)
    p.add_argument(
        "--gauges", type=Path, help="Default: <root>/processed/benchmarks/benchmark_gauges.nc"
    )
    p.add_argument("--nwm", type=Path, help="Default: <root>/processed/benchmarks/nwm_medium_range")
    p.add_argument(
        "--gefs",
        type=Path,
        help="Also score with GEFSv12 operational rain (download-gefs --source operational); "
        "restricts issues to days with GEFS (2020-09-23 on)",
    )
    p.add_argument("--checkpoint", action="append", metavar="NAME=PATH", help="Graph model")
    p.add_argument("--lstm", action="append", metavar="NAME=PATH", help="LSTM baseline")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sites", choices=["camelsh", "all"], default="camelsh")
    p.add_argument("--holdout-fold", type=int, default=0, help="Score this withheld fold only")
    p.add_argument("--folds", type=int, default=0)
    p.add_argument("--fold-seed", type=int, default=0)
    p.add_argument("--issues-from")
    p.add_argument("--issues-until")
    p.add_argument("--train-until", default="2019-01-01", help="Climatology period end")
    p.add_argument("--min-count", type=int, default=100)
    p.add_argument("--cache-months", type=int, default=4)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = p.parse_args()
    a.models, a.lstms = pairs(a.checkpoint, p, "--checkpoint"), pairs(a.lstm, p, "--lstm")
    rank, world, local = [
        int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0))
    ]
    device = torch.device(f"cuda:{local}" if a.device == "cuda" else "cpu")
    if a.device == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        kwargs = dict(
            backend="nccl" if a.device == "cuda" else "gloo", timeout=timedelta(minutes=90)
        )
        if a.device == "cuda":
            kwargs["device_id"] = device
        dist.init_process_group(**kwargs)
    try:
        evaluate(a, p, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def evaluate(a, p, rank, world, device):
    root = a.root.resolve()
    prep = a.preparation.resolve()
    out = a.out.resolve()
    if rank == 0:
        if out.exists():
            p.error(f"{out} exists; choose a new --out")
        out.mkdir(parents=True)
    gauges_nc = (a.gauges or root / "processed/benchmarks/benchmark_gauges.nc").resolve()
    catalog = Catalog(gauges_nc)
    with xr.open_dataset(gauges_nc) as info:
        usgs = info.usgs_id.values.astype(str)
        camelsh = info.camelsh.values.astype(bool)
    sites = camelsh if a.sites == "camelsh" else np.ones(len(catalog), bool)
    held = held_out(root, catalog, a.holdout_fold, a.folds, a.fold_seed)
    if a.holdout_fold:
        sites = sites & held
    scored = np.flatnonzero(sites)
    data = EncoderWindows(
        root,
        prep,
        "test",
        HORIZON,
        a.cache_months,
        array_cache=a.array_cache.resolve() if a.array_cache else None,
        future_rain=True,
    )
    with np.load(prep / "graph.npz") as graph:
        ids = graph["feature_id"].copy()
        edges = np.column_stack([graph["source"], graph["target"]])
    river = River.load(root, ids, edges)
    saved = {n: read_forecast(Path(path)) for n, path in a.models.items()}
    for n, s in saved.items():
        trained = s["config"].get("observed_finetune", {}).get("arguments", {})
        if (trained.get("holdout_fold") or 0, trained.get("folds") or 0) != (
            a.holdout_fold,
            a.folds if a.holdout_fold else 0,
        ):
            p.error(f"{n} was not trained with fold {a.holdout_fold} of {a.folds} withheld")
    directions = {s["config"]["architecture"].get("direction", "both") for s in saved.values()}
    if len(directions) > 1:
        p.error("Score whole-basin (upstream) and other graph checkpoints in separate runs")
    supports = sparse_supports(len(ids), edges)
    halo = max(
        [
            s["config"]["architecture"]["graph_layers"]
            + s["config"]["decoder"]["decoder_graph_layers"]
            for s in saved.values()
        ]
        or [0]
    )
    inputs = ForecastInputs(
        data, catalog, None, edges, supports, halo, river if directions == {"upstream"} else None
    )
    graph = inputs.graph(scored)
    graph_supports = [s.to(device) for s in graph["supports"]]
    static = graph["static"].to(device)
    models = {n: load_forecast(s, data, prep, supports, device) for n, s in saved.items()}
    steps = {n: s["config"].get("rain_hours", 96) // 6 for n, s in saved.items()}
    mean, std = inputs.mean, inputs.std
    # LSTM baselines read basin means; NWM rain becomes basin means of reach z values.
    lstms, series = {}, None
    if a.lstms:
        series = Series(prep, catalog, mean, std)
        rows = basin_rows(river, ids, catalog.comid[scored])
        covered = covered_reaches(root, ids)
        for name, path in a.lstms.items():
            state = torch.load(path, map_location="cpu", weights_only=False)
            c = state["config"]
            trained = c.get("arguments", {})
            if (trained.get("holdout_fold") or 0, trained.get("folds") or 0) != (
                a.holdout_fold,
                a.folds if a.holdout_fold else 0,
            ):
                p.error(f"{name} was not trained with fold {a.holdout_fold} of {a.folds} withheld")
            if list(c["gauges"]) != list(catalog.names):
                p.error(f"{name} was trained on different gauges")
            model = ForecastLSTM(hidden=c["hidden"], dropout=c["dropout"], horizon=HORIZON)
            model.load_state_dict(state["model"])
            lstms[name] = (
                model.to(device).eval(),
                c["rain_hours"] // 6,
                c["flow_mean"],
                c["flow_std"],
            )
    nwm, nwm_names = load_nwm(a.nwm or root / "processed/benchmarks/nwm_medium_range")
    if nwm_names is None:
        p.error("No NWM forecasts found; run download-nwm-forecasts first")
    if list(nwm_names) != list(catalog.names):
        p.error("NWM forecasts were downloaded for different gauges")
    clim = climatology(catalog, pd.Timestamp(a.train_until))
    gefs = load_rain(a.gefs) if a.gefs else None
    sources = ("nwm", "gefs", "aorc") if gefs else ("nwm", "aorc")
    names = [f"{n}+{r}" for n in list(models) + list(lstms) for r in sources]
    names += ["nwm", "persistence", "climatology"]
    first, last = (
        pd.Timestamp(a.issues_from or "1900-01-01"),
        pd.Timestamp(a.issues_until or "2200-01-01"),
    )
    times = data.times
    items = [
        i
        for i, issue in enumerate(data.issues)
        if times[issue].hour == 0
        and first <= times[issue] < last
        and times[issue].normalize() in nwm
        and (gefs is None or times[issue].normalize() in gefs)
    ]
    if not items:
        p.error("No issue time has an NWM forecast in the requested period")
    totals = {n: Accumulator(len(scored), device) for n in names}
    truth_all = inputs.observed[:, scored]
    kept_issue, kept_truth, kept_values = [], [], []
    began = time.perf_counter()
    with torch.inference_mode():
        for position, item in enumerate(items[rank::world]):
            sample = inputs.load(data, item, graph, held if a.holdout_fold else None)
            issue = sample["issue"]
            day = times[issue].normalize()
            nwm_flow, nwm_rain = nwm[day]
            truth = torch.as_tensor(truth_all[issue : issue + HORIZON], device=device)
            latest = torch.as_tensor(truth_all[issue - 1], device=device)
            nwm_flow = torch.as_tensor(nwm_flow[:, scored], dtype=torch.float64, device=device)
            mask = torch.isfinite(truth) & torch.isfinite(latest)[None] & torch.isfinite(nwm_flow)
            if not mask.any():
                continue
            truth = torch.nan_to_num(truth)
            values = {}
            dynamic = sample["dynamic"][None].to(device)
            forecasts = {"nwm": nwm_rain}
            if gefs is not None:
                forecasts["gefs"] = gefs[day]
            for name, model in models.items():
                rain_aorc = sample["future"].clone()
                rain_aorc[steps[name] :] = 0
                futures = {
                    k: inputs.rain_field(sample, graph, v, steps[name])
                    for k, v in forecasts.items()
                }
                for tag, future in list(futures.items()) + [("aorc", rain_aorc)]:
                    output = model(dynamic, static, future[None].to(device), graph_supports)[0]
                    values[f"{name}+{tag}"] = raw_flow(output[:, graph["local"]], mean, std)
            for name, (model, rain_steps, lmean, lstd) in lstms.items():
                history, basin_static, rain_aorc, _ = series.batch(
                    np.full(len(scored), issue), scored, rain_steps, held[scored]
                )
                basin = {
                    tag: basin_forecast(field, rows, covered, inputs.rain, rain_steps)
                    for tag, field in forecasts.items()
                }
                for tag, rain_in in list(basin.items()) + [("aorc", rain_aorc)]:
                    z_out = model(
                        torch.as_tensor(history, device=device),
                        torch.as_tensor(basin_static, device=device),
                        torch.as_tensor(rain_in, device=device),
                    )
                    values[f"{name}+{tag}"] = raw_flow(z_out.T, lmean, lstd)
            values["nwm"] = torch.nan_to_num(nwm_flow)
            values["persistence"] = torch.nan_to_num(latest)[None].expand(HORIZON, -1).double()
            ahead = times[issue + np.arange(HORIZON)]
            climate = clim[ahead.dayofyear, ahead.hour // 6][:, scored]
            values["climatology"] = torch.as_tensor(np.nan_to_num(climate), device=device)
            for name, value in values.items():
                totals[name].add(value.double(), truth.double(), mask)
            kept_issue.append(issue)
            kept_truth.append(torch.where(mask, truth, torch.nan).float().cpu().numpy())
            kept_values.append(
                np.stack(
                    [
                        torch.where(mask, values[n].double(), torch.nan).float().cpu().numpy()
                        for n in names
                    ]
                )
            )
            if rank == 0 and (position % 50 == 0):
                print(
                    f"issues {position + 1}/{len(items[rank::world])} per process; "
                    f"{(time.perf_counter() - began) / 60:.1f} min",
                    flush=True,
                )
    for accumulator in totals.values():
        accumulator.reduce(world)
    np.savez(
        out / f"forecasts_{rank}.npz",
        issue=np.array(kept_issue, dtype=np.int64),
        truth=np.array(kept_truth, dtype=np.float32).reshape(-1, HORIZON, len(scored)),
        values=np.array(kept_values, dtype=np.float32).reshape(
            -1, len(names), HORIZON, len(scored)
        ),
    )
    if world > 1:
        dist.barrier()
    if rank == 0:
        stations = pd.DataFrame(
            dict(station=catalog.names[scored], usgs_id=usgs[scored], comid=catalog.comid[scored])
        )
        merge(out, world, names, stations, times)
        write(
            a,
            out,
            stations,
            names,
            totals,
            saved,
            len(items),
            times[data.issues[items[0]]],
            times[data.issues[items[-1]]],
            gauges_nc,
        )


def merge(out, world, names, stations, times):
    parts = [np.load(out / f"forecasts_{r}.npz") for r in range(world)]
    parts = [q for q in parts if len(q["issue"])]
    issue = np.concatenate([q["issue"] for q in parts])
    order = np.argsort(issue)
    np.savez_compressed(
        out / "forecasts.npz",
        issue=issue[order],
        time=times[issue[order]].to_numpy(),
        truth=np.concatenate([q["truth"] for q in parts])[order],
        values=np.concatenate([q["values"] for q in parts])[order],
        methods=np.array(names),
        stations=stations.station.to_numpy().astype(str),
        usgs_id=stations.usgs_id.to_numpy().astype(str),
    )
    for r in range(world):
        (out / f"forecasts_{r}.npz").unlink()


def write(a, out, stations, names, totals, saved, issues, first, last, gauges_nc):
    metrics = {n: reach_metrics(**acc.numpy(), min_count=a.min_count) for n, acc in totals.items()}
    rows = []
    for name, m in metrics.items():
        for lead in range(HORIZON):
            for s, st in enumerate(stations.itertuples()):
                rows.append(
                    dict(
                        station=st.station,
                        usgs_id=st.usgs_id,
                        method=name,
                        lead_hours=6 * (lead + 1),
                        values=int(m["count"][lead, s]),
                        scored=bool(m["scored"][lead, s]),
                        **{
                            k: _finite(m[k][lead, s])
                            for k in (
                                "nse",
                                "kge",
                                "r",
                                "alpha",
                                "beta",
                                "rmse",
                                "peak_error",
                                "peak_error_at_target_time",
                            )
                        },
                    )
                )
    table = pd.DataFrame(rows)
    table.to_csv(out / "per_station.csv", index=False)
    persistence = table[table.method == "persistence"].set_index(["station", "lead_hours"]).nse
    nwm = table[table.method == "nwm"].set_index(["station", "lead_hours"]).nse
    summary = []
    for (name, lead), part in table.groupby(["method", "lead_hours"]):
        part = part[part.scored]
        index = list(zip(part.station, [lead] * len(part)))
        own = part.nse.to_numpy(dtype=float)
        summary.append(
            dict(
                method=name,
                lead_hours=lead,
                stations=len(part),
                median_nse=_median(part.nse),
                mean_nse=_mean(part.nse),
                median_kge=_median(part.kge),
                median_rmse_m3s=_median(part.rmse),
                median_peak_error_at_target_time_pct=_pct(_median(part.peak_error_at_target_time)),
                median_beta=_median(part.beta),
                median_alpha=_median(part.alpha),
                fraction_stations_nse_above_nwm=float(
                    np.mean(own > nwm.reindex(index).to_numpy(dtype=float))
                )
                if len(part)
                else None,
                fraction_stations_nse_above_persistence=float(
                    np.mean(own > persistence.reindex(index).to_numpy(dtype=float))
                )
                if len(part)
                else None,
            )
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary_by_lead.csv", index=False)
    atomic_json(
        out / "evaluation_manifest.json",
        dict(
            checkpoints={
                n: dict(path=str(Path(pth).resolve()), sha256=sha(pth))
                for n, pth in a.models.items()
            },
            lstm=dict(a.lstms),
            gauges=str(gauges_nc),
            sites=a.sites,
            held_out_fold=dict(fold=a.holdout_fold, folds=a.folds, seed=a.fold_seed)
            if a.holdout_fold
            else None,
            stations=stations.station.tolist(),
            usgs_ids=stations.usgs_id.tolist(),
            issue_times=issues,
            first_issue=str(first),
            last_issue=str(last),
            issue_cycle="00Z, days with an NWM medium-range forecast",
            observations="USGS NWIS instantaneous discharge, six-hour means over [t, t+6h) UTC",
            rain="+nwm: NWM medium-range forcing rain (GFS) to each model's rain window; "
            "+gefs: GEFSv12 operational control; +aorc: realized AORC to the same window",
            nwm="medium_range_mem1 streamflow, trapezoid six-hour means of 3-hourly instantaneous values",
            scoring="values with an observation, a latest observation and an NWM forecast",
            tva_data_used=False,
        ),
    )
    lines = [
        f"# Benchmark at {len(stations)} gauges, issues {str(first)[:10]} to {str(last)[:10]} ({issues} 00Z issues)",
        "",
    ]
    lines += [
        "| method | " + " | ".join(f"{h} h NSE" for h in REPORT_LEADS) + " | days 1-5 mean |",
        "|---|" + "---:|" * (len(REPORT_LEADS) + 1),
    ]
    for name in names:
        part = summary[summary.method == name].set_index("lead_hours")
        cells = [_fmt(part.median_nse.get(h)) for h in REPORT_LEADS]
        days = part.median_nse.reindex([24, 48, 72, 96, 120])
        cells.append(_fmt(days.mean() if days.notna().all() else None))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Median NSE across scored gauges per lead (six-hour means)."]
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report)
    print(report, flush=True)
    print(f"BENCHMARK COMPLETE: {out}", flush=True)


def _median(series):
    v = pd.to_numeric(series, errors="coerce").dropna()
    return float(v.median()) if len(v) else None


def _mean(series):
    v = pd.to_numeric(series, errors="coerce").dropna()
    return float(v.mean()) if len(v) else None


def _pct(value):
    return None if value is None else 100 * value


def _finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def _fmt(value):
    return "–" if value is None or pd.isna(value) else f"{value:.3f}"


if __name__ == "__main__":
    main()
