"""Skill against observed flow at TVA headwater gauges, beside TVA's own forecasts.

Forecast mode, as in operation. For each issue time where TVA issued a forecast
the model gets weather up to the issue time, observed flow up to then at every
catalog gauge (no NWM flow), and future precipitation only up to the rain
window. All methods are scored on the same values: leads with observed flow and
a TVA forecast, at issue times whose latest observation exists.

Per checkpoint (rain differs):

- ``+qpf``: TVA's basin forecast precipitation over each TVA gauge basin (TVA lead
  6(k+1) h = step k), unknown elsewhere: the fair comparison with TVA;
- ``+aorc``: realized AORC precipitation over the same basins: a perfect rain
  forecast, an upper bound that isolates forecast-rain error;
- ``+unknown``: no future rain.

References: ``persistence`` (the latest observed six-hour mean), ``tva_forecast``
(TVA's issued flow forecast) and ``nwm_retrospective`` (the NWM simulation driven
by realized AORC, scored only where NWM flow exists: it ends in January 2023).
Observed flow is TVA's six-hour mean over [t, t+6h) UTC; see ``data.tva``.
"""

import argparse
import os
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from tva_flood.common import atomic_json, sha
from tva_flood.data import tva
from tva_flood.data.gauges import Catalog
from tva_flood.data.inputs import ForecastInputs
from tva_flood.data.river import River
from tva_flood.data.schema import HORIZON
from tva_flood.data.ssl import EncoderWindows
from tva_flood.models.sparse import sparse_supports

from .hydrology import Accumulator, load_forecast, raw_flow, reach_metrics, read_forecast

LEADS = (1, 4, 12, 20, 40)  # 6 h, 24 h, 72 h, 120 h, 240 h
RAIN = ("qpf", "aorc", "unknown")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--preparation", type=Path)
    p.add_argument("--array-cache", type=Path)
    p.add_argument("--gauges", type=Path, help="Default: <root>/processed/gauges/gauges.nc")
    p.add_argument(
        "--tva-archive", type=Path, default=Path("/data/tva_data/raw/tva/TVA_headwater.zip")
    )
    p.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="train-forecast or finetune-observed checkpoint to score; repeat for several",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="test needs --preparation (and --array-cache) from prepare-test",
    )
    p.add_argument("--issues-from", help="Only issue times at or after this date")
    p.add_argument("--issues-until", help="Only issue times before this date")
    p.add_argument(
        "--rain-hours",
        type=int,
        help="Lead up to which future rain is known (default: each checkpoint's own)",
    )
    p.add_argument("--min-count", type=int, default=100)
    p.add_argument("--cache-months", type=int, default=4)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = p.parse_args()
    try:
        a.models = dict(item.split("=", 1) for item in a.checkpoint)
    except ValueError:
        p.error("--checkpoint must be NAME=PATH")
    rank, world, local = [
        int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0))
    ]
    device = torch.device(f"cuda:{local}" if a.device == "cuda" else "cpu")
    if a.device == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        kwargs = dict(
            backend="nccl" if a.device == "cuda" else "gloo", timeout=timedelta(minutes=60)
        )
        if a.device == "cuda":
            kwargs["device_id"] = device
        dist.init_process_group(**kwargs)
    try:
        with tempfile.TemporaryDirectory() as scratch:
            evaluate(a, p, rank, world, device, Path(scratch))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def evaluate(a, p, rank, world, device, scratch):
    root = a.root.resolve()
    prep = (a.preparation or root / "processed/pretraining_full").resolve()
    out = a.out.resolve()
    if rank == 0:
        if out.exists():
            p.error(f"{out} exists; choose a new --out")
        out.mkdir(parents=True)
    saved = {name: read_forecast(Path(path)) for name, path in a.models.items()}
    for name, s in saved.items():
        if not s["config"].get("decoder"):
            p.error(f"{name} has no future-rain decoder; train it in forecast mode")
    steps = {n: (a.rain_hours or s["config"].get("rain_hours", 96)) // 6 for n, s in saved.items()}
    data = EncoderWindows(
        root,
        prep,
        a.split,
        HORIZON,
        a.cache_months,
        array_cache=a.array_cache.resolve() if a.array_cache else None,
        future_rain=True,
    )
    with np.load(prep / "graph.npz") as graph:
        edges = np.column_stack([graph["source"], graph["target"]])
    a.gauges = (a.gauges or root / "processed/gauges/gauges.nc").resolve()
    catalog = Catalog(a.gauges)
    halo = max(
        s["config"]["architecture"]["graph_layers"] + s["config"]["decoder"]["decoder_graph_layers"]
        for s in saved.values()
    )
    supports = sparse_supports(len(data.ids), edges)
    directions = {s["config"]["architecture"].get("direction", "both") for s in saved.values()}
    if len(directions) > 1:
        p.error("Score upstream-only (whole-basin) and other checkpoints in separate runs")
    river = River.load(root, data.ids, edges) if directions == {"upstream"} else None
    folder = tva.extract(a.tva_archive, scratch)
    inputs = ForecastInputs(data, catalog, folder, edges, supports, halo, river)
    table = inputs.stations
    if table.empty:
        raise ValueError("No TVA station is in the gauge catalog with observed flow and forecasts")
    graph = inputs.graph(inputs.station_gauge)
    graph_supports = [s.to(device) for s in graph["supports"]]
    static = graph["static"].to(device)
    models = {name: load_forecast(s, data, prep, supports, device) for name, s in saved.items()}
    mean, std = inputs.mean, inputs.std
    variants = {name: [f"{name}+{rain}" for rain in RAIN] for name in models}
    names = [n for m in models for n in variants[m]]
    names += ["nwm_retrospective", "persistence", "tva_forecast"]
    totals = {name: Accumulator(len(table), device) for name in names}
    # Only issue times where TVA issued a forecast: every method shares these values.
    items = [i for i, issue in enumerate(data.issues) if int(issue) in inputs.forecast]
    first, last = (
        pd.Timestamp(a.issues_from or "1900-01-01"),
        pd.Timestamp(a.issues_until or "2200-01-01"),
    )
    items = [i for i in items if first <= data.times[data.issues[i]] < last]
    if not items:
        raise ValueError("No issue time with a TVA forecast in the requested period")
    local = items[rank::world]
    saved_issues, saved_truth, saved_values = [], [], []
    truth_all = inputs.observed[:, inputs.station_gauge]
    began = time.perf_counter()
    with torch.inference_mode():
        for position, item in enumerate(local):
            sample = inputs.load(data, item, graph)
            issue = sample["issue"]
            truth = torch.as_tensor(truth_all[issue : issue + HORIZON], device=device)
            latest = torch.as_tensor(truth_all[issue - 1], device=device)
            forecast_tva = torch.as_tensor(inputs.forecast[issue], device=device)
            mask = (
                torch.isfinite(truth) & torch.isfinite(forecast_tva) & torch.isfinite(latest)[None]
            )
            if not mask.any():
                continue
            truth = torch.nan_to_num(truth)
            dynamic = sample["dynamic"][None].to(device)
            values = {}
            for name, model in models.items():
                for rain in RAIN:
                    spec = None if rain == "unknown" else rain
                    future = inputs.future_rain(sample, graph, spec, steps[name])[None]
                    output = model(dynamic, static, future.to(device), graph_supports)[0]
                    values[f"{name}+{rain}"] = raw_flow(output[:, graph["local"]], mean, std)
            nwm = sample["target"].to(device)[:, graph["local"]]
            values["nwm_retrospective"] = raw_flow(nwm, mean, std)
            values["persistence"] = torch.nan_to_num(latest)[None].expand(HORIZON, -1).double()
            values["tva_forecast"] = torch.nan_to_num(forecast_tva)
            # NWM flow is scored only where it exists (none after January 2023).
            simulated = sample["target_valid"].to(device)[:, graph["local"]]
            kept = {}
            for name, value in values.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"Nonfinite {name} forecast")
                scored = mask & simulated if name == "nwm_retrospective" else mask
                totals[name].add(value.double(), truth.double(), scored)
                kept[name] = torch.where(scored, value.double(), torch.nan).float().cpu().numpy()
            saved_issues.append(issue)
            saved_truth.append(torch.where(mask, truth, torch.nan).float().cpu().numpy())
            saved_values.append(np.stack([kept[n] for n in names]))
            if rank == 0 and (position % 50 == 0 or position + 1 == len(local)):
                elapsed = time.perf_counter() - began
                print(
                    f"issues {position + 1}/{len(local)} per process; "
                    f"{elapsed / 60:.1f} min elapsed",
                    flush=True,
                )
    for accumulator in totals.values():
        accumulator.reduce(world)
    np.savez(
        out / f"forecasts_{rank}.npz",
        issue=np.array(saved_issues, dtype=np.int64),
        truth=np.array(saved_truth, dtype=np.float32),
        values=np.array(saved_values, dtype=np.float32),
    )
    if world > 1:
        dist.barrier()
    if rank == 0:
        merge_forecasts(out, world, names, table, data)
        write(a, out, table, totals, data, items, saved, steps)


def merge_forecasts(out, world, names, table, data):
    """One ``forecasts.npz``: every forecast and observation, NaN where a method was not scored.

    ``values[issue, method, lead, station]`` and ``truth[issue, lead, station]`` are
    m3/s at TVA lead 6(k+1) h; ``time`` is the issue time of each row.
    """
    parts = [np.load(out / f"forecasts_{r}.npz") for r in range(world)]
    parts = [p for p in parts if len(p["issue"])]
    issue = np.concatenate([p["issue"] for p in parts])
    order = np.argsort(issue)
    np.savez_compressed(
        out / "forecasts.npz",
        issue=issue[order],
        time=data.times[issue[order]].to_numpy(),
        truth=np.concatenate([p["truth"] for p in parts])[order],
        values=np.concatenate([p["values"] for p in parts])[order],
        methods=np.array(names),
        stations=table.tva_nwsid.to_numpy().astype(str),
    )
    for r in range(world):
        (out / f"forecasts_{r}.npz").unlink()


def write(a, out, table, totals, data, items, saved, steps):
    metrics = {n: reach_metrics(**acc.numpy(), min_count=a.min_count) for n, acc in totals.items()}
    station_rows, summary_rows = [], []
    for name, values in metrics.items():
        for lead in range(HORIZON):
            scored = values["scored"][lead]
            for s, row in enumerate(table.itertuples()):
                station_rows.append(
                    dict(
                        station=row.tva_nwsid,
                        comid=row.comid,
                        method=name,
                        lead_hours=6 * (lead + 1),
                        values=int(values["count"][lead, s]),
                        scored=bool(scored[s]),
                        mean_observed_m3s=values["mean_flow"][lead, s],
                        **{
                            k: _finite(values[k][lead, s])
                            for k in (
                                "nse",
                                "kge",
                                "r",
                                "alpha",
                                "beta",
                                "rmse",
                                "mse",
                                "peak_error",
                                "peak_error_at_target_time",
                            )
                        },
                    )
                )
    stations_frame = pd.DataFrame(station_rows)
    stations_frame.to_csv(out / "per_station.csv", index=False)
    tva = stations_frame[stations_frame.method == "tva_forecast"].set_index(
        ["station", "lead_hours"]
    )
    for (name, lead), part in stations_frame.groupby(["method", "lead_hours"]):
        part = part[part.scored]
        reference = tva.loc[[(s, lead) for s in part.station], "nse"].to_numpy(dtype=float)
        summary_rows.append(
            dict(
                method=name,
                lead_hours=lead,
                stations=len(part),
                median_nse=_median(part.nse),
                median_kge=_median(part.kge),
                median_rmse_m3s=_median(part.rmse),
                median_mse=_median(part.mse),
                median_peak_error_pct=_pct(_median(part.peak_error)),
                median_abs_peak_error_pct=_pct(_median(part.peak_error.abs())),
                median_peak_error_at_target_time_pct=_pct(_median(part.peak_error_at_target_time)),
                median_alpha=_median(part.alpha),
                median_beta=_median(part.beta),
                median_r=_median(part.r),
                fraction_stations_nse_above_tva=float(
                    np.mean(part.nse.to_numpy(dtype=float) > reference)
                )
                if len(part)
                else None,
            )
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "summary_by_lead.csv", index=False)
    table.drop(
        columns=[c for c in table.columns if c.startswith(("gageii", "streamcat"))], errors="ignore"
    ).to_csv(out / "stations.csv", index=False)
    atomic_json(
        out / "evaluation_manifest.json",
        dict(
            checkpoints={
                n: dict(
                    path=str(Path(p).resolve()),
                    sha256=sha(p),
                    update=saved[n].get("update"),
                    rain_hours=6 * steps[n],
                    history_gauge_catalog=sha(a.gauges),
                )
                for n, p in a.models.items()
            },
            tva_archive=str(a.tva_archive.resolve()),
            tva_archive_sha256=sha(a.tva_archive),
            stations=table.tva_nwsid.tolist(),
            split=a.split,
            issue_times=len(items),
            first_issue=str(data.times[data.issues[items[0]]]),
            last_issue=str(data.times[data.issues[items[-1]]]),
            alignment="observed [t,t+6h) UTC; TVA lead 6(k+1) h = model lead k",
            precipitation_coverage="TVA gauge basins (smallest gauge basin if nested); unknown elsewhere",
            history="observed flow at every catalog gauge; no NWM flow",
            scoring="issues with a TVA forecast and a latest observation; leads with observed flow",
            optimizer_steps=0,
        ),
    )
    period = f"{str(data.times[data.issues[items[0]]])[:10]} to {str(data.times[data.issues[items[-1]]])[:10]}"
    report = render(summary, saved, f"{a.split}, issues {period}")
    (out / "report.md").write_text(report)
    print(report, flush=True)
    print(f"EVALUATION COMPLETE: {out}", flush=True)


def _median(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else None


def _pct(value):
    return None if value is None else 100 * value


def _finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def render(summary, saved, period):
    lines = [f"# Observed-flow evaluation at TVA headwater gauges ({period})", ""]
    lines += [
        "",
        "| method | lead | stations | median NSE | median KGE | median RMSE | "
        "peak err at obs peak % | stations NSE > TVA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    part = summary[summary.lead_hours.isin([6 * k for k in LEADS])]
    for _, row in part.sort_values(["lead_hours", "method"]).iterrows():
        cells = [
            _fmt(row.median_nse, "{:.3f}"),
            _fmt(row.median_kge, "{:.3f}"),
            _fmt(row.median_rmse_m3s, "{:.2f}"),
            _fmt(row.median_peak_error_at_target_time_pct, "{:+.1f}"),
            _fmt(row.fraction_stations_nse_above_tva, "{:.0%}"),
        ]
        lines.append(
            f"| {row.method} | {row.lead_hours}h | {row.stations} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value, pattern):
    return "–" if value is None or pd.isna(value) else pattern.format(value)


if __name__ == "__main__":
    main()
