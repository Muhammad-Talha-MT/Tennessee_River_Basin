"""Per-reach hydrologic skill of a train-forecast checkpoint on a held-out year or period.

For each reach and lead time, forecasts issued at consecutive six-hour issue
times form a hydrograph that is compared with the NWM target series at the
same valid times. Per reach and lead: NSE, KGE (with r, alpha, beta), MSE and
RMSE in m3/s, and peak magnitude error (the forecast series' maximum against
the target maximum, and the forecast at the time of the target maximum).

Medians across reaches are reported for all reaches and by mean-flow
class, with the pooled RMSE over all values, which the largest rivers
dominate. Persistence is scored on exactly the same values. With full history, all methods use targets whose
latest history is available; the no-history scenario hides every flow
history. Targets are simulated NWM streamflow, not gauge observations.
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

from tva_flood.common import atomic_json, sha
from tva_flood.data.dataset import lineage
from tva_flood.data.river import River
from tva_flood.data.schema import HORIZON
from tva_flood.data.ssl import FLOW, FLOW_VALID, EncoderWindows
from tva_flood.models.sparse import sparse_supports
from tva_flood.models.ssl import ForecastModel, TemporalGraphEncoder, WeatherForecastModel
from tva_flood.training.forecast import MAX_LOG_FLOW
from tva_flood.training.runtime import reduced, sample_windows

HEADLINE_LEADS = (1, 4, 12, 20, 40)  # 6 h, 24 h, 72 h, 120 h, 240 h
REACH_TABLE_LEADS = (4, 12, 40)
FLOW_CLASSES = (
    ("all", 0.0),
    ("mean>=1", 1.0),
    ("mean>=10", 10.0),
    ("mean>=100", 100.0),
    ("mean>=1000", 1000.0),
)
FOCUS = [1868000, 1862014, 10575371, 1868002, 19633480]


def raw_flow(z, mean, std):
    value = (z.double() * std + mean).clamp(max=MAX_LOG_FLOW)
    return torch.expm1(value).clamp_min(0)


class Accumulator:
    """Additive per-reach sums for every lead, and the target-peak bookkeeping."""

    def __init__(self, nodes, device):
        shape = (HORIZON, nodes)
        self.sums = torch.zeros(shape + (6,), dtype=torch.float64, device=device)
        self.target_peak = torch.full(shape, -np.inf, dtype=torch.float64, device=device)
        self.forecast_peak = torch.full(shape, -np.inf, dtype=torch.float64, device=device)
        self.forecast_at_target_peak = torch.zeros(shape, dtype=torch.float64, device=device)

    def add(self, forecast, target, mask):
        weight = mask.double()
        s, o = forecast * weight, target * weight
        self.sums += torch.stack([weight, s, o, s * s, o * o, s * o], dim=-1)
        target_masked = torch.where(mask, target, -np.inf)
        newer = target_masked > self.target_peak
        self.forecast_at_target_peak = torch.where(newer, forecast, self.forecast_at_target_peak)
        self.target_peak = torch.maximum(self.target_peak, target_masked)
        self.forecast_peak = torch.maximum(self.forecast_peak, torch.where(mask, forecast, -np.inf))

    def reduce(self, world):
        reduced(self.sums, world)
        if world == 1:
            return
        # torch.where in add() inherits the forecast's strides, which may not be contiguous.
        self.target_peak = self.target_peak.contiguous()
        self.forecast_peak = self.forecast_peak.contiguous()
        self.forecast_at_target_peak = self.forecast_at_target_peak.contiguous()
        dist.all_reduce(self.forecast_peak, op=dist.ReduceOp.MAX)
        peaks = [torch.empty_like(self.target_peak) for _ in range(world)]
        values = [torch.empty_like(self.target_peak) for _ in range(world)]
        dist.all_gather(peaks, self.target_peak)
        dist.all_gather(values, self.forecast_at_target_peak)
        self.target_peak, self.forecast_at_target_peak = merge_peaks(peaks, values)

    def numpy(self):
        return dict(
            sums=self.sums.cpu().numpy(),
            target_peak=self.target_peak.cpu().numpy(),
            forecast_peak=self.forecast_peak.cpu().numpy(),
            forecast_at_target_peak=self.forecast_at_target_peak.cpu().numpy(),
        )


def merge_peaks(peaks, values):
    """Combine per-process target maxima and the forecasts at those times."""
    peaks, values = torch.stack(peaks), torch.stack(values)
    winner = peaks.argmax(dim=0, keepdim=True)
    return peaks.gather(0, winner)[0], values.gather(0, winner)[0]


def reach_metrics(sums, target_peak, forecast_peak, forecast_at_target_peak, min_count=100):
    """Per-reach metrics from additive sums; arrays share any leading shape.

    A reach is scored when it has at least ``min_count`` values, a positive
    target mean and a target standard deviation of at least 1e-3 m3/s; NSE and
    KGE are undefined for constant series. Variances are population variances,
    which cancel in NSE and in KGE's ratios.
    """
    n, ss, so, sss, soo, sso = np.moveaxis(np.asarray(sums, dtype=np.float64), -1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_s, mean_o = ss / n, so / n
        var_s = np.maximum(sss / n - mean_s**2, 0)
        var_o = np.maximum(soo / n - mean_o**2, 0)
        mse = np.maximum((sss - 2 * sso + soo) / n, 0)
        covariance = sso / n - mean_s * mean_o
        scored = (n >= min_count) & (mean_o > 0) & (np.sqrt(var_o) >= 1e-3)
        r = np.where(var_s > 0, covariance / np.sqrt(var_s * var_o), np.nan)
        alpha = np.sqrt(var_s / var_o)
        beta = mean_s / mean_o
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        nse = 1 - mse / var_o
        peak = (forecast_peak - target_peak) / target_peak
        peak_at_time = (forecast_at_target_peak - target_peak) / target_peak
    nan = np.full_like(mse, np.nan)
    return dict(
        count=n,
        mean_flow=mean_o,
        scored=scored,
        mse=mse,
        rmse=np.sqrt(mse),
        nse=np.where(scored, nse, nan),
        kge=np.where(scored, kge, nan),
        r=np.where(scored, r, nan),
        alpha=np.where(scored, alpha, nan),
        beta=np.where(scored, beta, nan),
        target_peak=target_peak,
        peak_error=np.where(scored, peak, nan),
        peak_error_at_target_time=np.where(scored, peak_at_time, nan),
    )


def summarize(metrics, sums, members):
    """Distribution across the member reaches of one lead."""
    valid = members & metrics["scored"]
    count = sums[..., 0]
    pooled = members & (count > 0)
    squared = (metrics["mse"] * count)[pooled].sum()

    def q(name, level):
        values = metrics[name][valid]
        return float(np.nanquantile(values, level)) if np.isfinite(values).any() else None

    def fraction(name, threshold):
        values = metrics[name][valid]
        values = values[np.isfinite(values)]
        return float((values > threshold).mean()) if len(values) else None

    return dict(
        reaches=int(members.sum()),
        reaches_scored=int(valid.sum()),
        median_nse=q("nse", 0.5),
        q25_nse=q("nse", 0.25),
        q75_nse=q("nse", 0.75),
        fraction_nse_above_0=fraction("nse", 0.0),
        median_kge=q("kge", 0.5),
        q25_kge=q("kge", 0.25),
        q75_kge=q("kge", 0.75),
        # KGE > -0.41 beats the target-mean benchmark (Knoben et al., 2019).
        fraction_kge_above_mean_benchmark=fraction("kge", 1 - np.sqrt(2)),
        median_r=q("r", 0.5),
        median_alpha=q("alpha", 0.5),
        median_beta=q("beta", 0.5),
        median_mse_m3s2=q("mse", 0.5),
        median_rmse_m3s=q("rmse", 0.5),
        pooled_mse_m3s2=float(squared / count[pooled].sum()) if pooled.any() else None,
        pooled_rmse_m3s=float(np.sqrt(squared / count[pooled].sum())) if pooled.any() else None,
        median_peak_error_pct=_pct(q("peak_error", 0.5)),
        median_abs_peak_error_pct=_pct(
            float(np.nanmedian(np.abs(metrics["peak_error"][valid]))) if valid.any() else None
        ),
        median_peak_error_at_target_time_pct=_pct(q("peak_error_at_target_time", 0.5)),
    )


def _pct(value):
    return None if value is None or not np.isfinite(value) else 100 * value


def read_forecast(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != "encoder-forecast-v1":
        raise ValueError(f"{path} is not a train-forecast checkpoint")
    return saved


def load_forecast(saved, data, prep, supports, device, weights=True):
    """The checkpoint's model; ``weights=False`` keeps its fresh initialization."""
    config = saved["config"]
    if config["dataset_identity"] != lineage(data.manifest):
        raise ValueError("Forecast checkpoint was trained on different prepared data")
    if config["normalization_sha256"] != sha(prep / "dynamic_normalization.json"):
        raise ValueError("Forecast checkpoint used a different normalization")
    encoder = TemporalGraphEncoder([s.to(device) for s in supports], **config["architecture"])
    if config.get("decoder"):
        model = WeatherForecastModel(encoder, HORIZON, **config["decoder"]).to(device)
    else:
        model = ForecastModel(encoder, HORIZON).to(device)
    if weights:
        model.load_state_dict(saved["model"], strict=True)
    return model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--preparation", type=Path)
    p.add_argument("--array-cache", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True, help="train-forecast checkpoint")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="test needs --preparation (and --array-cache) from prepare-test",
    )
    p.add_argument(
        "--windows",
        type=int,
        default=0,
        help="0 scores every issue time (continuous hydrographs; recommended)",
    )
    p.add_argument(
        "--min-count",
        type=int,
        default=100,
        help="Minimum valid values for a reach to receive NSE/KGE",
    )
    p.add_argument("--focus-comids", type=int, nargs="*", default=FOCUS)
    p.add_argument("--cache-months", type=int, default=4)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = p.parse_args()
    if a.windows < 0 or a.min_count < 2:
        p.error("Nonnegative --windows and --min-count >= 2 required")
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
        evaluate(a, p, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def evaluate(a, p, rank, world, device):
    root = a.root.resolve()
    prep = (a.preparation or root / "processed/pretraining_full").resolve()
    out = a.out.resolve()
    if rank == 0:
        if out.exists():
            p.error(f"{out} exists; choose a new --out")
        out.mkdir(parents=True)
    saved = read_forecast(a.checkpoint)
    decoder = saved["config"].get("decoder")
    future = bool(decoder)
    rain_steps = saved["config"].get("rain_hours", 6 * HORIZON) // 6
    gauges = saved["config"].get("history_gauge_comids")
    data = EncoderWindows(
        root,
        prep,
        a.split,
        HORIZON,
        a.cache_months,
        array_cache=a.array_cache.resolve() if a.array_cache else None,
        future_rain=future,
    )
    with np.load(prep / "graph.npz") as graph:
        ids = graph["feature_id"].copy()
        edges = np.column_stack([graph["source"], graph["target"]])
    missing = sorted(set(a.focus_comids) - set(ids.tolist()))
    if missing:
        p.error(f"Focus COMIDs not in graph: {missing}")
    supports = sparse_supports(len(ids), edges)
    model = load_forecast(saved, data, prep, supports, device)
    if saved["config"]["architecture"].get("direction") == "upstream":
        river = River.load(root, ids, edges)
        model.encoder.full_routing = river.operator(np.arange(len(ids))).to(device)
    static = torch.as_tensor(data.static, dtype=torch.float32, device=device)
    visible = None if gauges is None else torch.as_tensor(np.isin(ids, gauges), device=device)
    mean, std = float(data.mean[8]), float(data.std[8])
    methods = {"full": ["forecast", "persistence"], "none": ["forecast"]}
    totals = {(s, m): Accumulator(len(ids), device) for s in methods for m in methods[s]}
    items = sample_windows(len(data), a.windows or len(data))
    local = items[rank::world]
    began = time.perf_counter()
    with torch.inference_mode():
        for position, item in enumerate(local):
            sample = data.load(int(item))
            target = raw_flow(sample["target"].to(device), mean, std)
            valid = sample["target_valid"].to(device)
            history = sample["dynamic"][None].to(device)
            for scenario, names in methods.items():
                dynamic = history.clone()
                if scenario == "none":
                    dynamic[..., FLOW : FLOW_VALID + 1] = 0
                elif visible is not None:
                    dynamic[..., ~visible, FLOW : FLOW_VALID + 1] = 0
                available = dynamic[0, -1, :, FLOW_VALID] > 0.5
                mask = valid & available[None] if scenario == "full" else valid
                if future:
                    weather = sample["future"][None].to(device).clone()
                    weather[:, rain_steps:] = 0
                    forecasts = {"forecast": model(dynamic, static, weather)[0]}
                else:
                    forecasts = {"forecast": model(dynamic, static)[0]}
                if "persistence" in names:
                    forecasts["persistence"] = dynamic[0, -1, :, FLOW][None].expand(HORIZON, -1)
                for name, value in forecasts.items():
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f"Nonfinite {name} forecast")
                    # With flow history only at gauges, the model is scored at every reach.
                    scored = valid if name == "forecast" and visible is not None else mask
                    totals[(scenario, name)].add(raw_flow(value, mean, std), target, scored)
            if rank == 0 and (position % 25 == 0 or position + 1 == len(local)):
                elapsed = time.perf_counter() - began
                remaining = elapsed / (position + 1) * (len(local) - position - 1)
                print(
                    f"windows {position + 1}/{len(local)} per GPU; "
                    f"{elapsed / 60:.1f} min elapsed, about {remaining / 60:.1f} min left",
                    flush=True,
                )
    for accumulator in totals.values():
        accumulator.reduce(world)
    if rank != 0:
        return
    write_results(a, out, ids, items, data, saved, totals, methods)


def write_results(a, out, ids, items, data, saved, totals, methods):
    results = {key: acc.numpy() for key, acc in totals.items()}
    metrics = {key: reach_metrics(**value, min_count=a.min_count) for key, value in results.items()}
    # Flow classes come from the full-history 6 h target mean, shared by all methods.
    reference = metrics[("full", "forecast")]["mean_flow"][0]
    focus = np.isin(ids, a.focus_comids)
    rows = []
    for (scenario, method), values in metrics.items():
        sums = results[(scenario, method)]["sums"]
        for lead in range(HORIZON):
            lead_metrics = {k: v[lead] for k, v in values.items()}
            for name, threshold in FLOW_CLASSES:
                members = np.nan_to_num(reference, nan=-1) >= threshold
                rows.append(
                    dict(
                        scenario=scenario,
                        method=method,
                        lead_hours=6 * (lead + 1),
                        flow_class=name,
                        **summarize(lead_metrics, sums[lead], members),
                    )
                )
    by_lead = pd.DataFrame(rows)
    by_lead.to_csv(out / "by_lead.csv", index=False)
    headline = by_lead[by_lead.lead_hours.isin([6 * h for h in HEADLINE_LEADS])]
    headline.to_csv(out / "summary.csv", index=False)
    focus_rows = []
    for (scenario, method), values in metrics.items():
        for lead in HEADLINE_LEADS:
            for index in np.flatnonzero(focus):
                focus_rows.append(
                    dict(
                        comid=int(ids[index]),
                        scenario=scenario,
                        method=method,
                        lead_hours=6 * lead,
                        mean_flow_m3s=values["mean_flow"][lead - 1, index],
                        target_peak_m3s=values["target_peak"][lead - 1, index],
                        **{
                            k: _finite(values[k][lead - 1, index])
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
    pd.DataFrame(focus_rows).to_csv(out / "focus_reaches.csv", index=False)
    for lead in REACH_TABLE_LEADS:
        table = {"comid": ids, "mean_flow_m3s": reference}
        for (scenario, method), values in metrics.items():
            for k in ("nse", "kge", "rmse", "peak_error"):
                table[f"{scenario}_{method}_{k}"] = values[k][lead - 1]
        pd.DataFrame(table).to_parquet(out / f"per_reach_lead_{6 * lead:03d}h.parquet", index=False)
    atomic_json(
        out / "evaluation_manifest.json",
        dict(
            checkpoint=str(a.checkpoint.resolve()),
            checkpoint_sha256=sha(a.checkpoint),
            checkpoint_update=saved.get("update"),
            initialization=saved["config"]["initialization"],
            future_rain=saved["config"].get("future_rain", "none"),
            history_gauges=saved["config"].get("history_gauge_comids") is not None,
            validation_windows=len(items),
            all_windows=len(items) == len(data),
            first_issue=str(data.times[data.issues[items[0]]]),
            last_issue=str(data.times[data.issues[items[-1]]]),
            reaches=len(ids),
            min_count=a.min_count,
            methods={k: v for k, v in methods.items()},
            targets="NWM simulated streamflow (not gauge observations)",
            optimizer_steps=0,
        ),
    )
    report = render(headline, methods, a.split)
    (out / "report.md").write_text(report)
    print(report, flush=True)
    print(f"EVALUATION COMPLETE: {out}", flush=True)


def _finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def render(headline, methods, split):
    lines = [f"# Hydrologic evaluation ({split} split, NWM targets)", ""]
    columns = [
        ("median_nse", "median NSE", "{:.3f}"),
        ("median_kge", "median KGE", "{:.3f}"),
        ("median_rmse_m3s", "median RMSE", "{:.2f}"),
        ("pooled_rmse_m3s", "pooled RMSE", "{:.1f}"),
        ("median_peak_error_pct", "median peak err %", "{:+.1f}"),
        ("median_abs_peak_error_pct", "median |peak err| %", "{:.1f}"),
        ("reaches_scored", "reaches", "{:d}"),
    ]
    for scenario in methods:
        for flow_class in ("all", "mean>=100"):
            part = headline[(headline.scenario == scenario) & (headline.flow_class == flow_class)]
            lines += [
                f"## {scenario} history, {flow_class} reaches",
                "",
                "| method | lead | " + " | ".join(c[1] for c in columns) + " |",
                "|---|---:|" + "---:|" * len(columns),
            ]
            for _, row in part.sort_values(["lead_hours", "method"]).iterrows():
                cells = [
                    "–"
                    if row[k] is None or pd.isna(row[k])
                    else fmt.format(int(row[k]) if fmt == "{:d}" else row[k])
                    for k, _, fmt in columns
                ]
                lines.append(f"| {row.method} | {row.lead_hours}h | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
