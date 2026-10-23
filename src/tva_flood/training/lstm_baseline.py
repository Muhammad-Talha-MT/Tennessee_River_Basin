"""Train the lumped forecast LSTM baseline on observed flow at every catalog gauge.

Same information and periods as the graph model's fine-tuning: basin-mean AORC
weather history (``build-basin-series``), observed flow history up to the issue
time with an availability flag (10% of samples have it hidden, as gauge outages),
basin-mean attributes, and future basin-mean rain known only up to
``--rain-hours`` (realized AORC in training; no rain with probability 0.2, cut at
a random shorter lead with probability 0.5, as in the graph model's stage 1).
Targets are observed six-hour flow at the next 40 steps; issues whose targets end
before 2019 train, 2019 selects the checkpoint (median NSE over gauges, averaged
over the 6-120 h leads, with AORC rain to ``--rain-hours``; higher is better).

Loss: the graph model's stage-2 loss: lead-weighted normalized log-flow error
plus raw-flow error scaled by each gauge's training spread, x3 above its
training 90th percentile.

Public protocol (as ``observed-finetune --public``): ``--gauges`` from
``build-public-gauges``; ``--forecast-rain`` (GEFSv12 reforecast) replaces realized
rain for a ``--forecast-fraction`` share of samples (00Z issues); rain is unknown
with ``--unknown-rain-prob`` and cut short with ``--rain-cut-prob``; selection
(``training.selection``) uses ``--validation-rain`` at 00Z validation issues and
``--validation-gauges camelsh``. ``--sample-budget`` sets --updates from the
number of (issue, gauge) training pairs.

``--balanced`` draws a gauge uniformly, then one of its pairs (as the graph model
draws gauges), instead of drawing pairs uniformly. ``--validation-split`` selects on
the validation split's issues with the graph model's rule (00Z, a forecast, an
observed latest value and target at some selection gauge), so both select on
identical days. ``--holdout-fold k --folds n`` (``data.folds``) withholds one fold of
gauges entirely, for the ungauged test.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tva_flood.common import atomic_json
from tva_flood.data.basin_series import basin_forecast, basin_rows, covered_reaches
from tva_flood.data.folds import held_out
from tva_flood.data.gauges import Catalog
from tva_flood.data.rain_forecasts import RainForecasts
from tva_flood.data.river import River
from tva_flood.data.schema import HISTORY, HORIZON
from tva_flood.data.ssl import EncoderWindows
from tva_flood.models.lstm import ForecastLSTM

from .forecast import MAX_LOG_FLOW, lead_weights
from .selection import nse_by_lead, selection


class Series:
    """Basin series, observed flow and its normalized form on one timeline."""

    def __init__(self, preparation, catalog, mean, std):
        with np.load(preparation / "basin_series.npz", allow_pickle=True) as data:
            self.times = pd.DatetimeIndex(data["times"])
            self.weather = data["weather"]
            self.static = data["static"]
            names = data["gauge"].astype(str)
        if list(names) != list(catalog.names):
            raise ValueError("basin_series.npz gauges differ from the catalog")
        self.observed = catalog.on(self.times).astype(np.float64)
        finite = np.isfinite(self.observed)
        self.z = np.where(finite, (np.log1p(np.maximum(self.observed, 0)) - mean) / std, 0.0)
        self.z = self.z.astype(np.float32)
        self.finite = finite

    def batch(self, issues, gauges, rain_steps, hide=None, cut=None):
        """Inputs and raw targets for (issue index, gauge) pairs.

        ``hide`` [B] hides the flow history; ``cut`` [B] is the number of leads with
        known rain (default ``rain_steps``).
        """
        window = issues[:, None] + np.arange(-HISTORY, 0)[None]
        weather = self.weather[window, gauges[:, None]]  # [B, history, 16]
        flow = self.z[window, gauges[:, None]]
        flag = self.finite[window, gauges[:, None]].astype(np.float32)
        if hide is not None:
            flow = np.where(hide[:, None], 0.0, flow)
            flag = np.where(hide[:, None], 0.0, flag)
        history = np.concatenate([weather, flow[..., None], flag[..., None]], axis=-1)
        ahead = issues[:, None] + np.arange(HORIZON)[None]
        rain = self.weather[ahead, gauges[:, None]][..., [0, 8]]  # value, known share
        limit = np.full(len(issues), rain_steps) if cut is None else cut
        rain = np.where((np.arange(HORIZON)[None] < limit[:, None])[..., None], rain, 0.0)
        target = self.observed[ahead, gauges[:, None]]
        return history.astype(np.float32), self.static[gauges], rain.astype(np.float32), target


def forecast_inputs(root, prep, data, catalog, folder, times, steps):
    """Basin forecast rain [day, gauge, lead, 2] and each time's day (-1: none)."""
    forecasts = RainForecasts(folder)
    with np.load(prep / "graph.npz") as graph:
        ids = graph["feature_id"].copy()
        edges = np.column_stack([graph["source"], graph["target"]])
    rows = basin_rows(River.load(root, ids, edges), ids, catalog.comid)
    covered = covered_reaches(root, ids)
    rain = (data.precip_transform, float(data.mean[0]), float(data.std[0]))
    wanted = [t for t in times[times.hour == 0] if t in forecasts]
    basin = np.stack([basin_forecast(forecasts.get(t), rows, covered, rain, steps) for t in wanted])
    day = np.full(len(times), -1)
    day[times.get_indexer(wanted)] = np.arange(len(wanted))
    return basin, day


def raw(z, mean, std):
    return torch.expm1((z.double() * std + mean).clamp(max=MAX_LOG_FLOW)).clamp_min(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--preparation", type=Path)
    p.add_argument("--gauges", type=Path, help="Default: <root>/processed/gauges/gauges.nc")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--updates", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rain-hours", type=int, default=96)
    p.add_argument("--raw-weight", type=float, default=1.0)
    p.add_argument("--high-flow-weight", type=float, default=2.0)
    p.add_argument("--history-hide", type=float, default=0.1)
    p.add_argument("--unknown-rain-prob", type=float, default=0.2)
    p.add_argument("--rain-cut-prob", type=float, default=0.5, help="Rain known to a shorter lead")
    p.add_argument("--forecast-rain", type=Path, help="Public rain forecasts (download-gefs)")
    p.add_argument("--forecast-fraction", type=float, default=0.0)
    p.add_argument(
        "--validation-rain", type=Path, help="Select with this forecast rain (default: AORC)"
    )
    p.add_argument("--validation-gauges", choices=["all", "camelsh"], default="all")
    p.add_argument("--min-count", type=int, default=30)
    p.add_argument(
        "--sample-budget",
        type=int,
        default=0,
        help="Set --updates to consume this many (issue, gauge) pairs (0: off)",
    )
    p.add_argument("--balanced", action="store_true", help="Draw gauges uniformly, then pairs")
    p.add_argument(
        "--validation-split",
        action="store_true",
        help="Select on the validation split's issues, as finetune-observed --public",
    )
    p.add_argument("--holdout-fold", type=int, default=0, help="Withhold this fold (0: none)")
    p.add_argument("--folds", type=int, default=0)
    p.add_argument("--fold-seed", type=int, default=0)
    p.add_argument("--evaluations", type=int, default=0, help="Overrides --eval-every (0: off)")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.forecast_fraction and not a.forecast_rain:
        p.error("--forecast-fraction needs --forecast-rain")
    if a.holdout_fold and not 1 <= a.holdout_fold <= a.folds:
        p.error("--holdout-fold needs --folds and must be in 1..folds")
    if a.sample_budget:
        a.updates = math.ceil(a.sample_budget / a.batch_size)
    if a.evaluations:
        a.eval_every = max(1, math.ceil(a.updates / a.evaluations))
    root = a.root.resolve()
    prep = (a.preparation or root / "processed/pretraining_full").resolve()
    out = a.out.resolve()
    if (out / "best.pt").exists():
        p.error(f"{out} already holds a trained model")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device(a.device)
    data = EncoderWindows(root, prep, "train", HORIZON, 1)
    mean, std = float(data.mean[8]), float(data.std[8])
    catalog = Catalog((a.gauges or root / "processed/gauges/gauges.nc").resolve())
    series = Series(prep, catalog, mean, std)
    held = held_out(root, catalog, a.holdout_fold, a.folds, a.fold_seed)
    series.observed[:, held] = np.nan  # withheld gauges: no target, no history
    series.z[:, held] = 0.0
    series.finite[:, held] = False
    times = series.times
    rain_steps = a.rain_hours // 6
    end = pd.Timestamp(data.manifest["splits"]["train"][1])
    issue = np.arange(HISTORY, len(times) - HORIZON + 1)  # the last full target window too
    latest = series.finite[issue - 1]
    counts = np.vstack([np.zeros((1, len(catalog))), np.cumsum(series.finite, 0)])
    any_target = counts[issue + HORIZON] - counts[issue] > 0
    usable = latest & any_target
    train_rows = times[issue + HORIZON - 1] < end
    pairs = np.argwhere(usable & train_rows[:, None])
    train_issue, train_gauge = issue[pairs[:, 0]], pairs[:, 1]
    v_gauges = (
        np.flatnonzero(catalog.camelsh)
        if a.validation_gauges == "camelsh"
        else np.arange(len(catalog))
    )
    if a.validation_split:
        split = EncoderWindows(root, prep, "validation", HORIZON, 1)
        if not split.times.equals(times):
            raise ValueError("basin_series.npz and the preparation have different timelines")
        candidates = np.asarray(split.issues)
        ok = (
            series.finite[candidates - 1][:, v_gauges]
            & (counts[candidates + HORIZON] - counts[candidates] > 0)[:, v_gauges]
        )
        valid = np.isin(issue, candidates[ok.any(axis=1)])
    else:
        valid = (times[issue] >= end) & (times[issue] < end + pd.DateOffset(years=1))
    valid &= times[issue].hour == 0
    forecast_basin, forecast_day = {}, {}
    for folder in {f for f in (a.forecast_rain, a.validation_rain) if f}:
        forecast_basin[folder], forecast_day[folder] = forecast_inputs(
            root, prep, data, catalog, folder, times, rain_steps
        )
    if a.validation_rain:
        valid &= forecast_day[a.validation_rain][issue] >= 0
    v_issue = issue[valid]
    if a.forecast_fraction:
        with_forecast = np.flatnonzero(forecast_day[a.forecast_rain][train_issue] >= 0)

    def by_gauge(rows):
        """Pair rows sorted by gauge, with each gauge's first row and count."""
        order = rows[np.argsort(train_gauge[rows], kind="stable")]
        _, start, count = np.unique(train_gauge[order], return_index=True, return_counts=True)
        return order, start, count

    def balanced(group, size):
        """A gauge uniformly, then one of its pairs."""
        order, start, count = group
        k = rng.integers(0, len(start), size)
        return order[start[k] + (rng.random(size) * count[k]).astype(np.int64)]

    if a.balanced:
        groups = dict(all=by_gauge(np.arange(len(train_issue))))
        if a.forecast_fraction:
            groups["forecast"] = by_gauge(with_forecast)
    before = series.observed[: int(times.searchsorted(end))]
    with np.errstate(all="ignore"):
        scale = np.nan_to_num(np.maximum(np.nanstd(before, axis=0), 1e-2), nan=1.0)
        high = np.nan_to_num(np.nanquantile(before, 0.9, axis=0), nan=np.inf)
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=device)
    high_t = torch.as_tensor(high, dtype=torch.float32, device=device)
    weights = torch.as_tensor(lead_weights("balanced"), dtype=torch.float32, device=device)
    model = ForecastLSTM(hidden=a.hidden, dropout=a.dropout, horizon=HORIZON).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    warmup = min(500, a.updates // 10)

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        fraction = min(1.0, (step - warmup) / max(a.updates - warmup, 1))
        return 0.05 + 0.95 * (1 + math.cos(math.pi * fraction)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    config = dict(
        model="ForecastLSTM (handoff encoder-decoder, lumped)",
        hidden=a.hidden,
        dropout=a.dropout,
        horizon=HORIZON,
        history=HISTORY,
        rain_hours=a.rain_hours,
        flow_mean=mean,
        flow_std=std,
        gauges=catalog.names.tolist(),
        preparation=str(prep),
        training_pairs=int(len(train_issue)),
        training_pairs_with_forecast_rain=int(len(with_forecast)) if a.forecast_fraction else 0,
        sample_budget=int(a.updates * a.batch_size),
        selection="training.selection: mean over 6-120 h of median NSE over "
        f"{a.validation_gauges} gauges, 00Z validation issues, "
        f"{'forecast rain ' + str(a.validation_rain) if a.validation_rain else 'AORC rain'}",
        validation_issues=int(len(v_issue)),
        held_out_gauges=catalog.names[held].tolist(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
    )
    atomic_json(out / "run_config.json", config)
    print(f"{len(train_issue):,} training (issue, gauge) pairs; {len(v_issue)} validation issues")

    def validate():
        model.eval()
        predictions = np.full((len(v_issue), HORIZON, len(catalog)), np.nan)
        targets = np.full_like(predictions, np.nan)
        with torch.no_grad():
            for g in v_gauges:
                gauges = np.full(len(v_issue), g)
                history, static, rain, target = series.batch(v_issue, gauges, rain_steps)
                if a.validation_rain:
                    day = forecast_day[a.validation_rain][v_issue]
                    rain = forecast_basin[a.validation_rain][day, g]
                z = model(
                    torch.as_tensor(history, device=device),
                    torch.as_tensor(static, device=device),
                    torch.as_tensor(rain, device=device),
                )
                predictions[:, :, g] = raw(z, mean, std).cpu().numpy()
                ok = series.finite[v_issue - 1, g][:, None]
                targets[:, :, g] = np.where(ok, target, np.nan)
        model.train()
        scores = nse_by_lead(predictions[:, :, v_gauges], targets[:, :, v_gauges], a.min_count)
        score, median = selection(scores)
        return score, {f"nse_{6 * (k + 1)}h": float(median[k]) for k in (0, 3, 11, 19)}

    best, history_rows, began = -np.inf, [], time.perf_counter()
    model.train()
    for update in range(1, a.updates + 1):
        if a.balanced:
            pick = balanced(groups["all"], a.batch_size)
        else:
            pick = rng.integers(0, len(train_issue), a.batch_size)
        issues, gauges = train_issue[pick], train_gauge[pick]
        hide = rng.random(a.batch_size) < a.history_hide
        draw = rng.random(a.batch_size)
        cut = np.where(
            draw < a.unknown_rain_prob,
            0,
            np.where(
                rng.random(a.batch_size) < a.rain_cut_prob,
                rng.integers(4, rain_steps + 1, a.batch_size),
                rain_steps,
            ),
        )
        use = np.zeros(a.batch_size, dtype=bool)
        if a.forecast_fraction:  # drawn only when used: older runs keep their stream
            use = rng.random(a.batch_size) < a.forecast_fraction
            if a.balanced:
                pick[use] = balanced(groups["forecast"], int(use.sum()))
            else:
                pick[use] = with_forecast[rng.integers(0, len(with_forecast), int(use.sum()))]
            issues, gauges = train_issue[pick], train_gauge[pick]
            use &= cut > 0  # unknown rain stays unknown
            cut[use] = rain_steps
        history, static, rain, target = series.batch(issues, gauges, rain_steps, hide, cut)
        if use.any():
            day = forecast_day[a.forecast_rain][issues[use]]
            rain[use] = forecast_basin[a.forecast_rain][day, gauges[use]]
        z = model(
            torch.as_tensor(history, device=device),
            torch.as_tensor(static, device=device),
            torch.as_tensor(rain, device=device),
        )
        target = torch.as_tensor(target, dtype=torch.float32, device=device)
        mask = torch.isfinite(target)
        truth = torch.nan_to_num(target)
        weight = weights[None] * mask
        total = weight.sum().clamp_min(1)
        truth_z = (torch.log1p(truth.clamp_min(0)) - mean) / std
        log_term = (weight * (z - truth_z).square()).sum() / total
        g = torch.as_tensor(gauges, device=device)
        error = (raw(z, mean, std).float() - truth) / scale_t[g][:, None]
        emphasis = 1 + a.high_flow_weight * (truth > high_t[g][:, None]).float()
        raw_term = (weight * emphasis * error.square()).sum() / total
        loss = log_term + a.raw_weight * raw_term
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if update % 200 == 0:
            print(
                json.dumps(
                    dict(
                        update=update,
                        loss=round(loss.item(), 4),
                        log=round(log_term.item(), 4),
                        minutes=round((time.perf_counter() - began) / 60, 1),
                    )
                ),
                flush=True,
            )
        if update % a.eval_every == 0 or update == a.updates:
            score, extra = validate()
            improved = score > best
            row = dict(update=update, selection=score, best=improved, **extra)
            history_rows.append(row)
            pd.DataFrame(history_rows).to_csv(out / "validation_progress.csv", index=False)
            print(f"validation: {json.dumps(row)}", flush=True)
            state = dict(
                config=config, update=update, model=model.state_dict(), scale=scale, high=high
            )
            if improved:
                best = score
                torch.save(state, out / "best.pt")
    torch.save(state, out / "last.pt")
    print(f"Done. Best selection {best:.4f}; {out / 'best.pt'}")


if __name__ == "__main__":
    main()
