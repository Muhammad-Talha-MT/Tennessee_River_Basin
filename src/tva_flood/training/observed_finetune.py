"""Fine-tune a forecast-mode model on observed flow at every catalog gauge.

Forecast mode: at issue time t the model sees weather up to t, observed flow up
to t at every gauge (catalog: TVA headwater gauges and all CAMELS-H gauges in
Tennessee; no other flow history exists), and future precipitation only up to
--rain-hours. Targets are observed six-hour flow at the gauges for the next
40 steps.

Future rain is a proxy for TVA's forecast during training: TVA's own basin
forecast at TVA gauges from October 2016 (a share --qpf-fraction of each batch
is drawn from that era), and realized AORC elsewhere. Each sample knows rain for
a random share of gauge basins and for none with probability --unknown-rain-prob,
so the model also works where no forecast exists, as at CAMELS-H basins.

Each update draws --gauges-per-update gauges and --batch-size issue times. The
subgraph is those gauges' reaches plus a halo of as many hops as the model has
graph blocks, so their outputs equal full-graph outputs; flow history is visible
at every catalog gauge inside it, and the loss is at the drawn gauges only.

With --rain-error the realized AORC rain gets the errors of TVA's rain forecast
(``data.rain_error``), so training rain is no better than test-time rain.

Loss: normalized log-flow error plus a raw-flow error scaled by each gauge's
training spread. Above the gauge's training 90th percentile the raw error weighs
1 + --high-flow-weight times more, and under-prediction there a further
1 + --under-weight times (flood peaks are what a log loss learns to shrink).
--ema-decay keeps an exponential moving average of the weights, which is what is
validated and saved.

Training uses issues whose targets end before 2019 (or before --select-from, whose
period then joins 2019 as validation). --flood-fraction draws that share of samples
from issue times that reach the gauge's --flood-quantile flow within 10 days.
Validation uses TVA forecast rain as the only rain. ``selection`` (higher is better) is the median
over TVA gauges of NSE averaged over the 6-120 h leads (``nse_selection``), plus
--csi-weight times the critical success index for flow above each gauge's
training --flood-quantile, pooled over TVA gauges and averaged over the same
leads (``csi_selection``). The same scores of TVA's own forecast are
``tva_selection``, ``tva_nse_selection`` and ``tva_csi_selection``; CAMELS-H gauges
are reported separately. Test years (2020 on) are never touched.

Public protocol (``--public``, for the paper and the LSTM comparison): no TVA data
at all. The catalog must hold public series only (``build-public-gauges``), TVA's
archive is not read (no TVA forecasts, no rain errors), and selection uses the
same validation forecasts and score as ``train-lstm`` (``training.selection``):
00Z validation issues with ``--forecast-rain`` (GEFSv12 reforecast) as the only
rain, at every CAMELS-H gauge. ``--forecast-fraction`` draws that share of training
samples from 00Z issues with ``--forecast-rain``, which then replaces realized rain
over the whole subgraph (in either protocol).

``--score-inside`` (whole-basin models) puts the loss at every catalog gauge inside
the update's subgraph, not only the drawn ones: their whole basins are inside, so
their outputs are exact. ``--sample-budget`` sets --updates to consume that many
usable (issue, gauge) training pairs, the unit of ``train-lstm``'s budget.

``--from-scratch`` takes only the architecture (and data normalization) from
--init: weights start from a fresh initialization (seeded by --seed), so the
model learns from observed flow alone, with no NWM-trained stage 1. With it,
``--ablate`` removes parts of the river-network design to measure what each adds:
``routing`` (the decoder's travel-time routing block), ``basin`` (basin-mean
weather history and basin-mean forecast rain) and ``graph`` (the encoder's and
decoder's message-passing hops). Training samples do not depend on the model, so
every ablation sees the same (issue, gauge) draws.

``--holdout-fold k --folds n`` (``data.folds``) withholds one fold of gauges
completely, for the ungauged test: no targets, no flow history, no selection. Their
reaches stay in the river network.
"""

import argparse
import json
import math
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tva_flood.common import atomic_json, sha
from tva_flood.data import tva
from tva_flood.data.folds import held_out
from tva_flood.data.gauges import Catalog
from tva_flood.data.inputs import ForecastInputs
from tva_flood.data.rain_forecasts import RainForecasts
from tva_flood.data.river import River
from tva_flood.data.schema import HORIZON
from tva_flood.data.ssl import EncoderWindows
from tva_flood.evaluation.hydrology import (
    Accumulator,
    load_forecast,
    raw_flow,
    reach_metrics,
    read_forecast,
)
from tva_flood.models.sparse import sparse_supports

from .encoder_loop import derived_seed
from .forecast import lead_weights
from .runtime import capture_rng, restore_rng, save_atomic
from .selection import SELECTION_LEADS, nse_by_lead, selection

REPORT_LEADS = (1, 4, 12, 20, 40)

ABLATIONS = dict(
    routing=dict(decoder=dict(routing_bins=0)),
    basin=dict(architecture=dict(basin_inputs=False), decoder=dict(basin_rain=False)),
    graph=dict(architecture=dict(graph_layers=0), decoder=dict(decoder_graph_layers=0)),
)
# Arguments that define the experiment; a resumed run must match them exactly.
FIXED = (
    "init",
    "gauges",
    "tva_archive",
    "updates",
    "batch_size",
    "gauges_per_update",
    "lr",
    "encoder_lr_scale",
    "weight_decay",
    "warmup_updates",
    "min_lr_factor",
    "clip",
    "rain_hours",
    "qpf_fraction",
    "unknown_rain_prob",
    "rain_known_fraction",
    "gauge_dropout",
    "log_weight",
    "raw_weight",
    "high_flow_weight",
    "under_weight",
    "lead_weighting",
    "rain_error",
    "ema_decay",
    "csi_weight",
    "flood_quantile",
    "max_nodes",
    "flood_fraction",
    "select_from",
    "public",
    "forecast_rain",
    "forecast_fraction",
    "score_inside",
    "sample_budget",
    "amp",
    "from_scratch",
    "ablate",
    "holdout_fold",
    "folds",
    "fold_seed",
    "seed",
)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--preparation", type=Path)
    p.add_argument("--array-cache", type=Path)
    p.add_argument("--gauges", type=Path, help="Default: <root>/processed/gauges/gauges.nc")
    p.add_argument(
        "--tva-archive", type=Path, default=Path("/data/tva_data/raw/tva/TVA_headwater.zip")
    )
    p.add_argument(
        "--init", type=Path, required=True, help="Forecast-mode train-forecast checkpoint"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--updates", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=8, help="Issue times per update")
    p.add_argument("--gauges-per-update", type=int, default=32)
    p.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="Cap on reaches per update (whole-basin models): gauges are added while their "
        "subgraph stays within it, and fewer issue times are used if one basin exceeds it. "
        "0: no cap",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--encoder-lr-scale", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-updates", type=int, default=100)
    p.add_argument("--min-lr-factor", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--rain-hours", type=int, help="Default: the --init checkpoint's")
    p.add_argument(
        "--qpf-fraction",
        type=float,
        help="Share of samples from the TVA forecast era (TVA rain at TVA gauges); "
        "default 0.25, 0 with --public",
    )
    p.add_argument("--unknown-rain-prob", type=float, default=0.1)
    p.add_argument(
        "--rain-known-fraction",
        type=float,
        nargs=2,
        default=[0.1, 1.0],
        metavar=("LOW", "HIGH"),
        help="Share of gauge basins with known rain is drawn from this range",
    )
    p.add_argument(
        "--gauge-dropout",
        type=float,
        default=0.1,
        help="Probability that a gauge's flow history is hidden (outage)",
    )
    p.add_argument("--log-weight", type=float, default=1.0)
    p.add_argument("--raw-weight", type=float, default=1.0)
    p.add_argument("--high-flow-weight", type=float, default=2.0)
    p.add_argument(
        "--under-weight",
        type=float,
        default=0.0,
        help="Extra weight on under-predicting flow above the gauge's 90th percentile",
    )
    p.add_argument("--lead-weighting", choices=["balanced", "uniform"], default="balanced")
    p.add_argument(
        "--rain-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Give realized AORC training rain the errors of TVA's rain forecast",
    )
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="Validate and save a moving average of the weights with this decay (0: off)",
    )
    p.add_argument(
        "--csi-weight",
        type=float,
        default=0.0,
        help="Weight of the flood CSI in the selection score (0: NSE only)",
    )
    p.add_argument(
        "--flood-quantile",
        type=float,
        default=0.99,
        help="Per-gauge training quantile of six-hour flow that defines a flood for the CSI",
    )
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument(
        "--flood-fraction",
        type=float,
        default=0.0,
        help="Share of training samples drawn from issue times whose next 10 days reach the "
        "gauge's --flood-quantile flow (at 0.99 about 10%% of uniform draws)",
    )
    p.add_argument(
        "--select-from",
        help="Hold issue times from this date out of training and add them to the 2019 "
        "validation issues (e.g. 2018-01-01 selects on 2018-2019)",
    )
    p.add_argument("--validation-issues", type=int, default=300, help="0: every issue")
    p.add_argument(
        "--validation-gauges",
        choices=["all", "tva"],
        default="all",
        help="Validate at every gauge, or at TVA gauges only (CAMELS-H columns then empty)",
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="No TVA data: public catalog, no TVA archive, selection as train-lstm "
        "(--forecast-rain at 00Z validation issues, CAMELS-H gauges)",
    )
    p.add_argument(
        "--forecast-rain",
        type=Path,
        help="Public rain forecasts per reach (download-gefs), e.g. the GEFSv12 reforecast",
    )
    p.add_argument(
        "--forecast-fraction",
        type=float,
        default=0.0,
        help="Share of training samples with --forecast-rain instead of realized rain",
    )
    p.add_argument(
        "--score-inside",
        action="store_true",
        help="Loss at every catalog gauge inside the subgraph (whole-basin models)",
    )
    p.add_argument(
        "--sample-budget",
        type=int,
        default=0,
        help="Set --updates to consume this many usable (issue, gauge) training pairs (0: off)",
    )
    p.add_argument(
        "--evaluations",
        type=int,
        default=0,
        help="Validate this many times, evenly spaced (overrides --eval-every; 0: off)",
    )
    p.add_argument(
        "--holdout-fold",
        type=int,
        default=0,
        help="Withhold this fold of gauges entirely (data.folds; 0: none)",
    )
    p.add_argument("--folds", type=int, default=0, help="Number of folds for --holdout-fold")
    p.add_argument("--fold-seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="bfloat16 autocast")
    p.add_argument(
        "--from-scratch",
        action="store_true",
        help="Architecture from --init only; weights freshly initialized (no stage 1)",
    )
    p.add_argument(
        "--ablate",
        nargs="+",
        choices=sorted(ABLATIONS),
        default=[],
        help="With --from-scratch: remove these parts of the river-network design",
    )
    p.add_argument(
        "--validation-chunk", type=int, default=4, help="Issues per forward in --public validation"
    )
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--min-count", type=int, default=100)
    p.add_argument("--cache-months", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = p.parse_args()
    if a.qpf_fraction is None:
        a.qpf_fraction = 0.0 if a.public else 0.25
    if a.holdout_fold and not 1 <= a.holdout_fold <= a.folds:
        p.error("--holdout-fold needs --folds and must be in 1..folds")
    if a.ablate and not a.from_scratch:
        p.error("--ablate needs --from-scratch (pretrained weights would not fit)")
    a.ablate = sorted(set(a.ablate))
    if a.public and (a.qpf_fraction or a.rain_error or not a.forecast_rain):
        p.error("--public needs --forecast-rain, --qpf-fraction 0 and no --rain-error")
    if a.forecast_fraction and not a.forecast_rain:
        p.error("--forecast-fraction needs --forecast-rain")
    if a.public:
        a.tva_archive = None
    probabilities = (
        a.qpf_fraction,
        a.forecast_fraction,
        a.unknown_rain_prob,
        a.gauge_dropout,
        *a.rain_known_fraction,
    )
    if (
        min(probabilities) < 0
        or max(probabilities) > 1
        or a.rain_known_fraction[0] > a.rain_known_fraction[1]
    ):
        p.error("Probabilities and shares must be in [0,1]; --rain-known-fraction LOW <= HIGH")
    if a.max_nodes < 0:
        p.error("--max-nodes must be >= 0")
    if min(a.updates, a.batch_size, a.gauges_per_update, a.eval_every, a.log_every) < 1:
        p.error("Positive --updates, --batch-size, --gauges-per-update, --eval-every, --log-every")
    if a.lr <= 0 or a.encoder_lr_scale < 0 or a.clip <= 0 or not 0 < a.min_lr_factor <= 1:
        p.error("Positive --lr and --clip, nonnegative encoder scale, --min-lr-factor in (0,1]")
    if (
        min(a.log_weight, a.raw_weight, a.high_flow_weight, a.under_weight, a.csi_weight) < 0
        or not a.log_weight + a.raw_weight
    ):
        p.error("Nonnegative loss and selection weights, with a positive log or raw weight")
    if not 0 <= a.flood_fraction <= 1:
        p.error("--flood-fraction must be in [0,1]")
    if a.select_from:
        try:
            a.select_from = str(pd.Timestamp(a.select_from).date())
        except ValueError:
            p.error("--select-from must be a date")
    if not 0 <= a.ema_decay < 1 or not 0 < a.flood_quantile < 1:
        p.error("--ema-decay in [0,1), --flood-quantile in (0,1)")
    return p, a


def signature(a):
    return {
        k: str(getattr(a, k).resolve()) if isinstance(getattr(a, k), Path) else getattr(a, k)
        for k in FIXED
    }


def same_run(saved, current):
    """A resumed run must match; options newer than the run must be off (their default)."""
    return all(saved.get(k, v if not v else None) == v for k, v in current.items())


def issue_lists(inputs, split_data, steps):
    """Issue indices per gauge with a latest observation and a target, and QPF-era flags."""
    observed = np.isfinite(inputs.observed)
    counts = np.vstack([np.zeros((1, observed.shape[1]), dtype=np.int64), np.cumsum(observed, 0)])
    issues = split_data.issues
    usable = observed[issues - 1] & (counts[issues + HORIZON] - counts[issues] > 0)
    has_qpf = np.zeros(len(issues), dtype=bool)
    for k, issue in enumerate(issues):
        values = inputs.qpf.get(int(issue))
        has_qpf[k] = values is not None and np.isfinite(values[:steps]).any()
    return usable, has_qpf


def forecast_issues(data, forecasts):
    """Issues of ``data`` at 00Z with a public rain forecast (all False without one)."""
    if forecasts is None:
        return np.zeros(len(data.issues), dtype=bool)
    times = data.times[data.issues]
    return np.array([t.hour == 0 and t in forecasts for t in times])


def draw(update, inputs, usable, has_qpf, a, flooded=None, has_forecast=None):
    """Core gauges, issue times and per-sample rain/outage draws for one update.

    Also returns the scored gauges (the core, or with --score-inside every catalog
    gauge inside the subgraph) and the number of usable (issue, scored gauge) pairs.
    """
    rng = np.random.default_rng(derived_seed(a.seed, update))
    n = len(inputs.catalog)
    eligible = np.flatnonzero(usable.any(axis=0))
    core = rng.choice(eligible, min(a.gauges_per_update, len(eligible)), replace=False)
    issues = a.batch_size
    taken = None
    if a.max_nodes:
        taken = np.zeros(len(inputs.data.ids), dtype=bool)
        kept = []
        for g in core:
            extra = (~taken[inputs.around[g]]).sum()
            if kept and taken.sum() + extra > a.max_nodes:
                continue
            taken[inputs.around[g]] = True
            kept.append(g)
        core = np.array(kept)
        # One basin larger than the cap trains alone, on fewer issue times.
        issues = max(1, min(a.batch_size, a.batch_size * a.max_nodes // int(taken.sum())))
    core = np.sort(core)
    if a.score_inside:
        if taken is None:
            taken = np.zeros(len(inputs.data.ids), dtype=bool)
            taken[np.concatenate([inputs.around[g] for g in core])] = True
        scored = np.flatnonzero(taken[inputs.outlet])
    else:
        scored = core
    tva_core = core[inputs.is_tva[core]]
    batch = []
    for _ in range(issues):
        qpf_era = rng.random() < a.qpf_fraction
        # Drawn only when used, so runs without forecast rain keep their random stream.
        forecast_era = bool(a.forecast_fraction) and rng.random() < a.forecast_fraction
        forecast_era = forecast_era and not qpf_era
        pool = tva_core if qpf_era else core
        gauge = int(rng.choice(pool if len(pool) else core))
        wanted = usable[:, gauge] & (has_qpf if qpf_era else True)
        if forecast_era:
            wanted = wanted & has_forecast
        if flooded is not None and rng.random() < a.flood_fraction:
            wanted = wanted & flooded[:, gauge]
        candidates = np.flatnonzero(wanted)
        if not len(candidates):
            candidates = np.flatnonzero(usable[:, gauge])
        item = int(rng.choice(candidates))
        share = rng.uniform(*a.rain_known_fraction)
        known = None if rng.random() < a.unknown_rain_prob else rng.random(n) < share
        hidden = rng.random(n) < a.gauge_dropout
        error = int(rng.integers(2**31)) if a.rain_error else None
        use_forecast = forecast_era and bool(has_forecast[item])
        batch.append((item, known, hidden, error, use_forecast))
    pairs = int(sum(usable[item, scored].sum() for item, *_ in batch))
    return core, batch, scored, pairs


def plan_updates(inputs, usable, has_qpf, a, flooded, has_forecast):
    """Updates whose draws consume --sample-budget usable (issue, gauge) pairs."""
    total, update = 0, 0
    while total < a.sample_budget:
        total += draw(update, inputs, usable, has_qpf, a, flooded, has_forecast)[3]
        update += 1
    return update, total


class Objective:
    def __init__(self, inputs, scale, high, a, device):
        self.mean, self.std = inputs.mean, inputs.std
        self.scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
        self.high = torch.as_tensor(high, dtype=torch.float32, device=device)
        self.weights = torch.as_tensor(
            lead_weights(a.lead_weighting), dtype=torch.float32, device=device
        )
        self.a = a

    def __call__(self, prediction, target, gauge):
        """prediction [B, lead, K] normalized log; target raw [B, lead, K], NaN if unobserved.

        ``gauge`` [K] holds the catalog index of each column (for per-gauge scales).
        """
        mask = torch.isfinite(target)
        truth = torch.nan_to_num(target)
        weight = self.weights[None, :, None] * mask
        total = weight.sum().clamp_min(1)
        truth_z = (torch.log1p(truth.clamp_min(0)) - self.mean) / self.std
        log_term = (weight * (prediction - truth_z).square()).sum() / total
        flow = raw_flow(prediction, self.mean, self.std).float()
        error = (flow - truth) / self.scale[gauge]
        high = (truth > self.high[gauge]).float()
        emphasis = (1 + self.a.high_flow_weight * high) * (
            1 + self.a.under_weight * high * (flow < truth).float()
        )
        raw_term = (weight * emphasis * error.square()).sum() / total
        loss = self.a.log_weight * log_term + self.a.raw_weight * raw_term
        return loss, dict(
            log_mse=log_term.item(), raw_term=raw_term.item(), targets=int(mask.sum())
        )


def validate(model, inputs, pairs, a, device, flood):
    """TVA's forecast rain only (at TVA gauge basins); observed flow history at every gauge.

    ``pairs`` are (dataset, item) issue times; ``flood`` [gauge] is the flow above which
    a value counts as a flood for the CSI.
    """
    model.eval()
    count = len(inputs.catalog)
    # All gauges, or only TVA's (whole-basin graphs of all 128 gauges are slow to score).
    chosen = np.arange(count) if a.validation_gauges == "all" else inputs.station_gauge
    graph = inputs.graph(chosen)
    in_graph = torch.zeros(count, dtype=torch.bool, device=device)
    in_graph[torch.as_tensor(chosen, device=device)] = True
    supports = [s.to(device) for s in graph["supports"]]
    static = graph["static"].to(device)
    names = ("model", "tva", "persistence")
    totals = {name: Accumulator(count, device) for name in names}
    steps = a.rain_hours // 6
    is_tva = torch.as_tensor(inputs.is_tva, device=device)
    flood = torch.as_tensor(flood, dtype=torch.float64, device=device)[None]
    contingency = {
        n: torch.zeros(3, HORIZON, dtype=torch.float64, device=device) for n in ("model", "tva")
    }
    step = 1 if len(graph["nodes"]) > 15000 else 2  # whole-basin graphs are large
    with torch.inference_mode():
        for start in range(0, len(pairs), step):
            chunk = pairs[start : start + step]
            samples = [inputs.load(data, i, graph) for data, i in chunk]
            dynamic = torch.stack([s["dynamic"] for s in samples]).to(device)
            rain = torch.stack([inputs.future_rain(s, graph, "qpf", steps) for s in samples])
            output = model(dynamic, static, rain.to(device), supports)
            prediction = torch.zeros(len(chunk), HORIZON, count, dtype=torch.float64, device=device)
            prediction[:, :, torch.as_tensor(chosen, device=device)] = raw_flow(
                output[:, :, graph["local"]], inputs.mean, inputs.std
            )
            for row, sample in enumerate(samples):
                issue = sample["issue"]
                truth = torch.as_tensor(inputs.observed[issue : issue + HORIZON], device=device)
                latest = torch.as_tensor(inputs.observed[issue - 1], device=device)
                tva_forecast = torch.full((HORIZON, count), float("nan"), device=device)
                tva_forecast[:, inputs.station_gauge] = torch.as_tensor(
                    inputs.forecast[issue], dtype=torch.float32, device=device
                )
                mask = torch.isfinite(truth) & torch.isfinite(latest)[None] & in_graph[None]
                mask &= torch.isfinite(tva_forecast) | ~is_tva
                truth = torch.nan_to_num(truth)
                tva_forecast = torch.nan_to_num(tva_forecast).double()
                totals["model"].add(prediction[row], truth, mask)
                totals["tva"].add(tva_forecast, truth, mask)
                scored = mask & is_tva[None]
                observed = truth >= flood
                for name, value in (("model", prediction[row]), ("tva", tva_forecast)):
                    warned = value >= flood
                    contingency[name] += torch.stack(
                        [
                            (observed & warned & scored).sum(1),
                            (observed & ~warned & scored).sum(1),
                            (~observed & warned & scored).sum(1),
                        ]
                    )
                totals["persistence"].add(
                    torch.nan_to_num(latest)[None].expand(HORIZON, -1), truth, mask
                )
    model.train()
    metrics = {k: reach_metrics(**v.numpy(), min_count=a.min_count) for k, v in totals.items()}
    groups = dict(tva=inputs.is_tva, camelsh=~inputs.is_tva)

    def median(name, key, lead, group):
        scored = metrics[name]["scored"][lead] & groups[group]
        values = metrics[name][key][lead][scored]
        return float(np.nanmedian(values)) if len(values) else float("nan")

    def selection(name, group):
        return float(np.nanmean([median(name, "nse", k, group) for k in range(SELECTION_LEADS)]))

    def csi(name):
        hits, misses, false = contingency[name].cpu().numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            return hits / (hits + misses + false)

    def flood_score(name):
        values = csi(name)[:SELECTION_LEADS]
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    def combined(name):
        return float(selection(name, "tva") + a.csi_weight * np.nan_to_num(flood_score(name)))

    row = dict(
        selection=combined("model"),
        tva_selection=combined("tva"),
        nse_selection=selection("model", "tva"),
        tva_nse_selection=selection("tva", "tva"),
        csi_selection=flood_score("model"),
        tva_csi_selection=flood_score("tva"),
        camelsh_selection=selection("model", "camelsh"),
        camelsh_persistence_selection=selection("persistence", "camelsh"),
        validation_issues=len(pairs),
    )
    for lead in REPORT_LEADS:
        k = lead - 1
        for name in names:
            row[f"{name}_nse_{6 * lead}h"] = median(name, "nse", k, "tva")
        row[f"camelsh_nse_{6 * lead}h"] = median("model", "nse", k, "camelsh")
        row[f"model_csi_{6 * lead}h"] = float(csi("model")[k])
        row[f"tva_csi_{6 * lead}h"] = float(csi("tva")[k])
        both = metrics["model"]["scored"][k] & metrics["tva"]["scored"][k] & groups["tva"]
        row[f"wins_vs_tva_{6 * lead}h"] = int(
            (metrics["model"]["nse"][k][both] > metrics["tva"]["nse"][k][both]).sum()
        )
    return row


def validate_public(model, inputs, pairs, a, device, forecasts):
    """Selection as ``train-lstm``: --forecast-rain only, every CAMELS-H gauge.

    ``pairs`` are (dataset, item) 00Z issue times with a forecast. Truth needs an
    observation at the issue time too (as persistence and the benchmark).
    """
    model.eval()
    gauges = np.flatnonzero(inputs.catalog.camelsh)
    graph = inputs.graph(gauges)
    supports = [s.to(device) for s in graph["supports"]]
    static = graph["static"].to(device)
    steps = a.rain_hours // 6
    prediction, truth, persistence = [], [], []
    if device.type == "cuda":
        torch.cuda.empty_cache()  # release blocks cached by training
    size, begin = a.validation_chunk, 0
    with torch.inference_mode(), torch.autocast(device.type, torch.bfloat16, enabled=a.amp):
        while begin < len(pairs):
            chunk = pairs[begin : begin + size]
            samples = [inputs.load(data, item, graph) for data, item in chunk]
            rain = [
                inputs.rain_field(s, graph, forecasts.get(data.times[s["issue"]]), steps)
                for s, (data, _) in zip(samples, chunk)
            ]
            dynamic = torch.stack([s["dynamic"] for s in samples]).to(device)
            try:
                output = model(dynamic, static, torch.stack(rain).to(device), supports)
            except (torch.OutOfMemoryError, RuntimeError) as error:
                # Too large for one forward (memory, or a 32-bit index limit in convolutions).
                if size == 1 or not (
                    isinstance(error, torch.OutOfMemoryError) or "32BitIndexMath" in str(error)
                ):
                    raise
                size //= 2  # fewer issues per forward from here on
                del dynamic
                torch.cuda.empty_cache()
                continue
            flow = raw_flow(output[:, :, graph["local"]].float(), inputs.mean, inputs.std)
            prediction += list(flow.cpu().numpy())
            begin += len(chunk)
            for s in samples:
                latest = inputs.observed[s["issue"] - 1, gauges]
                target = inputs.observed[s["issue"] : s["issue"] + HORIZON][:, gauges]
                truth.append(np.where(np.isfinite(latest)[None], target, np.nan))
                persistence.append(np.broadcast_to(latest, target.shape))
    model.train()
    truth = np.stack(truth)
    score, median = selection(nse_by_lead(np.stack(prediction), truth, a.min_count))
    base, base_median = selection(nse_by_lead(np.stack(persistence), truth, a.min_count))
    row = dict(
        selection=score,
        nse_selection=score,
        persistence_selection=base,
        validation_issues=len(pairs),
    )
    for lead in REPORT_LEADS:
        row[f"model_nse_{6 * lead}h"] = float(median[lead - 1])
        row[f"persistence_nse_{6 * lead}h"] = float(base_median[lead - 1])
    return row


def schedule(a):
    warmup = min(a.warmup_updates, a.updates // 10)

    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        fraction = min(1.0, max(0.0, (step - warmup) / max(a.updates - warmup, 1)))
        return a.min_lr_factor + (1 - a.min_lr_factor) * (1 + math.cos(math.pi * fraction)) / 2

    return factor


def assemble(inputs, data, core, batch, scored, steps, forecasts=None):
    """Training batch for one update, built in the loader thread.

    A sample drawn with forecast rain (and not unknown rain) takes the public
    forecast over the whole subgraph; otherwise realized rain as ``future_rain``.
    """
    graph = inputs.graph(core)
    graph["gauge_index"] = torch.as_tensor(scored)
    graph["scored_local"] = np.searchsorted(graph["nodes"], inputs.outlet[scored])
    samples = [inputs.load(data, item, graph, hidden) for item, _, hidden, _, _ in batch]
    rain = []
    for sample, (_, known, _, error, use_forecast) in zip(samples, batch):
        if use_forecast and known is not None:
            field = forecasts.get(data.times[sample["issue"]])
            rain.append(inputs.rain_field(sample, graph, field, steps))
        else:
            noise = None if error is None else np.random.default_rng(error)
            rain.append(inputs.future_rain(sample, graph, known, steps, noise))
    dynamic = torch.stack([s["dynamic"] for s in samples])
    target = torch.stack([inputs.target(s["issue"], scored) for s in samples])
    return graph, dynamic, torch.stack(rain), target


def main():
    p, a = arguments()
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    root = a.root.resolve()
    prep = (a.preparation or root / "processed/pretraining_full").resolve()
    a.gauges = (a.gauges or root / "processed/gauges/gauges.nc").resolve()
    out = a.out.resolve()
    last = out / "last.pt"
    saved = read_forecast(a.init)
    config = saved["config"]
    if a.ablate:
        config = json.loads(json.dumps(config))  # a copy: the checkpoint records the ablation
        for name in a.ablate:
            for section, values in ABLATIONS[name].items():
                config[section].update(values)
        config["ablated"] = a.ablate
        saved = dict(saved, config=config)
    decoder = config.get("decoder")
    if not decoder or decoder.get("future_channels") != 2:
        p.error("--init must be a train-forecast checkpoint with future precipitation")
    a.rain_hours = a.rain_hours or config.get("rain_hours", 96)
    if not 6 <= a.rain_hours <= 6 * HORIZON or a.rain_hours % 6:
        p.error("--rain-hours must be a multiple of 6 in [6, 240]")
    resume = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else None
    if resume is None and out.exists() and any(out.iterdir()):
        p.error(f"{out} exists without last.pt; choose a new --out")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    array_cache = a.array_cache.resolve() if a.array_cache else None
    train, validation = (
        EncoderWindows(root, prep, split, HORIZON, a.cache_months, array_cache, future_rain=True)
        for split in ("train", "validation")
    )
    validation.cache = train.cache
    with np.load(prep / "graph.npz") as graph:
        edges = np.column_stack([graph["source"], graph["target"]])
    catalog = Catalog(a.gauges)
    halo = config["architecture"]["graph_layers"] + decoder["decoder_graph_layers"]
    river = (
        River.load(root, train.ids, edges)
        if config["architecture"].get("direction") == "upstream"
        else None
    )
    steps = a.rain_hours // 6
    if a.score_inside and river is None:
        p.error("--score-inside needs a whole-basin (upstream) model")
    supports = sparse_supports(len(train.ids), edges)
    if a.public:
        if (catalog.source == "tva").any():
            p.error("--public needs a catalog of public series (build-public-gauges)")
        inputs = ForecastInputs(train, catalog, None, edges, supports, halo, river)
    else:
        with tempfile.TemporaryDirectory() as scratch:
            folder = tva.extract(a.tva_archive, Path(scratch))
            inputs = ForecastInputs(train, catalog, folder, edges, supports, halo, river)
    withheld = held_out(root, catalog, a.holdout_fold, a.folds, a.fold_seed)
    inputs.withhold(withheld)
    forecasts = RainForecasts(a.forecast_rain) if a.forecast_rain else None
    usable, has_qpf = issue_lists(inputs, train, steps)
    has_forecast = forecast_issues(train, forecasts)
    split_end = pd.Timestamp(train.manifest["splits"]["train"][1])
    held = np.zeros(len(train.issues), dtype=bool)
    if a.select_from:
        # Targets of training issues end before the held-out period starts.
        split_end = min(split_end, pd.Timestamp(a.select_from))
        held = train.times[train.issues] >= split_end
        usable &= (train.times[train.issues + HORIZON - 1] < split_end)[:, None]
    if not usable.any():
        raise ValueError("No training issue has an observed target")
    pools = []
    for data, mask in ((train, held), (validation, np.ones(len(validation.issues), dtype=bool))):
        if not mask.any():
            continue
        v_usable, v_qpf = issue_lists(inputs, data, steps)
        if a.public:
            ok = mask & v_usable[:, catalog.camelsh].any(axis=1) & forecast_issues(data, forecasts)
        else:
            tva_forecast = np.array([int(i) in inputs.forecast for i in data.issues])
            ok = mask & v_usable[:, inputs.is_tva].any(axis=1) & v_qpf & tva_forecast
        pools += [(data, int(i)) for i in np.flatnonzero(ok)]
    if not pools:
        raise ValueError("No validation issue has a forecast and an observation")
    wanted = a.validation_issues or len(pools)
    spread = np.linspace(0, len(pools) - 1, min(wanted, len(pools)))
    validation_pairs = [pools[k] for k in np.unique(spread.round().astype(int))]
    before = inputs.observed[: int(train.times.searchsorted(split_end))]
    with np.errstate(all="ignore"):
        scale = np.maximum(np.nanstd(before, axis=0), 1e-2)
        high = np.nanquantile(before, 0.9, axis=0)
        flood = np.nanquantile(before, a.flood_quantile, axis=0)
    scale, high = np.nan_to_num(scale, nan=1.0), np.nan_to_num(high, nan=np.inf)
    flood = np.nan_to_num(flood, nan=np.inf)
    # Issue times whose next 10 days reach the gauge's flood threshold (--flood-quantile).
    flooded = None
    if a.flood_fraction:
        peak = (
            pd.DataFrame(inputs.observed[::-1])
            .rolling(HORIZON, min_periods=1)
            .max()
            .to_numpy()[::-1]
        )
        flooded = peak[train.issues] >= flood[None]
    planned_pairs = None
    if a.sample_budget:
        a.updates, planned_pairs = plan_updates(inputs, usable, has_qpf, a, flooded, has_forecast)
    if a.evaluations:
        a.eval_every = max(1, math.ceil(a.updates / a.evaluations))
    if resume is not None and not same_run(resume["finetune"]["arguments"], signature(a)):
        p.error("Saved run used different arguments; use the original command or a new --out")
    torch.manual_seed(a.seed)  # the initialization of --from-scratch
    model = load_forecast(saved, train, prep, supports, device, not a.from_scratch).train()
    head = [q for n, q in model.named_parameters() if not n.startswith("encoder.")]
    groups = [dict(params=head, lr=a.lr)]
    if a.encoder_lr_scale > 0:
        groups.append(dict(params=list(model.encoder.parameters()), lr=a.lr * a.encoder_lr_scale))
    else:
        model.encoder.requires_grad_(False)
    optimizer = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule(a))
    objective = Objective(inputs, scale, high, a, device)
    trainable = [q for q in model.parameters() if q.requires_grad]
    # Moving average of every floating-point weight; what is validated and saved.
    average = (
        {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}
        if a.ema_decay
        else None
    )

    config = dict(config)
    if a.public:
        config["future_rain"] = (
            "fine-tuned on public observed flow: public forecast rain (forecast-rain) for a "
            "forecast-fraction share of samples, realized AORC otherwise, known to rain_hours"
        )
    else:
        config["future_rain"] = (
            "fine-tuned on observed flow: TVA basin forecast rain at TVA gauges in its forecast "
            "era, realized AORC elsewhere, known to rain_hours over a random share of gauge basins"
        )
    config["rain_hours"] = a.rain_hours
    config["history_gauges"] = str(a.gauges)
    config["history_gauge_comids"] = catalog.comid.tolist()
    config["observed_finetune"] = dict(
        arguments=signature(a),
        init_sha256=sha(a.init),
        init_update=saved.get("update"),
        gauges_sha256=sha(a.gauges),
        tva_archive_sha256=sha(a.tva_archive) if a.tva_archive else None,
        tva_data_used=not a.public,
        held_out_gauges=catalog.names[withheld].tolist(),
        gauges=len(catalog),
        tva_evaluation_gauges=int(inputs.is_tva.sum()),
        training_issues=int(usable.any(axis=1).sum()),
        training_issues_with_tva_rain=int((usable.any(axis=1) & has_qpf).sum()),
        training_issues_with_forecast_rain=int((usable.any(axis=1) & has_forecast).sum()),
        planned_pairs=planned_pairs,
        validation_issues=len(validation_pairs),
        loss="lead-weighted normalized log MSE + raw-weight x (raw error / gauge training std)^2 "
        "x (1 + high-flow-weight above gauge training 90th percentile) "
        "x (1 + under-weight where also under-predicted)",
        selection="training.selection (as train-lstm): mean over 6-120 h of median NSE over "
        "CAMELS-H gauges; 00Z validation issues, forecast-rain only"
        if a.public
        else "mean over 6-120 h of median NSE at TVA gauges + csi-weight x mean over "
        "6-120 h of pooled CSI above each gauge's training flood-quantile; 2019, TVA "
        "forecast rain only",
        scored_gauges="every catalog gauge inside the subgraph" if a.score_inside else "drawn",
        rain_error="TVA rain-forecast errors on realized AORC rain (data.rain_error)"
        if a.rain_error
        else "none",
        weights="exponential moving average, decay ema-decay" if a.ema_decay else "raw",
        flood_sampling=f"{a.flood_fraction:.0%} of samples from issues reaching the gauge's training "
        f"{a.flood_quantile:.0%} quantile within 10 days"
        if a.flood_fraction
        else "uniform",
        validation_period=f"{a.select_from} to 2019 (held out of training)"
        if a.select_from
        else "2019",
    )
    state = dict(update=0, best_selection=-np.inf, best_update=None, last_evaluated=None, pairs=0)
    if resume is not None:
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        restore_rng(resume["rng"], device)
        state = resume["state"]
        state.setdefault("pairs", None)  # not counted by older runs
        if average is not None:
            average = {k: v.to(device) for k, v in resume["average"].items()}
        print(f"Resuming at update {state['update']}", flush=True)
    else:
        torch.manual_seed(a.seed)
        atomic_json(out / "run_config.json", config)
        table = pd.read_csv(a.gauges.with_name("gauges.csv"), dtype={"gauge": str})
        table.assign(training_flow_std=scale, training_flow_q90=high, flood=flood).to_csv(
            out / "gauges.csv", index=False
        )
    print(
        f"{len(catalog)} gauges ({int(inputs.is_tva.sum())} with TVA forecasts); "
        f"{config['observed_finetune']['training_issues']:,} training issues "
        f"({config['observed_finetune']['training_issues_with_tva_rain']:,} with TVA rain); "
        f"{len(validation_pairs)} validation issues; rain known to {a.rain_hours} h",
        flush=True,
    )

    def checkpoint(path, full):
        value = dict(
            format="encoder-forecast-v1",
            config=config,
            model=model.state_dict(),
            update=state["update"],
            finetune=dict(arguments=signature(a)),
        )
        if full:
            value.update(
                average=average,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                rng=capture_rng(device),
                state=state,
            )
        save_atomic(value, path)

    def evaluate():
        began = time.perf_counter()
        current = None
        if average is not None:
            current = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(average, strict=False)
        scores = (
            validate_public(model, inputs, validation_pairs, a, device, forecasts)
            if a.public
            else validate(model, inputs, validation_pairs, a, device, flood)
        )
        row = dict(update=state["update"], pairs=state["pairs"], **scores)
        if not np.isfinite(row["selection"]):
            raise ValueError(
                "No validation gauge has --min-count values; raise --validation-issues "
                "or lower --min-count"
            )
        row["best"] = row["selection"] > state["best_selection"]
        if row["best"]:
            state.update(best_selection=row["selection"], best_update=state["update"])
        state["last_evaluated"] = state["update"]
        checkpoint(out / "checkpoints" / f"update_{state['update']:07d}.pt", False)
        if row["best"]:
            checkpoint(out / "best.pt", False)
        if current is not None:
            model.load_state_dict(current)
        progress = out / "validation_progress.csv"
        frame = pd.read_csv(progress) if progress.exists() else pd.DataFrame()
        frame = frame[frame["update"] != row["update"]] if len(frame) else frame
        pd.concat([frame, pd.DataFrame([row])]).to_csv(progress, index=False)
        keys = (
            "update",
            "pairs",
            "selection",
            "tva_selection",
            "nse_selection",
            "csi_selection",
            "tva_csi_selection",
            "persistence_selection",
            "best",
        )
        shown = {
            k: round(row[k], 3) if isinstance(row[k], float) else row[k] for k in keys if k in row
        }
        print(f"validation ({time.perf_counter() - began:.0f} s): {json.dumps(shown)}", flush=True)

    if state["last_evaluated"] is None:
        evaluate()
        checkpoint(last, True)
    log = out / "training_log.jsonl"
    began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool:

        def prepare(update):
            core, batch, scored, pairs = draw(
                update, inputs, usable, has_qpf, a, flooded, has_forecast
            )
            return pool.submit(
                assemble, inputs, train, core, batch, scored, steps, forecasts
            ), pairs

        upcoming, pairs = prepare(state["update"])
        while state["update"] < a.updates:
            waited = time.perf_counter()
            graph, dynamic, rain, target = upcoming.result()
            waited = time.perf_counter() - waited
            consumed = pairs
            if state["update"] + 1 < a.updates:
                upcoming, pairs = prepare(state["update"] + 1)
            supports = [s.to(device) for s in graph["supports"]]
            with torch.autocast(device.type, torch.bfloat16, enabled=a.amp):
                prediction = model(
                    dynamic.to(device), graph["static"].to(device), rain.to(device), supports
                )
            prediction = prediction.float()
            local = torch.as_tensor(graph["scored_local"], device=device)
            loss, values = objective(
                prediction[:, :, local], target.to(device), graph["gauge_index"].to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, a.clip, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            state["update"] += 1
            if state["pairs"] is not None:
                state["pairs"] += consumed
            if average is not None:
                # Shorter memory early on, so the average is not dominated by the start.
                decay = min(a.ema_decay, (1 + state["update"]) / (10 + state["update"]))
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if k in average:
                            average[k].lerp_(v, 1 - decay)
            update = state["update"]
            if update % a.log_every == 0 or update == 1:
                row = dict(
                    update=update,
                    loss=loss.item(),
                    **values,
                    pairs=state["pairs"],
                    nodes=len(graph["nodes"]),
                    gradient_norm=float(norm),
                    lr=optimizer.param_groups[0]["lr"],
                    data_wait_seconds=round(waited, 3),
                    elapsed_minutes=round((time.perf_counter() - began) / 60, 2),
                )
                if device.type == "cuda":
                    row["peak_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 1)
                with open(log, "a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if update % a.eval_every == 0 or update == a.updates:
                evaluate()
                checkpoint(last, True)
    print(
        f"Done. Best update {state['best_update']} (selection {state['best_selection']:.4f}); "
        f"{out / 'best.pt'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
