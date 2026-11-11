"""Training on all GPUs (torchrun): stage A (pre-training on NWM) and stage B (forecast).

Every sample is a forecast at a 00Z issue: observed weather (AORC) before the issue and,
after it, only GEFSv12 reforecast rain to 240 h (other future weather unknown). Training
issues are the reforecast days whose targets end before 2019 (2000-2018); selection uses
2019 the same way.

Stage A, pre-training: no gauges. Loss: the simulation head against the NWM v3.0
retrospective at every reach over the scored window (8 days before the issue and the 10
days after), a bounded NSE per reach (raw and log flow). It learns runoff, graph mixing
and routing.

Stage B, forecast (trunk, simulation head and routing from stage A; other heads fresh).
Each sample shows 30 days of flow at a random half of the eligible gauges (none in 15% of
samples); the loss covers every training gauge, shown or not, so one model learns both
the gauged and the ungauged case. Loss = gauge forecasts (gauge head and routed observed
head; leads 1-40, flows above the gauge's 90th percentile x3) + 0.3 x NWM prior over the
scored window (simulation head everywhere; observed head at a tenth inside visible gauges'
basins) + 0.1 x spread + 0.01 x mean squared observed-head correction.
With ``--loss nse-q`` the gauge terms change (stage A, the NWM prior and the selection do not):
loss = NSE* of the forecasts (gauge head and routed observed head; squared error over the squared
(gauge flow std + 0.1 m3/s), Kratzert et al. 2019, no log terms, no flood weight) + the pinball
loss of a tail head (0.5, 0.9 and 0.99 flow quantiles around the detached forecast, same scale)
+ 0.3 x NWM prior [+ delta weight x correction penalty]; no spread term. ``--head lstm`` gives the gauge head
256 units, the LSTM baseline's log-flow scaling and its inputs (30 days of basin-mean weather and the 48
basin-mean attributes at every step), trained jointly with the rest.
A fold (``--fold k``) withholds that fold's gauges from everything: targets, inputs and
model selection.

Selection (stage B): on 2019 issues, the mean of (1) median NSE at training gauges with
their history shown, (2) the same with every history hidden, and (3) median NSE against
the NWM retrospective over all reaches; each averaged over leads 24, 48, 72, 96 and 120 h.
Stage A selects on its NWM loss.

  torchrun --standalone --nproc_per_node 4 -m tva_flood.hydro.train --stage A --out DIR
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from tva_flood.common import atomic_json
from tva_flood.training.runtime import save_atomic

from . import paths as P
from . import splits
from .data import timeline_index
from .model import ABLATIONS, QUANTILES, AttentionGNN, HydroModel, LumpedModel
from .samples import Device, Samples

DAYS_1_5 = [3, 7, 11, 15, 19]  # lead indices of 24, 48, 72, 96, 120 h
FUTURE = slice(P.INNOVATION, P.SCORED)  # leads within the scored window
LEADS = slice(P.HISTORY, P.WINDOW)  # leads within the sample window (observed flow)
SAMPLES = None  # one per worker process


def samples():
    global SAMPLES
    if SAMPLES is None:
        SAMPLES = Samples()
    return SAMPLES


SPLIT = dict(scheme="random", keep=1.0, keep_seed=0)  # set from the arguments (``splits``)
WITHHELD = {}


def withheld_gauges(fold, split=None):
    """Gauges the model for ``fold`` never sees, under ``split`` (default: this run's ``SPLIT``)."""
    split = split or SPLIT
    key = (fold, split["scheme"], split["keep"], split["keep_seed"])
    if key not in WITHHELD:
        s = samples()
        WITHHELD[key] = splits.withheld(s.gauges, s.network, fold, split["scheme"], split["keep"], split["keep_seed"])
    return WITHHELD[key]


class Schedule(Dataset):
    """Fixed list of (issue step, visibility seed) entries for one rank. ``reaches`` > 0 (lumped model): each
    sample also lists that many random reaches plus every gauge reach (``rows``), the reaches trained on."""

    def __init__(self, entries, fold, stage, split, reaches=0):
        self.entries, self.fold, self.stage, self.split, self.reaches = entries, fold, stage, split, reaches

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        step, seed = self.entries[i]
        s = samples()
        withheld = withheld_gauges(self.fold, self.split)
        visible = None
        rng = np.random.default_rng(seed)
        if self.stage == "B":
            if rng.random() >= 0.15:
                visible = s.eligible(step, withheld) & (rng.random(s.gauges.count) < 0.5)
        sample = s.build(step, "gefs_reforecast", visible, withheld)
        if self.reaches:
            pick = rng.choice(s.network.n, self.reaches, replace=False)
            sample["rows"] = np.unique(np.concatenate([pick, s.gauges.reach])).astype(np.int64)
        return sample


def reforecast_issues(first, last):
    """00Z steps in [first, last] with reforecast rain."""
    days = samples().rain.days("gefs_reforecast")
    return [timeline_index(d) for d in sorted(days) if first <= d <= last]


def training_entries(count, seed):
    """Random reforecast issues whose targets end before 2019."""
    rng = np.random.default_rng(seed)
    last = samples().arrays.times[timeline_index(P.TRAIN_UNTIL) - P.HORIZON]
    issues = np.array(reforecast_issues(P.TRAIN_FROM, last))
    return [(int(rng.choice(issues)), int(rng.integers(2**31))) for _ in range(count)]


def validation_entries(stage):
    """2019 reforecast issues: every 8th day (stage A), every day (stage B)."""
    issues = reforecast_issues(pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-21"))
    return issues[::8] if stage == "A" else issues


def bounded_nse(pred, truth, valid, var, eps, weight=None, rows=None):
    """Mean over rows of log(1 + MSE/var) and log(1 + mean squared log error), each halved.

    Per row (reach or gauge) this is about 1 - NSE for a good fit, in flow and in log
    flow, but a row with a huge relative error adds only logarithmically: no group of
    reaches (such as near-dry headwaters) can dominate the gradient. Rows need a quarter
    of their steps valid. ``weight`` [row, step] weights the steps (flood weighting),
    ``rows`` [row] weights the rows in the mean.
    """
    v = valid.float() if weight is None else valid.float() * weight
    n = v.sum(dim=1)
    mse = ((pred - truth) ** 2 * v).sum(dim=1) / n.clamp_min(1e-6)
    log = torch.log(pred + eps[:, None]) - torch.log(truth + eps[:, None])
    msle = (log**2 * v).sum(dim=1) / n.clamp_min(1e-6)
    per_row = 0.5 * torch.log1p(mse / var) + 0.5 * torch.log1p(msle)
    has = valid.sum(dim=1) >= max(1, pred.shape[1] // 4)
    if not has.any():
        return pred.sum() * 0
    if rows is None:
        return per_row[has].mean()
    return (per_row * rows)[has].sum() / rows[has].sum().clamp_min(1e-6)


def nwm_loss(sim, b, arrays, steps, rows=None):
    """A head [n, S] against the NWM retrospective over ``steps``, reach by reach."""
    z = b["nwm"][steps].T
    valid = b["nwm_valid"][steps].T.bool()
    target = torch.expm1(z.double() * arrays.std[8] + arrays.mean[8]).clamp_min(0).float()
    var = b["std_raw"] ** 2 + 1e-4
    return bounded_nse(sim[:, steps], target, valid, var, b["eps"], rows=rows)


def gauge_loss(gauge, sigma, b, use):
    """Forecast at gauges ``use`` [G] over the leads, gauge by gauge, flows above each gauge's
    training 90th percentile weighted x3 (floods); and the spread NLL."""
    observed = b["observed"][LEADS].T  # [G, lead]
    valid = torch.isfinite(observed) & use[:, None]
    truth = torch.nan_to_num(observed)
    pred = gauge[:, FUTURE]
    flood = 1.0 + 2.0 * (truth > b["gauge_p90"][:, None]).float()
    loss = bounded_nse(pred, truth, valid, b["gauge_std_raw"] ** 2 + 1e-4, b["gauge_eps"], weight=flood)
    mu = torch.log(pred.detach() + 0.01)
    y = torch.log(truth + 0.01)
    nll = torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2
    spread = (nll * valid).sum() / valid.sum().clamp_min(1)
    return loss, spread


def nse_star(pred, truth, valid, std, eps=0.1):
    """Mean over gauges of the mean squared error scaled by (std + eps)^2: the NSE* loss of
    Kratzert et al. (2019). Gauges need a quarter of their steps valid."""
    n = valid.float().sum(dim=1)
    mse = (((pred.float() - truth) ** 2) * valid).sum(dim=1) / n.clamp_min(1)
    per_row = mse / (std + eps) ** 2
    has = n >= max(1, pred.shape[1] // 4)
    return per_row[has].mean() if has.any() else pred.sum() * 0


def pinball(quantile, truth, valid, std, eps=0.1):
    """Mean pinball loss of the flow quantiles [gauge, step, Q] (``model.QUANTILES``), scaled as NSE*."""
    q = torch.tensor(QUANTILES, device=truth.device)
    miss = truth[..., None] - quantile.float()
    loss = torch.maximum(q * miss, (q - 1) * miss).mean(dim=-1) / (std + eps)[:, None]
    n = valid.float().sum(dim=1)
    per_row = (loss * valid).sum(dim=1) / n.clamp_min(1)
    has = n >= max(1, truth.shape[1] // 4)
    return per_row[has].mean() if has.any() else quantile.sum() * 0


def tail_loss(out, b, use):
    """NSE* of the forecasts at gauges ``use`` [G] (gauge head and routed observed head) and the pinball
    loss of the tail head."""
    observed = b["observed"][LEADS].T
    valid = torch.isfinite(observed) & use[:, None]
    truth = torch.nan_to_num(observed)
    std = b["gauge_std_raw"]
    forecast = 0.5 * (nse_star(out["gauge"][:, FUTURE], truth, valid, std) + nse_star(out["routed_gauge"][:, FUTURE], truth, valid, std))
    return forecast, pinball(out["quantile"], truth, valid, std)


class Sums:
    """Additive NSE statistics [rows, lead] (count, sum o, sum o2, sum e2)."""

    def __init__(self, rows, leads, device):
        self.s = torch.zeros(4, rows, leads, dtype=torch.float64, device=device)

    def add(self, pred, truth, valid):
        v = valid.double()
        t = torch.nan_to_num(truth).double()
        e = (pred.double() - t) * v
        self.s += torch.stack([v, t * v, t * t * v, e * e])

    def nse(self, min_count):
        if dist.is_initialized():
            dist.all_reduce(self.s)
        n, so, soo, se = self.s
        var = soo / n.clamp_min(1) - (so / n.clamp_min(1)) ** 2
        ok = (n >= min_count) & (var > 1e-6)
        return torch.where(ok, 1 - se / (n * var).clamp_min(1e-12), torch.nan).cpu().numpy()


def selection_score(results, rule):
    """Checkpoint score: "all" (the default, as every earlier model) is the mean of gauged, hidden and the NWM
    every-reach score; "gauges" leaves out the NWM score, so models trained without NWM select like the others."""
    parts = ("gauged", "hidden", "reaches") if rule == "all" else ("gauged", "hidden")
    return float(np.mean([results[k] for k in parts]))


def validate(model, gpu, stage, fold, rank, world, arrays, rule="all", limit=0):
    s = samples()
    model.eval()
    entries = validation_entries(stage)
    entries = (entries[:: max(1, len(entries) // limit)][:limit] if limit else entries)[rank::world]
    withheld = withheld_gauges(fold)
    trained = s.gauges.trained & ~withheld
    results = {}
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        if stage == "A":
            total = torch.zeros(2, dtype=torch.float64, device=gpu.device)
            for step in entries:
                b = gpu.batch(s.build(step, "gefs_reforecast"))
                out = model(b)
                total[0] += nwm_loss(out["sim"], b, arrays, slice(None)).double()
                total[1] += 1
            if dist.is_initialized():
                dist.all_reduce(total)
            results["nwm_loss"] = float(total[0] / total[1])
            results["score"] = -results["nwm_loss"]
        else:
            gauged = Sums(s.gauges.count, P.HORIZON, gpu.device)
            hidden = Sums(s.gauges.count, P.HORIZON, gpu.device)
            reaches = Sums(s.network.n, 3, gpu.device)
            use = torch.as_tensor(trained, device=gpu.device)
            below = torch.zeros(len(QUANTILES) + 1, dtype=torch.float64, device=gpu.device)  # share of truth under each quantile
            for step in entries:
                shown = s.eligible(step, withheld)
                for sums, visible in ((gauged, shown), (hidden, None)):
                    sample = s.build(step, "gefs_reforecast", visible, withheld)
                    b = gpu.batch(sample)
                    out = model(b, full=sums is gauged)
                    observed = b["observed"][LEADS].T
                    valid = torch.isfinite(observed) & use[:, None]
                    sums.add(out["gauge"][:, FUTURE], observed, valid)
                    if sums is gauged and "quantile" in out:
                        under = (torch.nan_to_num(observed)[..., None] <= out["quantile"].float()) & valid[..., None]
                        below += torch.cat([under.sum((0, 1)), valid.sum()[None]]).double()
                    if sums is gauged:
                        z = b["nwm"][FUTURE][[3, 11, 19]].T
                        target = torch.expm1(z.double() * arrays.std[8] + arrays.mean[8]).clamp_min(0)
                        ok = b["nwm_valid"][FUTURE][[3, 11, 19]].T.bool()
                        reaches.add(out["reach"][:, FUTURE][:, [3, 11, 19]], target, ok)
            for name, sums in (("gauged", gauged), ("hidden", hidden)):
                nse = sums.nse(min_count=2 if limit else 30)[:, DAYS_1_5]
                results[name] = float(np.nanmean(np.nanmedian(nse[trained], axis=0)))
            reach_nse = reaches.nse(min_count=2 if limit else 30)
            results["reaches"] = float(np.nanmean(np.nanmedian(reach_nse, axis=0)))
            results["score_all"] = selection_score(results, "all")
            results["score_gauges"] = selection_score(results, "gauges")
            results["score"] = selection_score(results, rule)
            if below[-1] > 0 or dist.is_initialized():
                if dist.is_initialized():
                    dist.all_reduce(below)
                if below[-1] > 0:
                    results["below_quantile"] = [round(float(x), 4) for x in below[:-1] / below[-1]]  # nominal: QUANTILES
    model.train()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["A", "B"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init", type=Path, help="Stage B: stage-A checkpoint")
    p.add_argument("--whole", action="store_true", help="Load every part of --init (continue a stage-B model)")
    p.add_argument("--fold", type=int, default=0, help="Withhold this fold's gauges (1-5)")
    p.add_argument("--updates", type=int, default=6000)
    p.add_argument("--accumulate", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--nwm-weight", type=float, default=0.3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--delta-weight", type=float, default=0.01, help="Observed-head penalty")
    p.add_argument("--loss", choices=["bounded", "nse-q"], default="bounded", help="Stage B gauge loss (see the module docstring)")
    p.add_argument("--tail-weight", type=float, default=1.0, help="nse-q: weight of the pinball loss")
    p.add_argument("--trunk", choices=["lstm", "attention"], default="lstm", help="Per-reach temporal model: an LSTM, or causal self-attention (AttentionGNN)")
    p.add_argument("--head", choices=["small", "lstm"], default="small", help="Gauge head: 64 units, log1p/5 inputs; or 256 units with the LSTM's scaling and inputs")
    p.add_argument("--validate-every", type=int, default=1000)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    # Attribution experiments (docs/ATTRIBUTION.md); the defaults reproduce the current model.
    p.add_argument("--off", nargs="*", default=[], choices=ABLATIONS, help="Parts of the river-network design switched off")
    p.add_argument("--lumped", action="store_true", help="Non-graph control: every reach a lumped basin (model.LumpedModel)")
    p.add_argument("--lumped-reaches", type=int, default=0, help="Lumped model: train on this many random reaches per sample plus the gauges (0: all)")
    p.add_argument("--no-nwm", action="store_true", help="No NWM anywhere: needs no --init, sets --nwm-weight 0 and --select gauges")
    p.add_argument("--select", choices=["all", "gauges"], default="all", help="Checkpoint score (train.selection_score)")
    p.add_argument("--scheme", choices=splits.SCHEMES, default="random", help="Held-out scheme of --fold (hydro.splits)")
    p.add_argument("--keep", type=float, default=1.0, help="Share of the training gauges kept (gauge coverage)")
    p.add_argument("--keep-seed", type=int, default=0, help="Draw of the kept gauges")
    p.add_argument("--validation-limit", type=int, default=0, help="Validate on this many 2019 issues only (testing)")
    p.add_argument("--resume", action="store_true", help="Continue from OUT/resume.pt if it exists (a cut-off run); otherwise start fresh")
    p.add_argument("--checkpoint-every", type=int, default=100, help="Save OUT/resume.pt (model, optimizer, progress) every this many updates")
    p.add_argument("--stop-after", type=int, default=0, help="Exit with code 2 after this update (testing the resume)")
    a = p.parse_args()
    if a.no_nwm:
        if a.init:
            p.error("--no-nwm trains without stage A: drop --init")
        if a.stage != "B":
            p.error("--no-nwm is a stage-B setting")
        a.nwm_weight, a.select = 0.0, "gauges"
    if a.lumped and (a.off or a.trunk != "lstm"):
        p.error("--lumped has no graph parts to switch off and uses the LSTM trunk")
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60), device_id=device)
    try:
        train(a, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def train(a, rank, world, device):
    torch.manual_seed(a.seed + rank)
    out = a.out.resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / "arguments.json", {k: str(v) for k, v in vars(a).items()})
    s = samples()
    arrays = s.arrays
    SPLIT.update(scheme=a.scheme, keep=a.keep, keep_seed=a.keep_seed)
    if a.fold and not 1 <= a.fold <= splits.fold_count(a.scheme, s.gauges, s.network):
        raise ValueError(f"--fold {a.fold} is not a fold of the {a.scheme} scheme")
    head = None
    if a.head == "lstm":  # 256 units, the LSTM baseline's log-flow scaling and its inputs (basin weather and attributes)
        head = dict(hidden=256, scale="z", lstm_inputs=True, flow_mean=float(arrays.mean[8]), flow_std=float(arrays.std[8]))
    common = dict(hidden=a.hidden, quantiles=a.loss == "nse-q", head=head)
    if a.lumped:
        model = LumpedModel(s.network.groups, s.network.delay, **common)
    elif a.trunk == "attention":
        model = AttentionGNN(s.network.groups, s.network.delay, **common, compile=True)
    else:
        model = HydroModel(s.network.groups, s.network.delay, **common, off=a.off)
    model = model.to(device)
    if a.init:  # stage A trains runoff and routing only; every head starts fresh
        state = torch.load(a.init, map_location="cpu", weights_only=False)
        if bool(state["config"].get("kind") == "lumped") != a.lumped:
            raise ValueError(f"{a.init} is {'not ' if a.lumped else ''}a lumped model")
        if a.whole:
            model.load_state_dict(state["model"])
        else:
            stage_a = ("trunk.", "simulation.", "router.")  # what stage A trains; heads start fresh
            if "routing" in a.off:
                stage_a = ("trunk.", "simulation.")  # the fixed kernels stay fixed
            model.load_state_dict({k: v for k, v in state["model"].items() if k.startswith(stage_a)}, strict=False)
    model.attach(s.network, s.network.edges, s.gauges.reach, device)
    if rank == 0:
        print(f"model: {type(model).__name__} {model.config.get('off', [])}, {model.parameter_count():,} trainable parameters; "
              f"split {splits.describe(SPLIT)}, fold {a.fold}, {int(withheld_gauges(a.fold).sum())} gauges withheld", flush=True)
    gpu = Device(s.network, s.gauges, arrays, device)
    ddp = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True) if world > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    per_rank = a.updates * a.accumulate
    entries = training_entries(per_rank * world, a.seed + 1000 * a.fold)[rank::world]
    loader = DataLoader(
        Schedule(entries, a.fold, a.stage, dict(SPLIT), a.lumped_reaches if a.lumped else 0),
        batch_size=None,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    withheld = torch.as_tensor(withheld_gauges(a.fold), device=device)
    trained = torch.as_tensor(s.gauges.trained, device=device) & ~withheld
    best, began, log = -math.inf, time.perf_counter(), []
    first = 1
    resume_file = out / "resume.pt"
    if a.resume and resume_file.exists():
        saved = torch.load(resume_file, map_location="cpu", weights_only=False)
        if saved["updates"] != a.updates or saved["accumulate"] != a.accumulate or saved["seed"] != a.seed:
            raise ValueError(f"{resume_file} belongs to another run (updates, accumulate or seed differ)")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        best, log, first = saved["best"], saved["log"], saved["update"] + 1
        began = time.perf_counter() - (log[-1]["minutes"] * 60 if log else 0)  # minutes continue from the first part
        loader = DataLoader(
            Schedule(entries[saved["update"] * a.accumulate :], a.fold, a.stage, dict(SPLIT), a.lumped_reaches if a.lumped else 0),
            batch_size=None, shuffle=False, num_workers=a.workers, pin_memory=True, prefetch_factor=2, persistent_workers=True,
        )
        if rank == 0:
            print(f"resumed at update {saved['update']} of {a.updates} (best score {best:.4f})", flush=True)
    running = np.zeros(4)
    iterator = iter(loader)

    def save_resume(update):  # rank 0 only
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), update=update, best=best, log=log,
                     updates=a.updates, accumulate=a.accumulate, seed=a.seed)
        save_atomic(state, resume_file)

    for update in range(first, a.updates + 1):
        lr = a.lr * min(1.0, update / a.warmup) * 0.5 * (1 + math.cos(math.pi * update / a.updates))
        for group in optimizer.param_groups:
            group["lr"] = max(lr, a.lr * 0.01)
        for micro in range(a.accumulate):
            b = gpu.batch(next(iterator))
            sync = micro == a.accumulate - 1 or world == 1
            context = ddp.no_sync() if (world > 1 and not sync) else torch.enable_grad()
            with context, torch.autocast("cuda", torch.bfloat16):
                outputs = ddp(b)
                rows = outputs.get("rows")  # lumped model on a reach subset: 0/1 weight of the reaches it ran on
                if a.stage == "A":
                    loss = nwm_loss(outputs["sim"], b, arrays, slice(None), rows=rows)
                    parts = (0.0, loss.item(), 0.0)
                else:
                    # The simulation head follows NWM everywhere. The observed head, which
                    # forecasts ungauged reaches, follows NWM fully outside the basins of
                    # visible gauges and at a tenth inside them, where the gauge should win.
                    inside = 1.0 - 0.9 * b["owned"]
                    everything = slice(None)
                    if a.nwm_weight > 0:
                        prior = 0.5 * (
                            nwm_loss(outputs["sim"], b, arrays, everything, rows=rows)
                            + nwm_loss(outputs["observed"], b, arrays, everything, rows=inside if rows is None else inside * rows)
                        )
                    else:  # no NWM: not even computed, so nothing of it reaches the gradient
                        prior = outputs["sim"].sum() * 0
                    # Gauges: the forecast (gauge head where visible) and the routed observed
                    # head (the ungauged forecast) at every training gauge.
                    stay = outputs["delta"].pow(2).mean()
                    if a.loss == "nse-q":
                        observed, tail = tail_loss(outputs, b, trained)
                        loss = observed + a.nwm_weight * prior + a.tail_weight * tail + a.delta_weight * stay
                        parts = (observed.item(), prior.item(), tail.item())
                    else:
                        observed, spread = gauge_loss(outputs["gauge"], outputs["sigma"], b, trained)
                        routed, _ = gauge_loss(outputs["routed_gauge"], outputs["sigma"], b, trained)
                        observed = 0.5 * (observed + routed)
                        loss = observed + a.nwm_weight * prior + 0.1 * spread + a.delta_weight * stay
                        parts = (observed.item(), prior.item(), spread.item())
                (loss / a.accumulate).backward()
            running += np.array([loss.item(), *parts])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if update % 50 == 0 and rank == 0:
            r = running / (50 * a.accumulate)
            running[:] = 0
            minutes = (time.perf_counter() - began) / 60
            print(
                f"update {update}/{a.updates} loss {r[0]:.4f} (obs {r[1]:.4f} nwm {r[2]:.4f} {'tail' if a.loss == 'nse-q' else 'spread'} {r[3]:.4f}) "
                f"lr {lr:.2e} {minutes:.1f} min",
                flush=True,
            )
        elif update % 50 == 0:
            running[:] = 0
        if update % a.validate_every == 0 or update == a.updates:
            result = validate(model, gpu, a.stage, a.fold, rank, world, arrays, a.select, a.validation_limit)
            if rank == 0:
                result.update(update=update, minutes=(time.perf_counter() - began) / 60)
                log.append(result)
                atomic_json(out / "validation.json", log)
                print("validation " + json.dumps(result), flush=True)
                state = dict(
                    model=model.state_dict(),
                    config=model.config,
                    stage=a.stage,
                    fold=a.fold,
                    split=dict(SPLIT),
                    nwm=a.nwm_weight > 0 or bool(a.init),
                    select=a.select,
                    update=update,
                    validation=result,
                    parameters=model.parameter_count(),
                    gpus=world,
                )
                save_atomic(state, out / "last.pt")
                save_atomic(state, out / f"checkpoint_{update:05d}.pt")  # every validation: selection can be redone
                if result["score"] > best:
                    best = result["score"]
                    save_atomic(state, out / "best.pt")
            if world > 1:
                dist.barrier()
        if update % a.checkpoint_every == 0 and update < a.updates and rank == 0:
            save_resume(update)
        elif update % a.validate_every == 0 and rank == 0:
            save_resume(update)
        if a.stop_after and update == a.stop_after:
            if rank == 0:
                save_resume(update)
                print(f"stopped after update {update} (--stop-after)", flush=True)
            sys.exit(2)
    if rank == 0:
        print(f"TRAINING COMPLETE: {out} (best score {best:.4f})", flush=True)


if __name__ == "__main__":
    main()
