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
from tva_flood.data.folds import holdout_folds
from tva_flood.training.runtime import save_atomic

from . import paths as P
from .data import timeline_index
from .model import HydroModel
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


def withheld_gauges(fold, folds=5):
    s = samples()
    if not fold:
        return np.zeros(s.gauges.count, bool)
    return holdout_folds(P.ROOT, s.gauges.catalog, folds) == fold


class Schedule(Dataset):
    """Fixed list of (issue step, visibility seed) entries for one rank."""

    def __init__(self, entries, fold, stage):
        self.entries, self.fold, self.stage = entries, fold, stage

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        step, seed = self.entries[i]
        s = samples()
        withheld = withheld_gauges(self.fold)
        visible = None
        if self.stage == "B":
            rng = np.random.default_rng(seed)
            if rng.random() >= 0.15:
                visible = s.eligible(step, withheld) & (rng.random(s.gauges.count) < 0.5)
        return s.build(step, "gefs_reforecast", visible, withheld)


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


def validate(model, gpu, stage, fold, rank, world, arrays):
    s = samples()
    model.eval()
    entries = validation_entries(stage)[rank::world]
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
            for step in entries:
                shown = s.eligible(step, withheld)
                for sums, visible in ((gauged, shown), (hidden, None)):
                    sample = s.build(step, "gefs_reforecast", visible, withheld)
                    b = gpu.batch(sample)
                    out = model(b, full=sums is gauged)
                    observed = b["observed"][LEADS].T
                    valid = torch.isfinite(observed) & use[:, None]
                    sums.add(out["gauge"][:, FUTURE], observed, valid)
                    if sums is gauged:
                        z = b["nwm"][FUTURE][[3, 11, 19]].T
                        target = torch.expm1(z.double() * arrays.std[8] + arrays.mean[8]).clamp_min(0)
                        ok = b["nwm_valid"][FUTURE][[3, 11, 19]].T.bool()
                        reaches.add(out["reach"][:, FUTURE][:, [3, 11, 19]], target, ok)
            for name, sums in (("gauged", gauged), ("hidden", hidden)):
                nse = sums.nse(min_count=30)[:, DAYS_1_5]
                results[name] = float(np.nanmean(np.nanmedian(nse[trained], axis=0)))
            reach_nse = reaches.nse(min_count=30)
            results["reaches"] = float(np.nanmean(np.nanmedian(reach_nse, axis=0)))
            results["score"] = (results["gauged"] + results["hidden"] + results["reaches"]) / 3
    model.train()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["A", "B"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init", type=Path, help="Stage B: stage-A checkpoint")
    p.add_argument("--fold", type=int, default=0, help="Withhold this fold's gauges (1-5)")
    p.add_argument("--updates", type=int, default=6000)
    p.add_argument("--accumulate", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--nwm-weight", type=float, default=0.3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--delta-weight", type=float, default=0.01, help="Observed-head penalty")
    p.add_argument("--validate-every", type=int, default=1000)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
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
    model = HydroModel(s.network.groups, s.network.delay, hidden=a.hidden).to(device)
    if a.init:  # stage A trains runoff and routing only; every head starts fresh
        state = torch.load(a.init, map_location="cpu", weights_only=False)
        stage_a = ("trunk.", "simulation.", "router.")  # what stage A trains; heads start fresh
        shared = {k: v for k, v in state["model"].items() if k.startswith(stage_a)}
        model.load_state_dict(shared, strict=False)
    model.attach(s.network, s.network.edges, s.gauges.reach, device)
    gpu = Device(s.network, s.gauges, arrays, device)
    ddp = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True) if world > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    per_rank = a.updates * a.accumulate
    entries = training_entries(per_rank * world, a.seed + 1000 * a.fold)[rank::world]
    loader = DataLoader(
        Schedule(entries, a.fold, a.stage),
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
    running = np.zeros(4)
    iterator = iter(loader)
    for update in range(1, a.updates + 1):
        lr = a.lr * min(1.0, update / a.warmup) * 0.5 * (1 + math.cos(math.pi * update / a.updates))
        for group in optimizer.param_groups:
            group["lr"] = max(lr, a.lr * 0.01)
        for micro in range(a.accumulate):
            b = gpu.batch(next(iterator))
            sync = micro == a.accumulate - 1 or world == 1
            context = ddp.no_sync() if (world > 1 and not sync) else torch.enable_grad()
            with context, torch.autocast("cuda", torch.bfloat16):
                outputs = ddp(b)
                if a.stage == "A":
                    loss = nwm_loss(outputs["sim"], b, arrays, slice(None))
                    parts = (0.0, loss.item(), 0.0)
                else:
                    # The simulation head follows NWM everywhere. The observed head, which
                    # forecasts ungauged reaches, follows NWM fully outside the basins of
                    # visible gauges and at a tenth inside them, where the gauge should win.
                    inside = 1.0 - 0.9 * b["owned"]
                    everything = slice(None)
                    prior = 0.5 * (
                        nwm_loss(outputs["sim"], b, arrays, everything)
                        + nwm_loss(outputs["observed"], b, arrays, everything, rows=inside)
                    )
                    # Gauges: the forecast (gauge head where visible) and the routed observed
                    # head (the ungauged forecast) at every training gauge.
                    observed, spread = gauge_loss(outputs["gauge"], outputs["sigma"], b, trained)
                    routed, _ = gauge_loss(outputs["routed_gauge"], outputs["sigma"], b, trained)
                    observed = 0.5 * (observed + routed)
                    stay = outputs["delta"].pow(2).mean()
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
                f"update {update}/{a.updates} loss {r[0]:.4f} (obs {r[1]:.4f} nwm {r[2]:.4f} spread {r[3]:.4f}) "
                f"lr {lr:.2e} {minutes:.1f} min",
                flush=True,
            )
        elif update % 50 == 0:
            running[:] = 0
        if update % a.validate_every == 0 or update == a.updates:
            result = validate(model, gpu, a.stage, a.fold, rank, world, arrays)
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
                    update=update,
                    validation=result,
                )
                save_atomic(state, out / "last.pt")
                if result["score"] > best:
                    best = result["score"]
                    save_atomic(state, out / "best.pt")
            if world > 1:
                dist.barrier()
    if rank == 0:
        print(f"TRAINING COMPLETE: {out} (best score {best:.4f})", flush=True)


if __name__ == "__main__":
    main()
