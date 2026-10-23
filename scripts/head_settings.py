"""Head-settings test: does the gauge head's size, input scaling, inputs or loss explain the day-1 gap to the LSTM?

The trunk, routing and everything upstream of the gauge head are frozen (a stage-B model). What the head reads is
cached once for every training and selection issue (GEFSv12 reforecast days 2000-2019, every eligible gauge visible, the
gauged test's view); heads are then trained from scratch on identical (issue, gauge) pairs, differing in one setting
at a time, and scored on 2019 at the training gauges (median NSE by lead; selection = days 1-5).

  torchrun --standalone --nproc_per_node 4 scripts/head_settings.py cache --model models/gefs_nseq_B --out models/head_settings
  python scripts/head_settings.py train --cache models/head_settings/cache.npz --variant V0 --gpu 0
  python scripts/head_settings.py check --cache models/head_settings/cache.npz --model models/gefs_nseq_B

Variants (each adds to the one before unless noted):
  V0  the stage-B head (64 units, log1p/5 inputs, routed x exp(correction)), NSE* loss, trained from scratch
  V1  256 units (the LSTM's size)
  V2  V1 with the LSTM's input scaling: (log1p(flow) - mean) / std of log flow
  V3  V2 with the LSTM's inputs: 30 days of basin-mean weather (16 channels) and the 48 basin-mean attributes at every step
  V4  V3 with the LSTM's loss: lead-weighted log-flow error + raw error scaled by the gauge's spread, x3 above its 90th percentile
  V5  V0 with the LSTM's loss only
"""

import argparse
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn

from tva_flood.common import atomic_json
from tva_flood.hydro import paths as P
from tva_flood.hydro.evaluate import load_model
from tva_flood.hydro.model import EPS, SparseProduct, mlp
from tva_flood.hydro.samples import Device, Samples
from tva_flood.hydro.train import DAYS_1_5, FUTURE, nse_star, reforecast_issues, withheld_gauges
from tva_flood.training.forecast import lead_weights

HISTORY = 120  # steps of observed flow and weather the head reads (30 days)
LABELS = ("12 h", "24 h", "48 h", "72 h", "96 h", "120 h")  # lead indices 1, 3, 7, 11, 15, 19
VARIANTS = {
    "V0": dict(hidden=64, scale="log5", lstm_inputs=False, loss="nse-star"),
    "V1": dict(hidden=256, scale="log5", lstm_inputs=False, loss="nse-star"),
    "V2": dict(hidden=256, scale="z", lstm_inputs=False, loss="nse-star"),
    "V3": dict(hidden=256, scale="z", lstm_inputs=True, loss="nse-star"),
    "V4": dict(hidden=256, scale="z", lstm_inputs=True, loss="lstm"),
    "V5": dict(hidden=64, scale="log5", lstm_inputs=False, loss="lstm"),
}


class ProbeHead(nn.Module):
    """``hydro.model.GaugeHead`` with settings. With the defaults it is the same network (same parameter names): routed flow x
    exp(correction), the correction a share of the latest misfit fading with lead plus a decoder output."""

    def __init__(self, hidden=64, scale="log5", lstm_inputs=False, mean=0.0, std=1.0):
        super().__init__()
        self.scale, self.lstm_inputs, self.mean, self.std = scale, lstm_inputs, mean, std
        extra = (16 + 48) if lstm_inputs else 0  # weather and basin attributes, at every encoder step; attributes at every decoder step
        self.describe = mlp(48 + 2, hidden, hidden)
        self.encode = nn.LSTM(4 + extra, hidden, batch_first=True)
        self.decode = nn.LSTM(4 + 16 + (48 if lstm_inputs else 0), hidden, batch_first=True)
        self.lead = nn.Embedding(P.HORIZON, 16)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.keep = nn.Parameter(3.0 - 0.25 * torch.arange(P.HORIZON, dtype=torch.float32))

    def flow(self, x):
        z = torch.log1p(x.clamp_min(0))
        return z / 5 if self.scale == "log5" else (z - self.mean) / self.std

    @torch.autocast("cuda", enabled=False)
    def forward(self, history, routed, simulated, rain, describe, weather=None, basin=None):
        """history [g, 120] observed flow (NaN missing); routed, simulated [g, S]; rain [g, lead, 2]; describe [g, 50];
        weather [g, 120, 16] and basin attributes [g, 48] with ``lstm_inputs``. Returns the forecast [g, lead]."""
        g, before = history.shape[0], P.INNOVATION
        valid = torch.isfinite(history)
        obs = torch.where(valid, self.flow(torch.nan_to_num(history)), 0.0)
        tail = torch.zeros(g, history.shape[1], 2, device=history.device)
        tail[:, -before:, 0] = self.flow(routed[:, :before])
        tail[:, -before:, 1] = self.flow(simulated[:, :before])
        steps = [obs[..., None], valid[..., None].float(), tail]
        if self.lstm_inputs:
            steps += [weather.float(), basin.float()[:, None].expand(-1, history.shape[1], -1)]
        h0 = self.describe(describe.float())[None].contiguous()
        _, (h, c) = self.encode(torch.cat(steps, dim=-1), (h0, torch.zeros_like(h0)))
        future = [self.flow(routed[:, before:])[..., None], self.flow(simulated[:, before:])[..., None], rain.float(),
                  self.lead.weight[None].expand(g, -1, -1)]
        if self.lstm_inputs:
            future.append(basin.float()[:, None].expand(-1, P.HORIZON, -1))
        decoded, _ = self.decode(torch.cat(future, dim=-1), (h, c))
        learned = 3 * torch.tanh(self.out(decoded)[..., 0])
        misfit = torch.log(torch.nan_to_num(history[:, -1]).clamp_min(0) + EPS) - torch.log(routed[:, before - 1] + EPS)
        misfit = torch.where(valid[:, -1], misfit.clamp(-5, 5), torch.zeros_like(misfit))
        return routed[:, before:] * torch.exp(torch.sigmoid(self.keep)[None] * misfit[:, None] + learned)


# ---------------------------------------------------------------- cache
def cache(a):
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=120), device_id=device)
    s = Samples()
    model, state = load_model(a.model / "best.pt", s, device)
    gpu = Device(s.network, s.gauges, s.arrays, device)
    withheld = withheld_gauges(state["fold"])
    last = s.arrays.times[s.arrays.steps - P.HORIZON - 1]
    issues = reforecast_issues(P.TRAIN_FROM, min(last, pd.Timestamp("2019-12-21")))[:: a.every]
    mine = issues[rank::world]
    obs = s.gauges.observed
    parts = {k: [] for k in ("step", "routed", "sim", "rain", "weather", "stage_b", "history", "target")}
    began = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for position, step in enumerate(mine):
            b = gpu.batch(s.build(step, "gefs_reforecast", s.eligible(step, withheld), withheld))
            out = model(b)
            values = dict(
                step=step,
                routed=out["routed_gauge"],
                sim=out["sim"][b["gauge_reach"]],
                rain=model.gauge_rain(b),
                weather=model.gauge_weather(b),
                stage_b=out["gauge"][:, FUTURE],
            )
            for k, v in values.items():
                parts[k].append(np.asarray(v) if k == "step" else v.float().cpu().numpy().astype(np.float16 if k == "weather" else np.float32))
            parts["history"].append(obs[step - HISTORY : step].T.astype(np.float32))
            parts["target"].append(obs[step : step + P.HORIZON].T.astype(np.float32))
            if rank == 0 and position % 200 == 0:
                print(f"cache {position + 1}/{len(mine)} {(time.perf_counter() - began) / 60:.1f} min", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / f".cache_{rank}.npz", **{k: np.stack(v) for k, v in parts.items()})
    if world > 1:
        dist.barrier()
    if rank == 0:
        loaded = [np.load(a.out / f".cache_{r}.npz") for r in range(world)]
        merged = {k: np.concatenate([p[k] for p in loaded]) for k in parts}
        order = np.argsort(merged["step"])
        fixed = dict(
            describe=model.gauge_describe(gpu.fixed).float().cpu().numpy(),
            basin=SparseProduct.apply(*model.basin_mean, gpu.fixed["static"]).float().cpu().numpy(),
            std=s.gauges.std_raw, p90=s.gauges.p90, trained=s.gauges.trained & ~withheld,
            flow_mean=float(s.arrays.mean[8]), flow_std=float(s.arrays.std[8]),
            times=s.arrays.times[merged["step"][order]].to_numpy(),
        )
        np.savez(a.out / "cache.npz", **{k: v[order] for k, v in merged.items()}, **fixed)
        for r in range(world):
            (a.out / f".cache_{r}.npz").unlink()
        print(f"CACHE COMPLETE: {len(order)} issues -> {a.out / 'cache.npz'}", flush=True)
    if world > 1:
        dist.destroy_process_group()


# ---------------------------------------------------------------- losses and scoring
def lstm_loss(pred, truth, valid, scale, high, mean, std, weights):
    """The LSTM baseline's loss: lead-weighted log-flow error (normalized) + raw-flow error / gauge spread, x3 above p90."""
    w = weights[None] * valid
    total = w.sum().clamp_min(1)
    z = (torch.log1p(pred.clamp_min(0)) - mean) / std
    z_truth = (torch.log1p(truth.clamp_min(0)) - mean) / std
    log_term = (w * (z - z_truth) ** 2).sum() / total
    emphasis = 1 + 2.0 * (truth > high[:, None]).float()
    raw_term = (w * emphasis * ((pred - truth) / scale[:, None]) ** 2).sum() / total
    return log_term + raw_term


class Cache:
    def __init__(self, path, device):
        c = np.load(path)
        self.device = device
        self.time = pd.DatetimeIndex(c["times"])
        self.sel = np.flatnonzero(self.time >= P.TRAIN_UNTIL)
        self.fit = np.flatnonzero(self.time < P.TRAIN_UNTIL)
        self.t = {k: torch.as_tensor(c[k], device=device) for k in ("routed", "sim", "rain", "weather", "stage_b", "history", "target")}
        self.describe, self.basin = (torch.as_tensor(c[k], device=device) for k in ("describe", "basin"))
        self.std = torch.as_tensor(c["std"], device=device)
        self.scale = self.std.clamp_min(1e-2)
        self.high = torch.as_tensor(np.nan_to_num(c["p90"], nan=np.inf), device=device)
        self.trained = c["trained"]
        self.flow_mean, self.flow_std = float(c["flow_mean"]), float(c["flow_std"])
        target, history = c["target"], c["history"]
        self.latest = np.isfinite(history[..., -1])  # [issue, gauge]: a latest observation
        ok = self.trained[None] & self.latest & np.isfinite(target).any(-1)
        self.pairs = np.argwhere(ok[self.fit]).astype(np.int64)
        self.pairs[:, 0] = self.fit[self.pairs[:, 0]]

    def gather(self, k, g):
        t = self.t
        return {name: t[name][k, g] for name in ("routed", "sim", "rain", "weather", "stage_b", "history", "target")}


def run(head, cache, k, g):
    x = cache.gather(k, g)
    kwargs = dict(weather=x["weather"], basin=cache.basin[g]) if head.lstm_inputs else {}
    return head(x["history"], x["routed"], x["sim"], x["rain"], cache.describe[g], **kwargs), x


@torch.no_grad()
def score(head, cache, source="head"):
    """Median NSE over training gauges by lead on the selection issues: the head's forecast, the stage-B head's, or the routed flow."""
    sums = torch.zeros(4, len(cache.std), P.HORIZON, dtype=torch.float64, device=cache.device)
    head.eval()
    for k in cache.sel:
        g = torch.as_tensor(np.flatnonzero(cache.trained & cache.latest[k]), device=cache.device)
        if not len(g):
            continue
        kk = torch.full_like(g, int(k))
        if source == "head":
            pred, x = run(head, cache, kk, g)
        else:
            x = cache.gather(kk, g)
            pred = x["stage_b"] if source == "stage_b" else x["routed"][:, P.INNOVATION :]
        truth = x["target"]
        v = torch.isfinite(truth).double()
        t = torch.nan_to_num(truth).double()
        sums[:, g] += torch.stack([v, t * v, t * t * v, (pred.double() - t) ** 2 * v])
    n, so, soo, se = sums
    var = soo / n.clamp_min(1) - (so / n.clamp_min(1)) ** 2
    nse = torch.where((n >= 30) & (var > 1e-6), 1 - se / (n * var).clamp_min(1e-12), torch.nan).cpu().numpy()[cache.trained]
    med = np.nanmedian(nse, axis=0)
    head.train()
    return dict(days_1_5=float(np.mean(med[DAYS_1_5])), **{"6 h": float(med[0])}, **{lab: float(med[i]) for lab, i in zip(LABELS, (1, 3, 7, 11, 15, 19))})


def train(a):
    device = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(device)
    cfg = VARIANTS[a.variant]
    cache_ = Cache(a.cache, device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    head = ProbeHead(cfg["hidden"], cfg["scale"], cfg["lstm_inputs"], cache_.flow_mean, cache_.flow_std).to(device)
    weights = torch.as_tensor(lead_weights("balanced"), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    out = a.out / a.variant
    out.mkdir(parents=True, exist_ok=True)
    print(f"{a.variant} {cfg}: {len(cache_.pairs):,} training pairs, {len(cache_.sel)} selection issues, "
          f"{sum(p.numel() for p in head.parameters()):,} parameters", flush=True)
    refs = {name: score(head, cache_, name) for name in ("stage_b", "routed")}
    print("reference, stage-B head:", {k: round(v, 3) for k, v in refs["stage_b"].items()}, flush=True)
    print("reference, routed flow: ", {k: round(v, 3) for k, v in refs["routed"].items()}, flush=True)
    best, log, began = -math.inf, [], time.perf_counter()
    for update in range(1, a.updates + 1):
        lr = a.lr * min(1.0, update / 200) * 0.5 * (1 + math.cos(math.pi * update / a.updates))
        for group in optimizer.param_groups:
            group["lr"] = max(lr, a.lr * 0.01)
        k, g = (torch.as_tensor(x, device=device) for x in cache_.pairs[rng.integers(0, len(cache_.pairs), a.batch)].T)
        pred, x = run(head, cache_, k, g)
        truth = x["target"]
        valid = torch.isfinite(truth)
        truth = torch.nan_to_num(truth)
        if cfg["loss"] == "nse-star":
            loss = nse_star(pred, truth, valid, cache_.std[g])
        else:
            loss = lstm_loss(pred, truth, valid.float(), cache_.scale[g], cache_.high[g], cache_.flow_mean, cache_.flow_std, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if update % a.validate_every == 0 or update == a.updates:
            result = score(head, cache_)
            result.update(update=update, loss=float(loss), minutes=(time.perf_counter() - began) / 60)
            log.append(result)
            print(f"update {update}/{a.updates} loss {float(loss):.4f} days 1-5 {result['days_1_5']:.4f}  6 h {result['6 h']:.3f}  "
                  + "  ".join(f"{lab} {result[lab]:.3f}" for lab in LABELS), flush=True)
            if result["days_1_5"] > best:
                best = result["days_1_5"]
                torch.save(head.state_dict(), out / "best.pt")
    atomic_json(out / "log.json", dict(variant=a.variant, config=cfg, references=refs, log=log, best=max(log, key=lambda r: r["days_1_5"])))
    print(f"VARIANT {a.variant} DONE best days 1-5 {best:.4f}", flush=True)


def check(a):
    """The cache and the head replica are right if ProbeHead with the stage-B head's weights reproduces its forecasts."""
    device = torch.device("cuda:0")
    s = Samples()
    model, _ = load_model(a.model / "best.pt", s, device)
    cache_ = Cache(a.cache, device)
    head = ProbeHead().to(device)
    head.load_state_dict(model.head.state_dict())
    k = torch.as_tensor(cache_.sel[:: max(1, len(cache_.sel) // 20)], device=device)
    worst, used = 0.0, 0
    for kk in k:
        g = torch.as_tensor(np.flatnonzero(cache_.trained), device=device)
        pred, x = run(head, cache_, torch.full_like(g, int(kk)), g)
        applied = (x["stage_b"] != x["routed"][:, P.INNOVATION :]).any(dim=1) & torch.isfinite(x["history"][:, -1])
        if applied.any():
            ref = x["stage_b"][applied]
            worst = max(worst, float(((pred[applied] - ref).abs() / (ref.abs() + 1)).max()))
            used += int(applied.sum())
    print(f"replica vs stage-B head on {used} (issue, gauge) pairs: worst relative difference {worst:.2e}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["cache", "train", "check"])
    p.add_argument("--model", type=Path, default=P.MODELS / "gefs_nseq_B")
    p.add_argument("--out", type=Path, default=P.MODELS / "head_settings")
    p.add_argument("--cache", type=Path, default=P.MODELS / "head_settings/cache.npz")
    p.add_argument("--variant", choices=sorted(VARIANTS), default="V0")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--every", type=int, default=1, help="cache: every n-th issue")
    p.add_argument("--updates", type=int, default=4000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--validate-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    a.model, a.out, a.cache = a.model.resolve(), a.out.resolve(), a.cache.resolve()
    {"cache": cache, "train": train, "check": check}[a.step](a)


if __name__ == "__main__":
    main()
