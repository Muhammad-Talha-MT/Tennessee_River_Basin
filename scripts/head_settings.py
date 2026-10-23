"""Head-settings test: does the gauge head's size, input scaling, inputs or loss explain the day-1 gap to the LSTM?

The trunk, routing and everything upstream of the gauge head are frozen (a stage-B model). What the head reads is
cached once for every training and selection issue (GEFSv12 reforecast days 2000-2019, every eligible gauge visible, the
gauged test's view); heads are then trained from scratch on identical (issue, gauge) pairs, differing in one setting
at a time, and scored on 2019 at the training gauges (median NSE by lead; selection = days 1-5).

  torchrun --standalone --nproc_per_node 4 scripts/head_settings.py cache --model models/gefs_nseq_B --out models/head_settings
  python scripts/head_settings.py train --cache models/head_settings/cache.npz --variant V0 --gpu 0
  python scripts/head_settings.py assemble --variant V4 --head models/head_settings/V4/best.pt --model models/gefs_nseq_B --out models/gnn_v4
  torchrun ... scripts/head_settings.py cache --model models/gnn_v4 --out models/gnn_v4_cache     # inputs and the model's own forecasts
  python scripts/head_settings.py tail --model models/gnn_v4 --cache models/gnn_v4_cache/cache.npz --out models/gnn_v4_tail
  python scripts/head_settings.py check --cache models/head_settings/cache.npz --model models/gefs_nseq_B
  torchrun ... scripts/head_settings.py cache --split test --out models/head_settings_test   # the 1,917 test issues, GEFS operational rain
  python scripts/head_settings.py test --cache models/head_settings_test/cache.npz --heads V4_s1=V4@models/head_settings/V4/best.pt

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
from tva_flood.training.runtime import save_atomic
from tva_flood.hydro import paths as P
from tva_flood.evaluation.benchmark import load_nwm
from tva_flood.hydro.evaluate import NWMReaches, load_model, test_issues
from tva_flood.hydro.model import GaugeHead, SparseProduct
from tva_flood.hydro.samples import Device, Samples
from tva_flood.hydro.model import QUANTILES
from tva_flood.hydro.train import DAYS_1_5, FUTURE, nse_star, pinball, reforecast_issues, withheld_gauges
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


# ---------------------------------------------------------------- cache
def batch_for(s, gpu, step, split, withheld):
    """The model's inputs at an issue: the training view (GEFS reforecast rain, eligible gauges visible) or the test view
    (operational GEFS rain, every eligible gauge visible: the gauged protocol)."""
    if split == "train":
        return gpu.batch(s.build(step, "gefs_reforecast", s.eligible(step, withheld), withheld))
    none = np.zeros(s.gauges.count, bool)
    view = gpu.with_gauges(gpu.batch(s.build(step, "aorc")), s.gauge_view(step, s.eligible(step, none), none))
    return gpu.with_rain(view, s.rain.get("gefs", s.arrays.times[step].normalize()))


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
    obs = s.gauges.observed
    if a.split == "test":
        nwm, _ = load_nwm(P.NWM_GAUGES)
        issues, _ = test_issues(s, nwm, NWMReaches().days())
    else:
        last = s.arrays.times[s.arrays.steps - P.HORIZON - 1]
        issues = reforecast_issues(P.TRAIN_FROM, min(last, pd.Timestamp("2019-12-21")))[:: a.every]
    mine = issues[rank::world]
    parts = {k: [] for k in ("step", "routed", "sim", "rain", "weather", "stage_b", "history", "target", "ratio", "flags")}
    began = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for position, step in enumerate(mine):
            b = batch_for(s, gpu, step, a.split, withheld)
            out = model(b)
            if "quantile" in out:  # the tail head's offsets: quantile / mean, to put the tail around another mean
                mean = out["gauge"][:, FUTURE]
                ratio = out["quantile"].float() / mean.float().clamp_min(1e-3)[..., None]
            else:
                ratio = torch.zeros(len(s.gauges.reach), P.HORIZON, 3, device=device)
            flags = torch.stack([b["visible"].float(), b["owned"][b["gauge_reach"]], torch.log1p(out["gauge"][:, -1].float()) / 5], dim=-1)
            values = dict(
                step=step,
                ratio=ratio,
                flags=flags,
                routed=out["routed_gauge"],
                sim=out["sim"][b["gauge_reach"]],
                rain=model.gauge_rain(b),
                weather=model.gauge_weather(b),
                stage_b=out["gauge"][:, FUTURE],
            )
            for k, v in values.items():
                parts[k].append(np.asarray(v) if k == "step" else v.float().cpu().numpy().astype(np.float16 if k in ("weather", "ratio") else np.float32))
            truth = obs[step : step + P.HORIZON].T.astype(np.float32)
            if a.split == "test":  # as the evaluation: CAMELS-H gauges with a latest observation and an NWM forecast
                day = s.arrays.times[step].normalize()
                keep = np.isfinite(truth) & np.isfinite(obs[step - 1])[:, None] & np.isfinite(nwm[day][0].T) & s.gauges.camelsh[:, None]
                truth = np.where(keep, truth, np.nan)
            parts["history"].append(obs[step - HISTORY : step].T.astype(np.float32))
            parts["target"].append(truth)
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
            static=gpu.fixed["static"][gpu.fixed["gauge_reach"]].float().cpu().numpy(),
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
        self.t = {k: torch.as_tensor(c[k], device=device) for k in ("routed", "sim", "rain", "weather", "stage_b", "history", "target", "ratio", "flags") if k in c.files}
        self.static = torch.as_tensor(c["static"], device=device) if "static" in c.files else None
        self.target_np = c["target"]
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
    """The head's forecast [pairs, lead] (routed flow x exp of its correction) and the cached inputs."""
    x = cache.gather(k, g)
    kwargs = dict(weather=x["weather"], basin=cache.basin[g]) if head.lstm_inputs else {}
    correction = head(x["history"], x["routed"], x["sim"], x["rain"], cache.describe[g], **kwargs)
    return x["routed"][:, P.INNOVATION :] * torch.exp(correction), x


def new_head(cfg, cache):
    return GaugeHead(hidden=cfg["hidden"], scale=cfg["scale"], lstm_inputs=cfg["lstm_inputs"], flow_mean=cache.flow_mean, flow_std=cache.flow_std)


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
    head = new_head(cfg, cache_).to(device)
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


@torch.no_grad()
def test(a):
    """Forecasts [issue, lead, gauge] of saved heads on a test cache (NaN where a gauge has no latest observation), and the
    stage-B head's, with the truth masked as in the evaluation. ``--heads tag=VARIANT@path`` ..."""
    device = torch.device(f"cuda:{a.gpu}")
    cache_ = Cache(a.cache, device)
    out = dict(time=cache_.time.to_numpy(), truth=cache_.target_np, stage_b=cache_.t["stage_b"].permute(0, 2, 1).cpu().numpy(),
               ratio=cache_.t["ratio"].permute(0, 2, 1, 3).cpu().numpy())
    for spec in a.heads:
        tag, rest = spec.split("=")
        variant, path = rest.split("@")
        cfg = VARIANTS[variant]
        head = new_head(cfg, cache_).to(device).eval()
        head.load_state_dict(torch.load(path, map_location=device))
        mu = np.full((len(cache_.time), P.HORIZON, len(cache_.std)), np.nan, np.float32)
        for k in range(len(cache_.time)):
            g = torch.as_tensor(np.flatnonzero(cache_.latest[k]), device=device)
            if len(g):
                pred, _ = run(head, cache_, torch.full_like(g, k), g)
                mu[k][:, g.cpu().numpy()] = pred.T.float().cpu().numpy()
        out[tag] = mu
        print("forecast", tag, flush=True)
    np.savez(a.out / "test_forecasts.npz", **out)
    print(f"TEST FORECASTS COMPLETE: {a.out / 'test_forecasts.npz'}", flush=True)


def tail(a):
    """Train the model's tail head (flow quantiles around the forecast) on a frozen model: pinball loss on the cached forecasts
    (``stage_b`` of a cache built with this model), selected on 2019. Writes the model's checkpoint with the new tail."""
    device = torch.device(f"cuda:{a.gpu}")
    s = Samples()
    model, state = load_model(a.model / "best.pt", s, device)
    cache_ = Cache(a.cache, device)
    params = list(model.tail.parameters()) + list(model.lead.parameters())
    for p_ in model.parameters():
        p_.requires_grad_(False)
    for p_ in params:
        p_.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    def quantiles(k, g):
        t = cache_.t
        return model.quantiles(cache_.static[g], t["flags"][k, g], t["stage_b"][k, g]), t["target"][k, g]

    @torch.no_grad()
    def validate():
        loss, below, n = 0.0, torch.zeros(len(QUANTILES), device=device, dtype=torch.float64), 0
        for k in cache_.sel:
            g = torch.as_tensor(np.flatnonzero(cache_.trained & cache_.latest[k]), device=device)
            if not len(g):
                continue
            q, truth = quantiles(torch.full_like(g, int(k)), g)
            valid = torch.isfinite(truth)
            loss += float(pinball(q, torch.nan_to_num(truth), valid, cache_.std[g])) * len(g)
            below += ((torch.nan_to_num(truth)[..., None] <= q) & valid[..., None]).sum((0, 1)).double() / valid.sum()
            n += len(g)
        return loss / n, (below / len(cache_.sel)).cpu().numpy()

    best, log, began = math.inf, [], time.perf_counter()
    for update in range(1, a.updates + 1):
        lr = a.lr * min(1.0, update / 100) * 0.5 * (1 + math.cos(math.pi * update / a.updates))
        for group in optimizer.param_groups:
            group["lr"] = max(lr, a.lr * 0.01)
        k, g = (torch.as_tensor(x, device=device) for x in cache_.pairs[rng.integers(0, len(cache_.pairs), a.batch)].T)
        q, truth = quantiles(k, g)
        valid = torch.isfinite(truth)
        loss = pinball(q, torch.nan_to_num(truth), valid, cache_.std[g])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if update % a.validate_every == 0 or update == a.updates:
            v, below = validate()
            log.append(dict(update=update, train=float(loss), validation=v, below=[round(float(x), 4) for x in below]))
            print(f"update {update}/{a.updates} train {float(loss):.4f} validation {v:.4f} share below q50/q90/q99 {np.round(below, 3)}", flush=True)
            if v < best:
                best = v
                state["model"].update({f"tail.{k_}": x.cpu() for k_, x in model.tail.state_dict().items()})
                state["model"]["lead.weight"] = model.lead.weight.detach().cpu()
                state["tail_head"] = dict(validation_pinball=v, update=update, cache=str(a.cache))
    a.out.mkdir(parents=True, exist_ok=True)
    save_atomic(state, a.out / "best.pt")
    atomic_json(a.out / "tail.json", log)
    print(f"TAIL COMPLETE: validation pinball {best:.4f} -> {a.out / 'best.pt'}", flush=True)


def assemble(a):
    """One checkpoint: the stage-B model with its gauge head replaced by a trained variant (settings in the config)."""
    state = torch.load(a.model / "best.pt", map_location="cpu", weights_only=False)
    cfg = VARIANTS[a.variant]
    c = np.load(a.cache)
    head = dict(hidden=cfg["hidden"], scale=cfg["scale"], lstm_inputs=cfg["lstm_inputs"], flow_mean=float(c["flow_mean"]), flow_std=float(c["flow_std"]))
    weights = torch.load(a.head, map_location="cpu")
    model = {k: v for k, v in state["model"].items() if not k.startswith("head.")}
    model.update({f"head.{k}": v for k, v in weights.items()})
    state.update(model=model, head_swap=dict(variant=a.variant, source=str(a.head), stage_b=str(a.model)))
    state["config"] = dict(state["config"], head=head)
    a.out.mkdir(parents=True, exist_ok=True)
    save_atomic(state, a.out / "best.pt")
    print(f"ASSEMBLED {a.out / 'best.pt'}: {a.model.name} with the {a.variant} head {head}")


def check(a):
    """The cache and the head replica are right if GaugeHead with the stage-B head's weights reproduces its forecasts."""
    device = torch.device("cuda:0")
    s = Samples()
    model, _ = load_model(a.model / "best.pt", s, device)
    cache_ = Cache(a.cache, device)
    head = GaugeHead().to(device)
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
    p.add_argument("step", choices=["cache", "train", "check", "test", "assemble", "tail"])
    p.add_argument("--split", choices=["train", "test"], default="train", help="cache: the training/selection issues or the test issues")
    p.add_argument("--heads", nargs="*", default=[], help="test: tag=VARIANT@path of saved heads")
    p.add_argument("--head", type=Path, help="assemble: the trained head's best.pt")
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
    {"cache": cache, "train": train, "check": check, "test": test, "assemble": assemble, "tail": tail}[a.step](a)


if __name__ == "__main__":
    main()
