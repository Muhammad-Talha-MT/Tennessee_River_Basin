"""Stage C: the gauge head trained at the scale of the LSTM, on a frozen stage-B model.

A stage-B update shows each gauge a few times; the LSTM baseline sees 3.3 million (issue,
gauge) pairs. Here the rest of the model is frozen and what the head reads is cached once:
for every training and selection issue (GEFSv12 reforecast days, 2000-2019) one forward
pass with every eligible gauge visible (the gauged test's view; a fold's gauges stay
hidden) gives, at each gauge, the routed and simulated flow over the scored window, the
basin-mean forecast rain and the basin-mean weather history; the observed flow comes from
the gauge records. The head (started from stage B) is then trained on (issue, gauge) pairs
with the stage-B gauge loss and selected on 2019 (median NSE over the training gauges,
leads 24-120 h). The result is the stage-B checkpoint with the new head (``final.pt``).

  torchrun --standalone --nproc_per_node 4 -m tva_flood.hydro.head cache --model DIR
  python -m tva_flood.hydro.head train --model DIR
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

from tva_flood.common import atomic_json
from tva_flood.training.runtime import save_atomic

from . import paths as P
from .model import RegulatedModel
from .samples import Device, Samples
from .train import DAYS_1_5, FUTURE, LEADS, bounded_nse, reforecast_issues, withheld_gauges

HEAD_HISTORY = 120  # steps of observed flow and weather the head reads


def load(path, s, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    c = state["config"]
    model = RegulatedModel(c["groups"], c["delay"], c["dams"], c["hidden"], c["channels"], c["graph_layers"], c["pool_layers"], c["plan"])
    model.load_state_dict(state["model"])
    model.attach(s.network, s.network.edges, s.gauges.reach, device, s.dams)
    return model.to(device).eval(), state


def inputs(model, b):
    """What the head reads at every gauge: routed, simulated [G, S]; rain [G, lead, 2];
    weather [G, H, 16]."""
    out = model(b)
    rain, weather = model.gauge_inputs(b)
    return out["routed"], out["sim"][b["gauge_reach"]], rain, weather


def cache(a):
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60), device_id=device)
    s = Samples()
    model, state = load(a.model / "best.pt", s, device)
    gpu = Device(s.network, s.gauges, s.arrays, device, s.dams)
    withheld = withheld_gauges(state["fold"])
    last = s.arrays.times[s.arrays.steps - P.HORIZON - 1]
    issues = reforecast_issues(P.TRAIN_FROM, min(last, pd.Timestamp("2019-12-21")))[:: a.every]
    mine = issues[rank::world]
    parts = {k: [] for k in ("step", "routed", "sim", "rain", "weather")}
    began = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for position, step in enumerate(mine):
            b = gpu.batch(s.build(step, "gefs_reforecast", s.eligible(step, withheld), withheld))
            routed, sim, rain, weather = inputs(model, b)
            for key, value in (("routed", routed), ("sim", sim), ("rain", rain), ("weather", weather)):
                parts[key].append(value.float().cpu().numpy().astype(np.float16 if key == "weather" else np.float32))
            parts["step"].append(step)
            if rank == 0 and position % 200 == 0:
                print(f"cache {position + 1}/{len(mine)} {(time.perf_counter() - began) / 60:.1f} min", flush=True)
    np.savez(a.model / f".head_cache_{rank}.npz", **{k: np.stack(v) for k, v in parts.items()})
    if world > 1:
        dist.barrier()
    if rank == 0:
        loaded = [np.load(a.model / f".head_cache_{r}.npz") for r in range(world)]
        merged = {k: np.concatenate([part[k] for part in loaded]) for k in parts}
        order = np.argsort(merged["step"])
        np.savez(a.model / "head_cache.npz", **{k: v[order] for k, v in merged.items()})
        for r in range(world):
            (a.model / f".head_cache_{r}.npz").unlink()
        print(f"CACHE COMPLETE: {len(order)} issues -> {a.model / 'head_cache.npz'}", flush=True)
    if world > 1:
        dist.destroy_process_group()


def train(a):
    device = torch.device("cuda:0")
    s = Samples()
    model, state = load(a.model / "best.pt", s, device)
    head = model.head.train()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    for p_ in head.parameters():
        p_.requires_grad_(True)
    gpu = Device(s.network, s.gauges, s.arrays, device, s.dams)
    describe = model.gauge_describe(gpu.fixed)
    c = np.load(a.model / "head_cache.npz")
    steps = c["step"]
    data = {k: torch.as_tensor(c[k], device=device) for k in ("routed", "sim", "rain", "weather")}
    observed = s.gauges.observed
    withheld = withheld_gauges(state["fold"])
    trained = s.gauges.trained & ~withheld
    selection = s.arrays.times[steps] >= P.TRAIN_UNTIL
    # (issue, gauge) pairs with a target and a latest observation, as the LSTM's
    pairs = []
    for k in np.flatnonzero(~selection):
        ahead = observed[steps[k] : steps[k] + P.HORIZON]
        ok = trained & np.isfinite(observed[steps[k] - 1]) & np.isfinite(ahead).any(axis=0)
        pairs += [(k, g) for g in np.flatnonzero(ok)]
    pairs = np.array(pairs)
    updates = math.ceil(a.pairs / a.batch)
    print(f"{len(pairs):,} training pairs; {updates:,} updates of {a.batch}; {int(selection.sum())} selection issues", flush=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    std = torch.as_tensor(s.gauges.std_raw, device=device)
    eps = gpu.fixed["gauge_eps"]
    p90 = gpu.fixed["gauge_p90"]
    rng = np.random.default_rng(a.seed)

    def forward(k, g):
        """Forecast [pairs, lead] at gauges g of cached issues k (numpy index arrays)."""
        history = torch.as_tensor(np.stack([observed[steps[i] - HEAD_HISTORY : steps[i], j] for i, j in zip(k, g)]), device=device)
        kk, gg = torch.as_tensor(k, device=device), torch.as_tensor(g, device=device)
        routed = data["routed"][kk, gg]
        correction = head(history, routed, data["sim"][kk, gg], data["rain"][kk, gg], describe[gg], data["weather"][kk, gg].float())
        return routed[:, FUTURE] * torch.exp(correction)

    def score():
        head.eval()
        sums = torch.zeros(4, s.gauges.count, P.HORIZON, dtype=torch.float64, device=device)
        with torch.no_grad():
            for k in np.flatnonzero(selection):
                g = np.flatnonzero(trained & np.isfinite(observed[steps[k] - 1]))
                if not len(g):
                    continue
                pred = forward(np.full(len(g), k), g).double()
                truth = torch.as_tensor(observed[steps[k] : steps[k] + P.HORIZON, g].T, device=device).double()
                v = torch.isfinite(truth).double()
                t = torch.nan_to_num(truth)
                sums[:, torch.as_tensor(g, device=device)] += torch.stack([v, t * v, t * t * v, (pred - t) ** 2 * v])
        head.train()
        n, so, soo, se = sums
        var = soo / n.clamp_min(1) - (so / n.clamp_min(1)) ** 2
        nse = torch.where((n >= 30) & (var > 1e-6), 1 - se / (n * var).clamp_min(1e-12), torch.nan).cpu().numpy()
        value = float(np.nanmean(np.nanmedian(nse[trained][:, DAYS_1_5], axis=0)))
        return value if np.isfinite(value) else -math.inf  # too few selection issues to score

    best = score()
    print(f"stage-B head: selection {best:.4f}", flush=True)
    best_state = {k: v.clone() for k, v in head.state_dict().items()}
    log = [dict(update=0, selection=best if np.isfinite(best) else None)]
    began = time.perf_counter()
    for update in range(1, updates + 1):
        lr = a.lr * min(1.0, update / 200) * 0.5 * (1 + math.cos(math.pi * update / updates))
        for group in optimizer.param_groups:
            group["lr"] = max(lr, a.lr * 0.01)
        k, g = pairs[rng.integers(0, len(pairs), a.batch)].T
        pred = forward(k, g)
        truth = torch.as_tensor(np.stack([observed[steps[i] : steps[i] + P.HORIZON, j] for i, j in zip(k, g)]), device=device)
        valid = torch.isfinite(truth)
        truth = torch.nan_to_num(truth)
        gg = torch.as_tensor(g, device=device)
        flood = 1.0 + 2.0 * (truth > p90[gg][:, None]).float()
        loss = bounded_nse(pred, truth, valid, std[gg] ** 2 + 1e-4, eps[gg], weight=flood)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if update % a.validate_every == 0 or update == updates:
            selected = score()
            log.append(dict(update=update, selection=selected if np.isfinite(selected) else None, loss=float(loss), minutes=(time.perf_counter() - began) / 60))
            print(f"update {update}/{updates} loss {float(loss):.4f} selection {selected:.4f}", flush=True)
            if selected > best:
                best, best_state = selected, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    state["model"] = model.state_dict()
    state["head"] = dict(selection=best, pairs=a.pairs, batch=a.batch, lr=a.lr)
    save_atomic(state, a.model / "final.pt")
    atomic_json(a.model / "head.json", log)
    print(f"HEAD COMPLETE: selection {best:.4f} -> {a.model / 'final.pt'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=["cache", "train"])
    p.add_argument("--model", type=Path, required=True, help="Stage-B output folder (best.pt)")
    p.add_argument("--pairs", type=int, default=3_300_807, help="Training (issue, gauge) pairs, as the LSTM")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--validate-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--every", type=int, default=1, help="Cache: every n-th issue only (testing)")
    a = p.parse_args()
    a.model = a.model.resolve()
    cache(a) if a.step == "cache" else train(a)


if __name__ == "__main__":
    main()
