"""Forecasts for the strength analysis of the graph model (torchrun, all GPUs).

At every test issue (as ``tva_flood.hydro.evaluate``), at every catalog gauge, the mean
forecast and the 0.5, 0.9 and 0.99 flow quantiles of the tail head of
  shown:  the all-gauge model, every gauge's recent flow visible (the gauged test);
  hidden: the all-gauge model, no gauge's recent flow visible (value of the gauge);
  fold:   each fold model at its own withheld gauges (the ungauged test).
Truth is NaN outside the evaluation's mask (CAMELS-H, latest observation and NWM forecast
present). Around Hurricane Helene (issues 2024-09-22 to 09-29) also the forecast at every
reach: the all-gauge model with GEFS rain and with realized rain, and the fold composite
(a reach in a gauge's basin takes that gauge's fold model).

  torchrun --standalone --nproc_per_node 4 scripts/strength_forecasts.py --out models/gnn_strength
"""

import argparse
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from tva_flood.evaluation.benchmark import load_nwm
from tva_flood.hydro import paths as P
from tva_flood.hydro.evaluate import FOLDS, FUTURE, NWMReaches, load_model, reach_folds, test_issues
from tva_flood.hydro.samples import Device, Samples

HELENE = pd.date_range("2024-09-22", "2024-09-29", freq="D")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=P.MODELS / "gnn_B/best.pt")
    p.add_argument("--fold-models", default=str(P.MODELS / "gnn_B_f{k}/best.pt"))
    p.add_argument("--limit", type=int, default=0, help="First n issues and two Helene issues only (testing)")
    a = p.parse_args()
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=120), device_id=device)
    s = Samples()
    gauges, times = s.gauges, s.arrays.times
    gpu = Device(s.network, gauges, s.arrays, device)
    models = {0: load_model(a.model, s, device)[0]}
    models.update({k: load_model(Path(a.fold_models.format(k=k)), s, device)[0] for k in range(1, FOLDS + 1)})
    reach_fold, gauge_fold = reach_folds(s)
    reach_model = torch.as_tensor(np.maximum(reach_fold, 0), device=device)
    none = np.zeros(gauges.count, bool)
    nwm, _ = load_nwm(P.NWM_GAUGES)
    issues, _ = test_issues(s, nwm, NWMReaches().days())
    helene = {step for step in issues if times[step].normalize() in HELENE}
    if a.limit:
        issues = issues[: a.limit] + sorted(helene)[:2]
        helene &= set(issues)
    rows, maps = [], []
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for position, step in enumerate(issues[rank::world]):
            day = times[step].normalize()
            truth = gauges.observed[step : step + P.HORIZON].copy()
            ok = np.isfinite(truth) & np.isfinite(gauges.observed[step - 1])[None] & np.isfinite(nwm[day][0]) & gauges.camelsh[None]
            truth[~ok] = np.nan
            base = gpu.batch(s.build(step, "aorc"))
            rain = s.rain.get("gefs", day)
            full = step in helene
            row = dict(step=step, truth=truth.astype(np.float32))
            reach = {}
            for k, model in models.items():
                withheld = gauge_fold == k if k else none
                view = gpu.with_gauges(base, s.gauge_view(step, s.eligible(step, withheld), withheld))
                out = model(gpu.with_rain(view, rain), full=full)
                mu, q = out["gauge"][:, FUTURE].T.float().cpu().numpy(), out["quantile"].permute(1, 0, 2).float().cpu().numpy()
                if k == 0:
                    row.update(shown_mu=mu, shown_q=q)
                    hidden = model(gpu.with_rain(gpu.with_gauges(base, s.gauge_view(step, none, none)), rain))
                    row.update(hidden_mu=hidden["gauge"][:, FUTURE].T.float().cpu().numpy(), hidden_q=hidden["quantile"].permute(1, 0, 2).float().cpu().numpy())
                    row.update(fold_mu=np.full_like(mu, np.nan), fold_q=np.full_like(q, np.nan))
                    if full:
                        reach["model"] = out["reach"][:, FUTURE].T
                        reach["fold"] = reach["model"].clone()
                        reach["model_aorc"] = model(view, full=True)["reach"][:, FUTURE].T.float().cpu().numpy()
                else:
                    row["fold_mu"][:, withheld], row["fold_q"][:, withheld] = mu[:, withheld], q[:, withheld]
                    if full:
                        reach["fold"] = torch.where((reach_model == k)[None], out["reach"][:, FUTURE].T, reach["fold"])
            rows.append(row)
            if full:
                maps.append(dict(step=step, model=reach["model"].float().cpu().numpy(), fold=reach["fold"].float().cpu().numpy(),
                                 model_aorc=reach["model_aorc"], gefs_rain=rain.astype(np.float32)))
            if rank == 0 and position % 25 == 0:
                print(f"issue {position + 1}/{len(issues[rank::world])}", flush=True)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f".gauges_{rank}.npz", **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
    if maps:
        np.savez(out / f".helene_{rank}.npz", **{k: np.stack([m[k] for m in maps]) for k in maps[0]})
    dist.barrier()
    if rank == 0:
        for name in ("gauges", "helene"):
            parts = [np.load(f) for f in sorted(out.glob(f".{name}_*.npz"))]
            merged = {k: np.concatenate([q[k] for q in parts]) for k in parts[0].files}
            order = np.argsort(merged["step"])
            merged = {k: v[order] for k, v in merged.items()}
            np.savez(out / f"{name}.npz", time=times[merged["step"]].to_numpy(), stations=gauges.names, comid=s.network.ids, **merged)
            for f in out.glob(f".{name}_*.npz"):
                f.unlink()
        print(f"STRENGTH FORECASTS COMPLETE: {out}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
