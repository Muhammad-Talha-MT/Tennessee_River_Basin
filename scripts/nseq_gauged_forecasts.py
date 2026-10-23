"""All-gauge gefs_nseq_B at every test issue (gauged protocol, recent flow shown): mean forecast and the 0.5/0.9/0.99
flow quantiles at every catalog gauge; truth masked as in the evaluation. torchrun, all GPUs."""
import os, sys
from datetime import timedelta
import numpy as np, pandas as pd, torch, torch.distributed as dist
from tva_flood.evaluation.benchmark import load_nwm
from tva_flood.hydro import paths as P
from tva_flood.hydro.evaluate import FUTURE, NWMReaches, load_model, test_issues
from tva_flood.hydro.samples import Device, Samples
rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
device = torch.device(f"cuda:{local}"); torch.cuda.set_device(device)
dist.init_process_group("nccl", timeout=timedelta(minutes=90), device_id=device)
s = Samples(); gpu = Device(s.network, s.gauges, s.arrays, device)
model, state = load_model(P.MODELS / "gefs_nseq_B/best.pt", s, device)
nwm, _ = load_nwm(P.NWM_GAUGES); times = s.arrays.times
issues, _ = test_issues(s, nwm, NWMReaches().days())
none = np.zeros(s.gauges.count, bool); rows = []
with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
    for position, step in enumerate(issues[rank::world]):
        day = times[step].normalize()
        truth = s.gauges.observed[step : step + P.HORIZON].copy()
        ok = np.isfinite(truth) & np.isfinite(s.gauges.observed[step - 1])[None] & np.isfinite(nwm[day][0]) & s.gauges.camelsh[None]
        truth[~ok] = np.nan
        batch = gpu.with_rain(gpu.with_gauges(gpu.batch(s.build(step, "aorc")), s.gauge_view(step, s.eligible(step, none), none)), s.rain.get("gefs", day))
        out = model(batch)
        rows.append(dict(step=step, truth=truth.astype(np.float32), mu=out["gauge"][:, FUTURE].T.float().cpu().numpy(),
                         q=out["quantile"].permute(1, 0, 2).float().cpu().numpy()))
        if rank == 0 and position % 40 == 0: print(f"{position + 1}/{len(issues[rank::world])}", flush=True)
np.savez(f"{os.environ['OUT']}/.part_{rank}.npz", **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
dist.barrier()
if rank == 0:
    parts = [np.load(f"{os.environ['OUT']}/.part_{r}.npz") for r in range(world)]
    m = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}; o = np.argsort(m["step"])
    np.savez(f"{os.environ['OUT']}/nseq_gauged.npz", time=times[m["step"][o]].to_numpy(), best_update=state["update"], **{k: v[o] for k, v in m.items()})
    for r in range(world): os.remove(f"{os.environ['OUT']}/.part_{r}.npz")
    print("DONE", state["update"])
dist.destroy_process_group()
