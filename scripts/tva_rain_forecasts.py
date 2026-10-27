"""The graph model with TVA's own rain forecast at TVA's headwater gauges (torchrun, all GPUs): a rain diagnostic.

At every test issue (as ``tva_flood.hydro.evaluate``) where TVA issued a forecast, every reach of a TVA gauge's basin
gets TVA's basin-mean rain forecast in place of the GEFSv12 control (a reach in several TVA basins takes the smallest);
every other reach keeps GEFS. Two variants:
  tva:      TVA's rain at every lead, as TVA's own model had it (real to about 72 h, near zero after about 5 days);
  tva_gefs: TVA's rain to 72 h, GEFS after it.
For each, the mean forecast at every catalog gauge of
  shown: the all-gauge model, every gauge's recent flow visible (the gauged test);
  fold:  each fold model at its own withheld gauges (the ungauged test), only without ``--gauged-only``.
TVA's forecast uses the gauge's flow, so it has no ungauged counterpart: the gauged run is the like-for-like test.
``rain_ok`` [issue, gauge] marks TVA gauges with a full TVA rain forecast (lead 6 to 240 h) at that issue; compare
with models/gnn_strength (the same models and code path with GEFS rain).

  torchrun --standalone --nproc_per_node 4 scripts/tva_rain_forecasts.py --out models/gnn_tva_rain
"""

import argparse
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from tva_flood.data import tva
from tva_flood.evaluation.benchmark import load_nwm
from tva_flood.hydro import paths as P
from tva_flood.hydro.evaluate import FOLDS, FUTURE, NWMReaches, load_model, reach_folds, test_issues
from tva_flood.hydro.samples import Device, Samples

TVA_LEADS = 12  # TVA's rain forecast is real to 72 h


def tva_rain(s):
    """TVA basin rain forecasts {issue step: [lead, station] mm}, the stations' gauge indices and basins (largest first)."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = tva.extract(P.ROOT / "raw/tva/TVA_headwater.zip", tmp)
        table = tva.gauges(folder, s.network.ids)
        names = list(s.gauges.names)
        table = table[table.tva_nwsid.isin(names)].reset_index(drop=True)
        _, _, rain = tva.align(folder, table, s.arrays.times)
    gauge = np.array([names.index(n) for n in table.tva_nwsid])
    order = np.argsort(-s.gauges.size[gauge], kind="stable")  # smaller basins written last win
    return rain, gauge[order], order


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=P.MODELS / "gnn_B/best.pt")
    p.add_argument("--fold-models", default=str(P.MODELS / "gnn_B_f{k}/best.pt"))
    p.add_argument("--gauged-only", action="store_true", help="Skip the fold models (the ungauged variant)")
    p.add_argument("--limit", type=int, default=0, help="First n issues only (testing)")
    a = p.parse_args()
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=120), device_id=device)
    s = Samples()
    gauges, times = s.gauges, s.arrays.times
    gpu = Device(s.network, gauges, s.arrays, device)
    models = {0: load_model(a.model, s, device)[0]}
    if not a.gauged_only:
        models.update({k: load_model(Path(a.fold_models.format(k=k)), s, device)[0] for k in range(1, FOLDS + 1)})
    _, gauge_fold = reach_folds(s)
    none = np.zeros(gauges.count, bool)
    nwm, _ = load_nwm(P.NWM_GAUGES)
    issues, _ = test_issues(s, nwm, NWMReaches().days())
    rain_tva, tva_gauge, column = tva_rain(s)
    issues = [step for step in issues if step in rain_tva]
    if a.limit:
        issues = issues[: a.limit]
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for position, step in enumerate(issues[rank::world]):
            day = times[step].normalize()
            base = gpu.batch(s.build(step, "aorc"))
            gefs = s.rain.get("gefs", day)
            mm = rain_tva[step][:, column]  # [lead, station], largest basin first
            ok = np.isfinite(mm).all(axis=0) & (mm >= 0).all(axis=0)
            variants = {"tva": gefs.astype(np.float32).copy(), "tva_gefs": gefs.astype(np.float32).copy()}
            for g, rain, good in zip(tva_gauge, mm.T, ok):
                if good:
                    basin = gauges.basins[g]
                    variants["tva"][:, basin] = rain[:, None]
                    variants["tva_gefs"][:TVA_LEADS, basin] = rain[:TVA_LEADS, None]
            rain_ok = np.zeros(gauges.count, bool)
            rain_ok[tva_gauge[ok]] = True
            row = dict(step=step, rain_ok=rain_ok)
            for name, rain in variants.items():
                fold = np.full((P.HORIZON, gauges.count), np.nan, np.float32)
                for k, model in models.items():
                    withheld = gauge_fold == k if k else none
                    view = gpu.with_gauges(base, s.gauge_view(step, s.eligible(step, withheld), withheld))
                    mu = model(gpu.with_rain(view, rain))["gauge"][:, FUTURE].T.float().cpu().numpy()
                    if k == 0:
                        row[f"{name}_shown"] = mu
                    else:
                        fold[:, withheld] = mu[:, withheld]
                if not a.gauged_only:
                    row[f"{name}_fold"] = fold
            rows.append(row)
            if rank == 0 and position % 25 == 0:
                print(f"issue {position + 1}/{len(issues[rank::world])}", flush=True)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f".gauges_{rank}.npz", **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
    dist.barrier()
    if rank == 0:
        parts = [np.load(f) for f in sorted(out.glob(".gauges_*.npz"))]
        merged = {k: np.concatenate([q[k] for q in parts]) for k in parts[0].files}
        order = np.argsort(merged["step"])
        merged = {k: v[order] for k, v in merged.items()}
        np.savez(out / "gauges.npz", time=times[merged["step"]].to_numpy(), stations=gauges.names, **merged)
        for f in out.glob(".gauges_*.npz"):
            f.unlink()
        print(f"TVA RAIN FORECASTS COMPLETE: {out}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
