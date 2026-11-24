"""Phase 1.1: switch insertion on and off in a real M0 fold-1 model (same weights, fp32, CPU) and see where the forecast changes.

Expectation from the code: visible gauges keep the gauge head's forecast (computed before insertion), so insertion changes only gauges that are NOT visible
at the issue and lie below a visible gauge. Writes reports/v3_tables/insertion_toggle_check.csv. Run: python scripts/v3_insertion_toggle_check.py
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
from v3_common import A, OUT, Context
from tva_flood.hydro import paths as P
from tva_flood.hydro.evaluate import FUTURE, load_model
from tva_flood.hydro.samples import Device, Samples

torch.set_num_threads(6)
device = torch.device("cpu")
s = Samples()
gpu = Device(s.network, s.gauges, s.arrays, device)
ctx = Context()
steps = np.load(A / "eval_M0_s1" / "forecasts_gauged.npz", allow_pickle=True)["step"]
model, _ = load_model(A / "M0_s1_f1" / "best.pt", s, device)
model.trunk.chunk = 4000
withheld = ctx.withheld("heldout", 1)
rows = []
for issue in (1000, 1500):  # two test issues
    step = int(steps[issue])
    day = s.arrays.times[step].normalize()
    base = gpu.limit_rain(gpu.batch(s.build(step, "aorc")), P.HORIZON)
    rain = s.rain.get("gefs", day)
    visible = s.eligible(step, withheld)
    batch = gpu.with_rain(gpu.with_gauges(base, s.gauge_view(step, visible, withheld)), rain)
    outs = {}
    for flag in (True, False):
        model.config["insert"] = flag
        with torch.inference_mode():
            outs[flag] = model(batch)["gauge"][:, FUTURE].float().numpy()
    model.config["insert"] = True
    d = np.abs(outs[True] - outs[False])  # [gauge, lead]
    below = np.array([len(ctx.upstream_gauges(g)[visible[ctx.upstream_gauges(g)]]) > 0 for g in range(ctx.gauges.count)])  # a visible gauge upstream
    for label, mask in (("visible gauges", visible), ("hidden, no visible gauge upstream", ~visible & ~below), ("hidden, visible gauge upstream", ~visible & below)):
        if mask.any():
            rows.append(dict(issue_step=step, subset=label, gauges=int(mask.sum()), max_abs_change_m3s=float(d[mask].max()), median_abs_change_m3s=float(np.median(d[mask])),
                             max_flow_m3s=float(outs[False][mask].max())))
df = pd.DataFrame(rows)
df.to_csv(OUT / "insertion_toggle_check.csv", index=False)
print(df.round(5).to_string(index=False))
