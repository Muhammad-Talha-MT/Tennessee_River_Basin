"""Phase 1.1: leakage and causality checks on a real M0 / M1 fold model (one GPU, a few test issues).

Forecasts at every gauge must not change when we alter what the model must not see:
  A. the flow at visible gauges after the issue (unknown at issue time);
  B. the true flow of the withheld gauges at any time (they are hidden by ``Samples.gauge_view``);
and must change when we hide a visible gauge upstream of a withheld one (control: the model does use upstream observations).
Writes reports/v3_tables/leak_check.csv. Run: python scripts/v3_leak_check.py
"""
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
from v3_common import A, OUT, Context
from tva_flood.hydro import paths as P
from tva_flood.hydro.evaluate import FUTURE, load_model
from tva_flood.hydro.samples import Device, Samples

OUT.mkdir(parents=True, exist_ok=True)
BF16 = os.environ.get("BF16", "0") == "1"  # the evaluation uses bf16; fp32 here makes the checks exact up to float noise
device = torch.device("cuda:0")
s = Samples()
gpu = Device(s.network, s.gauges, s.arrays, device)
ctx = Context()
steps = np.load(A / "eval_M0_s1" / "forecasts_gauged.npz", allow_pickle=True)["step"]
recent_ok = np.stack([np.isfinite(s.gauges.observed[t - 32 : t]).sum(0) >= 8 for t in steps])

rows = []
for name, fold in (("M0", 1), ("M1", 1)):
    model, _ = load_model(A / f"{name}_s1_f{fold}" / "best.pt", s, device)
    model.trunk.chunk = 2500  # fewer reaches per LSTM call: same result, far less memory in fp32
    withheld = ctx.withheld("heldout", fold)
    # held-out gauges of this fold that have a visible upstream gauge
    targets = [g for g in np.flatnonzero(ctx.fold["heldout"] == fold) if len(ctx.upstream_gauges(g)[~withheld[ctx.upstream_gauges(g)]])]
    for rep, (g, issue) in enumerate([(g, i) for g in targets[:2] for i in range(40, len(steps), 700)]):
        step = int(steps[issue])
        up = ctx.upstream_gauges(g)
        up = up[~withheld[up] & recent_ok[issue, up]]
        if len(up) == 0:
            continue
        day = s.arrays.times[step].normalize()
        base = gpu.limit_rain(gpu.batch(s.build(step, "aorc")), P.HORIZON)
        rain = s.rain.get("gefs", day)

        def run(view):
            with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16, enabled=BF16):
                out = model(gpu.with_rain(view, rain))
            return out["gauge"][:, FUTURE].float().cpu().numpy()

        visible = s.eligible(step, withheld)
        view0 = s.gauge_view(step, visible, withheld)
        ref = run(gpu.with_gauges(base, view0))
        again = run(gpu.with_gauges(base, view0))
        # A: observed flow after the issue changes (visible gauges)
        va = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in view0.items()}
        va["observed"][P.HISTORY :, :] = np.where(np.isfinite(va["observed"][P.HISTORY :, :]), va["observed"][P.HISTORY :, :] + 1000.0, np.nan)
        out_a = run(gpu.with_gauges(base, va))
        # B: the withheld gauges' true flow is replaced before the view is built
        saved = s.gauges.observed[:, withheld].copy()
        s.gauges.observed[:, withheld] = 1.0e6
        out_b = run(gpu.with_gauges(base, s.gauge_view(step, visible, withheld)))
        s.gauges.observed[:, withheld] = saved
        # control: hide the visible upstream gauges of the target
        vis_c = visible.copy()
        vis_c[up] = False
        out_c = run(gpu.with_gauges(base, s.gauge_view(step, vis_c, withheld)))
        sel = [g]
        rows.append(dict(model=name, fold=fold, target=ctx.names[g], issue_step=step, visible_upstream=len(up),
                         rerun_max_abs_diff=float(np.abs(again - ref).max()),
                         A_future_obs_changed_max_abs_diff=float(np.abs(out_a - ref).max()),
                         B_withheld_true_flow_changed_max_abs_diff=float(np.abs(out_b - ref).max()),
                         control_upstream_hidden_target_max_abs_diff=float(np.abs(out_c[sel] - ref[sel]).max()),
                         target_flow_mean_m3s=float(ref[g].mean())))
df = pd.DataFrame(rows)
df.to_csv(OUT / ("leak_check_bf16.csv" if BF16 else "leak_check.csv"), index=False)
print(df.round(5).to_string(index=False))
