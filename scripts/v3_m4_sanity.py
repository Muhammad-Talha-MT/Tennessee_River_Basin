"""M4 sanity check before launching M4 (reports/v3):
1. mass conservation of the instantaneous router on the real 58,834-reach network (CPU);
2. skill of a short M4 sanity run (``sanity_M4_f1``: 300 updates of stage A + 300 of stage B on held-out fold 1) against M1 fold 1 (3,000 updates) on the
   same test issues at the fold-1 held-out gauges: median NSE at 6, 24, 72 h;
3. one flood hydrograph at a held-out gauge: observed, M1, sanity M4.
Writes reports/v3_tables/m4_sanity_*.csv and figures/paper/sanity_m4_hydrograph.{png,pdf}. Run on one free GPU: CUDA_VISIBLE_DEVICES=2 python scripts/v3_m4_sanity.py
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from v3_common import A, FIG, OUT, Context, scores
from tva_flood.hydro import paths as P
from tva_flood.hydro.evaluate import FUTURE, load_model
from tva_flood.hydro.model import Router
from tva_flood.hydro.samples import Device, Samples

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
ctx = Context()
net = ctx.network

# ---- 1. mass conservation on the real network -----------------------------------------------------------------------------
groups = int(net.group.max()) + 1  # the kernel groups: irrelevant with all weight on the first tap
r = Router(groups, np.zeros(groups), learned=False)
r.attach(net, ctx.gauges.reach, torch.device("cpu"))
q = torch.rand(net.n, 8)
out = r(q)
expected = np.zeros((net.n, 8))
qn = q.numpy()
rows = np.repeat(np.arange(net.n), np.diff(net.indptr))
np.add.at(expected, rows, net.share[:, None] * qn[net.members])
err_pair = float(np.abs(out.numpy() - expected).max() / expected.max())
# an outlet drains its whole basin: its flow is the share-weighted sum over the basin, and every reach's lateral inflow leaves through the outlets
outlets = np.flatnonzero(np.bincount(net.edges[:, 0], minlength=net.n) == 0)  # reaches with no downstream reach
total_out = out[outlets].sum(0)
total_in = q.sum(0)
mass = pd.DataFrame(dict(check=["max relative error of the router against the share-weighted sum of upstream inflow",
                                f"flow leaving the {len(outlets)} outlets / lateral inflow of the whole network, 8 random steps"],
                         value=[err_pair, float((total_out / total_in).mean())],
                         spread=[np.nan, float((total_out / total_in).std())]))
mass.to_csv(OUT / "m4_sanity_mass.csv", index=False)
print(mass.to_string(index=False))

# ---- 2. skill and hydrograph ---------------------------------------------------------------------------------------------
device = torch.device("cuda:0")
s = Samples()
gpu = Device(s.network, s.gauges, s.arrays, device)
steps = np.load(A / "eval_M0_s1" / "forecasts_gauged.npz", allow_pickle=True)["step"]
withheld = ctx.withheld("heldout", 1)
fold = np.flatnonzero(ctx.fold["heldout"] == 1)
models = {"M1": load_model(A / "M1_s1_f1" / "best.pt", s, device)[0], "M4 sanity run": load_model(A / "sanity_M4_f1" / "best.pt", s, device)[0]}
for m in models.values():
    m.trunk.chunk = 6000

# issues: every 12th test issue, plus the flood chosen for the hydrograph (the largest event at a big held-out gauge with its peak 2-6 days ahead)
obs = s.gauges.observed
best = None
for g in fold:
    mean = np.nanmean(obs[:, g])
    if not mean > 20:
        continue
    for i, t in enumerate(steps):
        w = obs[t : t + P.HORIZON, g]
        if np.isfinite(w).all() and 8 <= int(np.argmax(w)) <= 24 and w.max() / mean > (best[0] if best else 0):
            best = (w.max() / mean, g, i)
_, g_plot, i_plot = best
issues = sorted(set(range(0, len(steps), 12)) | {i_plot})
fc = {m: np.full((len(issues), P.HORIZON, ctx.gauges.count), np.nan, np.float32) for m in models}
truth = np.full((len(issues), P.HORIZON, ctx.gauges.count), np.nan, np.float32)
for n, i in enumerate(issues):
    step = int(steps[i])
    day = s.arrays.times[step].normalize()
    base = gpu.limit_rain(gpu.batch(s.build(step, "aorc")), P.HORIZON)
    rain = s.rain.get("gefs", day)
    view = gpu.with_gauges(base, s.gauge_view(step, s.eligible(step, withheld), withheld))
    truth[n] = obs[step : step + P.HORIZON]
    for name, m in models.items():
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            out = m(gpu.with_rain(view, rain))
        fc[name][n] = out["gauge"][:, FUTURE].float().cpu().numpy().T
    if n % 20 == 0:
        print("issue", n + 1, "of", len(issues), flush=True)

rows = []
for h in (6, 24, 72):
    row = dict(lead_h=h, issues=len(issues), gauges=len(fold))
    for name in models:
        nse = scores(truth, fc[name], min_count=40)["nse"][h // 6 - 1][fold]
        row[f"median_nse_{name}"] = float(np.nanmedian(nse))
        row[f"gauges_scored_{name}"] = int(np.isfinite(nse).sum())
    rows.append(row)
skill = pd.DataFrame(rows)
skill.to_csv(OUT / "m4_sanity_skill.csv", index=False)
print(skill.round(3).to_string(index=False))

n_plot = issues.index(i_plot)
lead_h = (np.arange(P.HORIZON) + 1) * 6
fig, ax = plt.subplots(figsize=(6.4, 3.3))
ax.plot(lead_h, truth[n_plot][:, g_plot], "k", lw=2, label="observed")
ax.plot(lead_h, fc["M1"][n_plot][:, g_plot], color="#c9822b", lw=1.6, label="M1 (3,000 updates)")
ax.plot(lead_h, fc["M4 sanity run"][n_plot][:, g_plot], color="#b05a8c", lw=1.6, label="M4 sanity run (300 + 300 updates)")
ax.set_xlabel("Lead time (h)")
ax.set_ylabel("Flow (m$^3$/s)")
t0 = s.arrays.times[int(steps[i_plot])]
ax.set_title(f"{ctx.names[g_plot]}, held-out gauge, issue {t0:%Y-%m-%d} 00Z")
ax.legend(frameon=False, fontsize=8)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"sanity_m4_hydrograph.{ext}", dpi=300, bbox_inches="tight")
pd.DataFrame(dict(lead_h=lead_h, observed=truth[n_plot][:, g_plot], M1=fc["M1"][n_plot][:, g_plot], M4_sanity=fc["M4 sanity run"][n_plot][:, g_plot])).to_csv(OUT / "m4_sanity_hydrograph.csv", index=False)
print("gauge", ctx.names[g_plot], "issue", t0)
