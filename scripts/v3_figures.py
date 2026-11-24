"""Draft paper figures for the v3 comparison (PDF and PNG in figures/paper/), drawn from reports/v3_tables/*.csv and the saved forecasts.

(a) skill vs lead; (b) per-gauge M0 - LSTM difference vs drainage area; (c) ablation ladder with the +-0.01 equivalence band; (d) KGE components;
(e) reach-level skill by size class. Models: M0, M1, M2, LSTM (area-scaled LSTM), NWM (reference). Pending runs are drawn as "not run".
Run: python scripts/v3_figures.py
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from v3_common import *  # noqa: F401,F403

FIG.mkdir(parents=True, exist_ok=True)
COL = {"M0": "#1b7f79", "M1": "#c9822b", "M2": "#7a5bb5", "M3": "#6b8e23", "M4": "#b05a8c", "LSTM": "#c8434f", "NWM": "#555555"}
PNAME = {"gauged": "Gauged (101 gauges)", "heldout": "Held-out, random folds (98)", "blocked": "Blocked basins (29)"}
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "legend.fontsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "pdf.fonttype": 42})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


lead = pd.read_csv(OUT / "lead_curves.csv")
paired = pd.read_csv(OUT / "paired_tests.csv")
comp = pd.read_csv(OUT / "kge_components.csv")

# ---- (a) skill vs lead ---------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(7.6, 5.0), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
for j, proto in enumerate(PROTOCOLS):
    d = lead[lead.protocol == proto]
    for m, ls in (("NWM", "--"), ("LSTM", "-"), ("M0", "-")):
        ax[0, j].plot(d.lead_h, d[f"median_nse_{m}"], ls, color=COL[m], lw=1.6, label={"M0": "M0", "LSTM": "LSTM", "NWM": "NWM (reference)"}[m])
    ax[0, j].set_title(PROTOCOLS and PNAME[proto])
    ax[0, j].set_ylim(-0.4, 1.0)
    ax[0, j].axhline(0, color="#999", lw=0.5)
    for m in ("M0", "M2"):
        ax[1, j].plot(d.lead_h, d[f"diff_median_{m}_minus_LSTM"], color=COL[m], lw=1.4, label=f"{m} - LSTM")
        ax[1, j].fill_between(d.lead_h, d[f"diff_{m}_minus_LSTM_ci95_lo"], d[f"diff_{m}_minus_LSTM_ci95_hi"], color=COL[m], alpha=0.15, lw=0)
    ax[1, j].axhline(0, color="#999", lw=0.8)
    ax[1, j].set_xlabel("Lead time (h)")
    ax[1, j].set_xticks([6, 48, 96, 144, 192, 240])
    ax[1, j].set_ylim(-0.2, 0.25)
ax[0, 0].set_ylabel("Median NSE over gauges")
ax[1, 0].set_ylabel("Difference in median NSE\n(95% gauge bootstrap)")
ax[0, 0].legend(frameon=False, loc="upper right")
ax[1, 0].legend(frameon=False, loc="lower left")
fig.suptitle("NWM forecasts nudge toward every USGS gauge, including those withheld from M0 and LSTM: reference, not a held-out competitor", fontsize=7.5, y=0.0, va="top")
save(fig, "fig_a_skill_vs_lead")

# ---- (b) per-gauge difference vs drainage area ---------------------------------------------------------------------------
ctx = Context()
has_up = np.array([len(ctx.upstream_gauges(g)) > 0 for g in range(ctx.gauges.count)])
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8), sharey=True)
for j, proto in enumerate(PROTOCOLS):
    d = load_protocol(proto, ("M0", "LSTM"))
    sc = {m: scores(*d[m]) for m in d}
    ok = np.all([np.isfinite(sc[m]["nse"][3]) for m in sc], axis=0)
    diff = (days15(sc["M0"]["nse"]) - days15(sc["LSTM"]["nse"]))[ok]
    a = ctx.area[ok]
    up = has_up[ok]
    ax[j].scatter(a[~up], diff[~up], s=12, color="#7a8a99", alpha=0.8, lw=0, label="no gauge upstream")
    ax[j].scatter(a[up], diff[up], s=14, facecolor="none", edgecolor=COL["M0"], lw=0.9, label="gauge(s) upstream")
    edges = np.array([0, 200, 600, 2000, 1e9])
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (a >= lo) & (a < hi)
        if sel.sum() >= 3:
            ax[j].plot([max(lo, 12), min(hi, 6000)], [np.median(diff[sel])] * 2, color="black", lw=2.0)
    ax[j].axhline(0, color="#999", lw=0.8)
    ax[j].set_xscale("log")
    ax[j].set_xlabel("Drainage area (km$^2$)")
    ax[j].set_title(PNAME[proto])
    ax[j].set_ylim(-0.5, 0.8)
ax[0].set_ylabel("M0 - LSTM, days 1-5 NSE\n(per gauge)")
ax[0].legend(frameon=False, loc="lower left", fontsize=7)
fig.text(0.5, -0.08, "Black bars: median per area bin (<200, 200-600, 600-2,000, >2,000 km$^2$).", ha="center", fontsize=7.5)
save(fig, "fig_b_gap_vs_area")

# ---- (c) component effects: insertion (M0 - M1), message passing (M0 - M2), routing (M1 - M4) --------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
comps = [("insertion\n(M0 - M1)", "M0", "M1", COL["M1"]), ("message passing\n(M0 - M2)", "M0", "M2", COL["M2"]), ("routing\n(M1 - M4)", "M1", "M4", COL["M4"])]
for j, proto in enumerate(("heldout", "blocked")):
    ax[j].axhspan(-0.01, 0.01, color="#cfd8dc", alpha=0.8, lw=0, label="$\\pm$0.01 equivalence band")
    ax[j].axhline(0, color="#555", lw=0.8)
    for i, (lab, a_, b_, col) in enumerate(comps):
        r = paired[(paired.protocol == proto) & (paired.model_a == a_) & (paired.model_b == b_)]
        if len(r) and np.isfinite(r.mean_diff.iloc[0]):
            r = r.iloc[0]
            ax[j].errorbar(i, r.mean_diff, yerr=[[r.mean_diff - r.boot90_lo], [r.boot90_hi - r.mean_diff]], fmt="o", color=col, capsize=3, ms=5)
        else:
            note = "cannot act:\nno visible\nupstream gauge" if (proto == "blocked" and b_ == "M1" and a_ == "M0") else "not run"
            ax[j].text(i, 0.05, note, ha="center", va="center", fontsize=7, color="#888")
    ax[j].set_xticks(range(len(comps)))
    ax[j].set_xticklabels([c[0] for c in comps], fontsize=7)
    ax[j].set_xlim(-0.5, len(comps) - 0.5)
    ax[j].set_title(PNAME[proto])
    ax[j].set_ylim(-0.1, 0.1)
ax[0].set_ylabel("Effect of the component, days 1-5 NSE\n(mean paired difference, 90% gauge bootstrap)")
ax[0].legend(frameon=False, loc="lower left", fontsize=7)
save(fig, "fig_c_component_effects")

# ---- (d) KGE components ---------------------------------------------------------------------------------------------------
c = comp[comp.lead == "days 1-5"]
fig, ax = plt.subplots(1, 4, figsize=(8.2, 2.8))
fig.subplots_adjust(wspace=0.45)
models = ["M0", "M1", "M2", "LSTM", "NWM"]
for k, (col, ttl) in enumerate((("median_kge", "KGE"), ("median_r", "Correlation r"), ("median_alpha", "Variability ratio $\\alpha$"), ("median_beta", "Bias ratio $\\beta$"))):
    for p_i, proto in enumerate(PROTOCOLS):
        for m_i, m in enumerate(models):
            row = c[(c.protocol == proto) & (c.model == m)]
            if len(row):
                ax[k].bar(p_i + (m_i - 2) * 0.15, row[col].iloc[0], 0.14, color=COL[m], label=m if p_i == 0 else None)
    ax[k].set_xticks(range(3))
    ax[k].set_xticklabels(["gauged", "held-\nout", "blocked"], fontsize=6.5)
    ax[k].set_title(ttl)
    if col in ("median_alpha", "median_beta"):
        ax[k].axhline(1, color="black", lw=0.8)
    ax[k].set_ylim(0, 1.05 if col != "median_alpha" else 1.05)
ax[0].legend(frameon=False, fontsize=7, loc="upper right", ncol=1)
fig.text(0.5, -0.04, "Median over gauges, mean over the leads 24-120 h. Ideal values: KGE, r, $\\alpha$, $\\beta$ = 1.", ha="center", fontsize=7.5)
save(fig, "fig_d_kge_components")

# ---- (e) reach level by size class ----------------------------------------------------------------------------------------
from tva_flood.hydro.evaluate import reach_table  # noqa: E402

net = ctx.network
m0r = np.load(A / "eval_M0_reach" / "reach_metrics.npz", allow_pickle=True)
lst = np.load(A / "ladder_L4_s1_reaches" / "metrics.npz", allow_pickle=True)
table, _ = reach_table({"LSTM": lst["nse/gefs"]}, m0r["reach_fold"], net, ctx.regulated)
m0csv = pd.read_csv(A / "eval_M0_reach" / "reaches.csv")  # only its M0 and NWM columns are used
reach = table.merge(m0csv[["lead_h", "scope", "M0+gefs", "NWM forecast"]].rename(columns={"M0+gefs": "M0", "NWM forecast": "NWM"}), on=["lead_h", "scope"])
reach["M1"] = reach["M2"] = reach["M4"] = np.nan  # not run at the reach level (none planned)
reach.to_csv(OUT / "reach_level.csv", index=False)
scopes = ["held-out basins, < 50 km2", "held-out basins, 50-500", "held-out basins, 500-5,000", "held-out basins, > 5,000"]
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8), sharey=True)
for j, h in enumerate((24, 72, 120)):
    d = reach[reach.lead_h == h].set_index("scope")
    for m_i, m in enumerate(("M0", "LSTM", "NWM")):
        ax[j].bar(np.arange(4) + (m_i - 1) * 0.26, [d.loc[s_, m] for s_ in scopes], 0.25, color=COL[m], label={"NWM": "NWM (reference)"}.get(m, m))
    ax[j].set_xticks(range(4))
    ax[j].set_xticklabels(["<50", "50-500", "500-\n5,000", ">5,000"], fontsize=7.5)
    ax[j].set_title(f"{h} h lead")
    ax[j].axhline(0, color="#555", lw=0.6)
    ax[j].set_ylim(-1.0, 0.8)
ax[0].set_ylabel("Median NSE against the\nNWM retrospective")
ax[0].legend(frameon=False, fontsize=7, loc="lower left")
fig.supxlabel("Basin size class (km$^2$), held-out basins", y=-0.12, fontsize=8.5)
fig.text(0.5, -0.24, "Consistency with the NWM retrospective, not skill against observations. >5,000 km$^2$: 281 reaches; bars are clipped at -1 (LSTM, 24 h, >5,000 km$^2$: -3.28).",
         ha="center", fontsize=7)
save(fig, "fig_e_reach_level")
