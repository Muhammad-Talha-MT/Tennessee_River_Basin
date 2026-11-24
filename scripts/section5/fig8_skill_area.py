"""Figure 8: skill by drainage area for the GNN (M0), the LSTM and the NWM forecast (ablations are shown elsewhere). One x-axis (drainage-area class) in all three
panels. (a) median NSE (days 1-5) at every reach against the NWM v3.0 retrospective; (b) share of those reaches with NSE > 0; (c) median NSE (days 1-5) against
observed discharge at the evaluation gauges of the random hold-out. The retrospective is the reference on which the GNN was pretrained and the LSTM was not;
number of scored reaches / gauges under each class. Reads reports/section5/s5_7_every_reach.csv, s5_5g_skill_by_area.csv.
Run: python scripts/section5/fig8_skill_area.py   Writes figures/fig8_skill_area.{pdf,png}.
"""
import matplotlib.pyplot as plt
import numpy as np
from style import COLOR, STYLE, WIDTH, read, save

t = read("s5_7_every_reach.csv")
t = t[t.scope == "all reaches"].reset_index(drop=True)
g = read("s5_5g_skill_by_area.csv")
g = g[g.protocol == "Random hold-out"].reset_index(drop=True)
NAME = {"M0": "GNN", "LSTM": "LSTM", "NWM": "NWM"}
SHORT = {"< 5": "<5", "5-50": "5-50", "50-500": "50-500", "500-5,000": "0.5-5k", "> 5,000": ">5k", "< 50": "<50", "50-200": "50-200", "200-600": "200-600",
         "600-2,000": "0.6-2k", "> 2,000": ">2k"}


def tick(area, n):
    """Tick label: the class (a range breaks after its en dash) and the count of reaches or gauges."""
    cls = SHORT[area]
    return cls + "\n(" + kfmt(n) + ")"


def kfmt(n):
    """Counts of 1,000 or more as 25.2k, smaller counts as they are."""
    return f"{n / 1000:.1f}k" if n >= 1000 else f"{int(n)}"


fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
fig.subplots_adjust(left=0.075, right=0.985, bottom=0.30, top=0.90, wspace=0.5)
axa, axb, axc = axes
xa, xc = np.arange(len(t)), np.arange(len(g))
for m in ("NWM", "LSTM", "M0"):
    kw = dict(ls=STYLE[m], marker="o", ms=3.8, color=COLOR[m], lw=WIDTH[m], label=NAME[m], zorder=3)
    axa.plot(xa, t[f"{m}_median_nse"].astype(float), **kw)
    axb.plot(xa, t[f"{m}_share_nse_above_0"].astype(float), **kw)
    axc.plot(xc, g[f"{m}_median_nse"].astype(float), **kw)
for ax in (axa, axc):
    ax.axhline(0, color="black", lw=0.8, zorder=2)
axa.set_ylim(-2.0, 0.6)
axa.set_yticks([-2, -1, 0])
axb.set_ylim(0, 1.0)
axb.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axc.set_ylim(-0.5, 0.6)
axc.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
for ax, x, labs, ns in ((axa, xa, t.area_km2, t.M0_scored), (axb, xa, t.area_km2, t.M0_scored), (axc, xc, g.area_km2, g.n)):
    ax.set_xticks(x)
    ax.set_xticklabels([tick(a, n) for a, n in zip(labs, ns)], fontsize=6)
    ax.set_xlim(-0.35, len(x) - 0.65)
    ax.set_xlabel("Drainage area (km$^2$)")
axa.set_ylabel("Median NSE")
axb.set_ylabel("Share with NSE > 0")
axc.set_ylabel("Median NSE")
axa.set_title("(a) Every reach", fontsize=9)
axb.set_title("(b) Every reach", fontsize=9)
axc.set_title("(c) Evaluation gauges", fontsize=9)
handles, labels = axa.get_legend_handles_labels()
order = [labels.index(n) for n in ("GNN", "LSTM", "NWM")]
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=3, bbox_to_anchor=(0.53, 0.0), columnspacing=1.8, handlelength=2.6)
save(fig, "fig8_skill_area", list(axes))
