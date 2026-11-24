"""Figure 5: paired difference in days 1-5 NSE (first minus second model), mean over gauges with a 90% (5-95%) gauge-bootstrap interval, grouped by research
question; one marker per protocol; shaded band = +-0.02 equivalence margin. Reads reports/section5/s5_3_contrasts.csv.
Run: python scripts/section5/fig5_contrasts.py   Writes figures/fig5_contrasts.{pdf,png}.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from style import PLONG, pretty, read, save

con = read("s5_3_contrasts.csv")
ROWS = [("RQ1", "M0 - LSTM", "network"), ("RQ2", "M0 - M2", "message passing"), ("RQ2", "M1 - M4", "routing"), ("RQ3", "M0 - M1", "insertion")]
PROT = {"gauged": ("o", "#000000", "Gauged"), "heldout": ("s", "#1F78B4", "Random hold-out"), "blocked": ("^", "#C62828", "Blocked basins")}
OFF = {"gauged": -0.22, "heldout": 0.0, "blocked": 0.22}
have = con[con.contrast.isin([r[1] for r in ROWS]) & (con.n > 0)]
lo, hi = min(-0.03, have.ci5.min() - 0.01), max(0.03, have.ci95.max() + 0.01)
fig, ax = plt.subplots(figsize=(3.5, 3.4))
fig.subplots_adjust(left=0.37, right=0.96, bottom=0.25, top=0.97)
ax.axvspan(-0.02, 0.02, color="#E8E8E8", lw=0, zorder=0)
ax.axvline(0, color="#000000", lw=0.8, zorder=1)
for i, (rq, cn, what) in enumerate(ROWS):
    rows = con[(con.contrast == cn) & (con.n > 0)]
    for r in rows.itertuples():
        mk, col, _ = PROT[PLONG[r.protocol]]
        y = i + OFF[PLONG[r.protocol]]
        ax.errorbar(r.mean_diff, y, xerr=[[r.mean_diff - r.ci5], [r.ci95 - r.mean_diff]], fmt=mk, color=col, ms=5, mfc=col, mec=col, elinewidth=1.1, capsize=2.5, zorder=3)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([(pretty(cn) if len(pretty(cn)) < 20 else pretty(cn).replace(" - ", " -\n")) + "\n(" + what + ")" for rq, cn, what in ROWS])
ax.set_ylim(len(ROWS) - 0.5, -0.5)
ax.grid(axis="y", visible=False)
for i in range(1, len(ROWS)):
    if ROWS[i][0] != ROWS[i - 1][0]:
        ax.axhline(i - 0.5, color="#AAAAAA", lw=0.6)
ax.set_xlim(lo, hi)
ax.set_xticks([v for v in (-0.05, 0.0, 0.05, 0.10, 0.15) if lo <= v <= hi])
ax.set_xlabel("Difference in NSE (days 1-5)")
handles = [Line2D([], [], marker=m, color=c, ls="", ms=5, label=l) for m, c, l in PROT.values()]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.55, 0.0), columnspacing=0.6, handletextpad=0.2, fontsize=7)
save(fig, "fig5_contrasts", [ax])
