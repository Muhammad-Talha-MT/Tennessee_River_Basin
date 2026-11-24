"""Figure S4 (exploratory): per nested gauge, the M0 - M1 gain at days 1-2 (leads 24, 48 h) against the area ratio of the nearest visible upstream gauge
(largest upstream area) to the gauge; random hold-out filled, gauged open; least-squares lines (random solid, gauged dashed) with Spearman rho.
Reads reports/section5/s5_5d_gain_vs_ratio.csv, s5_5d_rank_correlations.csv.
Run: python scripts/section5/figS4_area_ratio.py   Writes figures/figS4_area_ratio.{pdf,png}.
"""
import matplotlib.pyplot as plt
import numpy as np
from style import NESTED, read, save

gain = read("s5_5d_gain_vs_ratio.csv")
rho = read("s5_5d_rank_correlations.csv")
fig, ax = plt.subplots(figsize=(3.5, 3.0))
fig.subplots_adjust(left=0.2, right=0.96, bottom=0.16, top=0.95)
ax.axhline(0, color="#000000", lw=0.8)
for prot, face, ls, y_lab, short in (("Gauged", "white", "--", -0.355, "gauged"), ("Random hold-out", NESTED, "-", -0.30, "random")):
    d = gain[gain.protocol == prot].dropna(subset=["area_ratio", "gain_days1_2"])
    ax.scatter(d.area_ratio, d.gain_days1_2, s=18, facecolor=face, edgecolor=NESTED, lw=0.9, zorder=3)
    b1, b0 = np.polyfit(d.area_ratio, d.gain_days1_2, 1)
    ax.plot([0, 1], [b0, b0 + b1], ls, color="#000000", lw=1.0, zorder=2)
    r = rho[(rho.protocol == prot) & (rho.variable == "area_ratio")].iloc[0]
    ax.text(0.02, y_lab, f"{short}: rho = {r.spearman_rho:+.2f} (n = {int(r.n)})", ha="left", va="center")
ax.set_xlim(0, 1)
ax.set_ylim(-0.4, 0.17)
ax.set_yticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1])
ax.set_xlabel("Area ratio, upstream gauge / gauge")
ax.set_ylabel("GNN - GNN-noIns, NSE days 1-2")
save(fig, "figS4_area_ratio", [ax])
