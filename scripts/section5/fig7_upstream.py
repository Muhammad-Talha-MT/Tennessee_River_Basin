"""Figure 7: upstream observations (RQ3). M0 - M1 difference in NSE against lead at nested gauges (a catalog gauge upstream that the scoring model may see,
so insertion can act) and headwater gauges; line = mean over gauges, band = 90% gauge bootstrap; (a) gauged, (b) random hold-out (the blocked protocol has no
nested gauge). Reads reports/section5/s5_5a_position_by_lead.csv.
Run: python scripts/section5/fig7_upstream.py   Writes figures/fig7_upstream.{pdf,png}.
"""
import matplotlib.pyplot as plt
from style import HEADWATER, NESTED, PNAME, read, save

pl = read("s5_5a_position_by_lead.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
fig.subplots_adjust(left=0.09, right=0.975, bottom=0.17, top=0.91, wspace=0.06)
for k, (ax, p) in enumerate(zip(axes, ("gauged", "heldout"))):
    ax.axhline(0, color="#000000", lw=0.8)
    for grp, col, lab in (("nested", NESTED, "Nested"), ("headwater", HEADWATER, "Headwater")):
        d = pl[(pl.protocol == p) & (pl.group == grp)].sort_values("lead_h")
        ax.fill_between(d.lead_h, d.ci5, d.ci95, color=col, alpha=0.15, lw=0)
        ax.plot(d.lead_h, d.mean_diff, color=col, lw=1.5, label=f"{lab} (n = {int(d.n.max())})")
    ax.set_title(f"({'ab'[k]}) {PNAME[p]}")
    ax.set_xlim(6, 240)
    ax.set_xticks([6, 48, 96, 144, 192, 240])
    ax.set_xlabel("Lead time (h)")
    ax.legend(loc="lower right")
axes[0].set_ylabel("Difference in NSE, GNN - GNN-noIns")
axes[0].set_ylim(-0.065, 0.075)
axes[0].set_yticks([-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06])
save(fig, "fig7_upstream", list(axes))
