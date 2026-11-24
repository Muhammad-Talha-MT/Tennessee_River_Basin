"""Figure S5 (grid): reach-level maps of the NSE difference M0 - LSTM against the NWM v3.0 retrospective. Rows: reaches by drainage area (>= 5,000, >= 500,
>= 50, >= 5 km2, all). Columns: lead 6 h, 12 h, 18 h, and the mean over days 1-5 (leads 24-120 h). Purple: M0 higher; orange: LSTM higher; scale clipped at +-0.5;
light gray: not scored. Line width grows with drainage area. Reads reports/section5/s5_perreach_nse.csv and s5_perreach_nse_leads.csv; geometry is the
study-area flowlines and basin outline (NHDPlus V2.1, HUC2 06). Reference = retrospective on which the reach-scale models were pretrained.
Run: python scripts/section5/figS5_reach_grid.py   Writes figures/figS5_reach_grid.{pdf,png}.
"""
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "section3"))
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from style import DIVERGING, FIG, read

from common import INPUTS  # noqa: E402

CRS, LIM = "EPSG:5070", 0.5
a = read("s5_perreach_nse.csv").set_index("comid")
b = read("s5_perreach_nse_leads.csv").set_index("comid")
diff = {"6 h": b.nse_M0_6h - b.nse_LSTM_6h, "12 h": b.nse_M0_12h - b.nse_LSTM_12h, "18 h": b.nse_M0_18h - b.nse_LSTM_18h, "Days 1-5": a.nse_M0 - a.nse_LSTM}
lines = gpd.read_file(INPUTS["flowlines"], columns=["COMID"])
lines = lines[lines.COMID.isin(a.index)].to_crs(CRS)
lines["area_km2"] = a.area_km2.reindex(lines.COMID).to_numpy()
for k, v in diff.items():
    lines[k] = v.reindex(lines.COMID).to_numpy()
lines = lines.sort_values("area_km2")
outline = gpd.read_file(INPUTS["huc2"]).to_crs(CRS)
x0, y0, x1, y1 = outline.total_bounds
pad = 0.015 * (x1 - x0)
norm, cmap = Normalize(-LIM, LIM), plt.get_cmap(DIVERGING)
ROWS = [(5000, "$\\geq$ 5,000 km$^2$"), (500, "$\\geq$ 500 km$^2$"), (50, "$\\geq$ 50 km$^2$"), (5, "$\\geq$ 5 km$^2$"), (0, "All reaches")]
W = 7.2
aspect = (y1 - y0 + 2 * pad) / (x1 - x0 + 2 * pad)
cw = 1.55  # panel width (in)
fig = plt.figure(figsize=(W, 0.35 + len(ROWS) * cw * aspect + 0.95))
H = fig.get_figheight()
left = 0.95
colx = [(left + i * cw) / W for i in range(4)]
med = {}
for r, (thr, rlab) in enumerate(ROWS):
    sub = lines[lines.area_km2 >= thr]
    lw = 0.12 + 0.28 * np.log10(np.maximum(sub.area_km2.to_numpy(), 1.0))
    for c, k in enumerate(diff):
        pw = 0.95 * cw  # panel width (in); the rest is the gap between boxed panels
        ax = fig.add_axes([colx[c] + (cw - pw) / 2 / W, (0.95 + (len(ROWS) - 1 - r) * cw * aspect + (cw - pw) * aspect / 2) / H, pw / W, pw * aspect / H])
        outline.plot(ax=ax, facecolor="#FAFAFA", edgecolor="#1F2937", lw=0.5, zorder=0)
        ok = sub[k].notna().to_numpy()
        if (~ok).any():
            sub[~ok].plot(ax=ax, color="#D4D4D4", linewidth=lw[~ok], zorder=1, rasterized=True)
        sub[ok].plot(ax=ax, color=cmap(norm(np.clip(sub[k].to_numpy(float)[ok], -LIM, LIM))), linewidth=lw[ok], zorder=2, rasterized=True)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.8)
            sp.set_edgecolor("#374151")
        med[(rlab, k)] = float(np.nanmedian(sub[k]))
        ax.text(0.99, 0.02, f"{med[(rlab, k)]:+.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
        if r == 0:
            ax.set_title(k if k != "Days 1-5" else "Days 1-5", fontsize=9, pad=2)
        if c == 0:
            ax.text(-0.04, 0.5, f"{rlab}\n({len(sub):,})", transform=ax.transAxes, ha="right", va="center", fontsize=7.5)
cax = fig.add_axes([0.3, 0.5 / H, 0.4, 0.14 / H])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="both")
cb.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
cb.set_label("Difference in NSE (GNN - LSTM); numbers: median", labelpad=2, fontsize=8)
cb.ax.grid(False)
fig.text(0.28, 0.57 / H, "LSTM higher", ha="right", va="center", fontsize=7.5)
fig.text(0.72, 0.57 / H, "GNN higher", ha="left", va="center", fontsize=7.5)
for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
    fig.savefig(FIG / f"figS5_reach_grid.{ext}", facecolor="white", **kw)
for k, v in med.items():
    print(k, f"{v:+.3f}")
print("size", fig.get_size_inches())
