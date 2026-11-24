"""Figure 6: where the differences occur. Per-gauge difference in days 1-5 NSE in the random hold-out (n = 98), one panel per contrast (M0 - LSTM,
M0 - M2 message passing, M1 - M4 routing, M0 - M1 insertion), on the basin with its major rivers; red = first model better, blue = second model
better; one shared diverging scale clipped at +-0.15. Reads reports/section5/s5_pergauge_nse.csv (per-gauge days 1-5 NSE written by s5_facts.py); the
basin outline and rivers are the study-area geometry (NHDPlus V2.1, HUC2 06). Run: python scripts/section5/fig6_map_contrasts.py
Writes figures/fig6_map_contrasts.{pdf,png}.
"""
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "section3"))
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from style import DIVERGING, pretty, read, save

from common import INPUTS  # noqa: E402  (section 3 data paths only)

CRS = "EPSG:5070"
pg = read("s5_pergauge_nse.csv")
pg = pg[pg.protocol == "heldout"].copy()
PANELS = [("M0", "LSTM", "(a) M0 - LSTM"), ("M0", "M2", "(b) M0 - M2 (message passing)"),
          ("M1", "M4", "(c) M1 - M4 (routing)"), ("M0", "M1", "(d) M0 - M1 (insertion)")]
outline = gpd.read_file(INPUTS["huc2"]).to_crs(CRS)
nodes = pd.read_parquet(INPUTS["nodes"], columns=["comid", "totdasqkm"]).set_index("comid")
lines = gpd.read_file(INPUTS["flowlines"], columns=["COMID"])
lines = lines[lines.COMID.isin(nodes.index)]
lines["a"] = nodes.totdasqkm.reindex(lines.COMID).to_numpy()
lines = lines[lines.a >= 1000].to_crs(CRS)
pts = gpd.GeoSeries(gpd.points_from_xy(pg.lon, pg.lat), crs="EPSG:4326").to_crs(CRS)
pg["x"], pg["y"] = pts.x.to_numpy(), pts.y.to_numpy()
x0, y0, x1, y1 = outline.total_bounds
pad = 0.02 * (x1 - x0)
LIM = 0.15
norm = TwoSlopeNorm(vmin=-LIM, vcenter=0, vmax=LIM)
cmap = plt.get_cmap(DIVERGING)
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.25))
fig.subplots_adjust(left=0.005, right=0.995, bottom=0.165, top=0.955, wspace=0.02, hspace=0.14)
for ax, (a, b, title) in zip(axes.flat, PANELS):
    outline.plot(ax=ax, color="#F2F2F2", edgecolor="#4D4D4D", lw=0.7)
    lines.plot(ax=ax, color="#9DB9D3", lw=0.5 + 1.2 * (lines.a >= 10000))
    if f"nse_{a}" in pg and f"nse_{b}" in pg and pg[f"nse_{b}"].notna().any():
        d = (pg[f"nse_{a}"] - pg[f"nse_{b}"]).to_numpy()
        ok = np.isfinite(d)
        order = np.argsort(np.abs(d[ok]))  # largest differences drawn on top
        sub = pg[ok].iloc[order]
        ax.scatter(sub.x, sub.y, c=np.clip(d[ok][order], -LIM, LIM), cmap=cmap, norm=norm, s=22, edgecolor="#000000", lw=0.4, zorder=3)
        ax.text(0.985, 0.04, f"{int((d[ok] > 0).sum())} of {int(ok.sum())} gauges > 0", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5)
    ax.set_title(pretty(title), loc="left", fontsize=8.5)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
cax = fig.add_axes([0.3, 0.085, 0.4, 0.028])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="both")
cb.set_ticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])
cb.set_label("Difference in NSE, days 1-5", labelpad=2)
fig.text(0.28, 0.099, "second model better", ha="right", va="center", fontsize=7.5)
fig.text(0.72, 0.099, "first model better", ha="left", va="center", fontsize=7.5)
cb.ax.grid(False)
save(fig, "fig6_map_contrasts", list(axes.flat), lines=False)
