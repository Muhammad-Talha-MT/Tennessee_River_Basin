"""Figure S5 (a-e): reach-level maps of the difference in days 1-5 NSE between M0 and the LSTM against the NWM v3.0 retrospective, for every reach, shown in
five steps by drainage area: (a) reaches of at least 5,000 km2, (b) 500, (c) 50, (d) 5 km2, (e) all reaches. Purple: M0 higher; orange: LSTM higher; the color scale
is clipped at +-0.5 (ends marked by arrows); light gray: reaches not scored (fewer than 100 values or a constant reference). Line width grows with
drainage area. The reference is the retrospective on which the reach-scale models were pretrained. Reads reports/section5/s5_perreach_nse.csv; the
flowlines and basin outline are the study-area geometry (NHDPlus V2.1, HUC2 06).
Run: python scripts/section5/figS5_reach_maps.py   Writes figures/figS5{a..e}_reach_map.{pdf,png}.
"""
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "section3"))
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from style import DIVERGING, FIG, read

from common import INPUTS  # noqa: E402  (study-area data paths only)

CRS = "EPSG:5070"
pr = read("s5_perreach_nse.csv").set_index("comid")
lines = gpd.read_file(INPUTS["flowlines"], columns=["COMID"])
lines = lines[lines.COMID.isin(pr.index)].to_crs(CRS)
lines["area_km2"] = pr.area_km2.reindex(lines.COMID).to_numpy()
lines["diff"] = (pr.nse_M0 - pr.nse_LSTM).reindex(lines.COMID).to_numpy()
lines = lines.sort_values("area_km2")  # small streams first, large on top
outline = gpd.read_file(INPUTS["huc2"]).to_crs(CRS)
x0, y0, x1, y1 = outline.total_bounds
pad = 0.015 * (x1 - x0)
LIM = 0.5
norm = Normalize(vmin=-LIM, vmax=LIM)
cmap = plt.get_cmap(DIVERGING)
STEPS = [("a", 5000, "reaches $\\geq$ 5,000 km$^2$"), ("b", 500, "reaches $\\geq$ 500 km$^2$"), ("c", 50, "reaches $\\geq$ 50 km$^2$"),
         ("d", 5, "reaches $\\geq$ 5 km$^2$"), ("e", 0, "all reaches")]
for tag, thr, label in STEPS:
    sub = lines[lines["area_km2"] >= thr]
    ok = sub["diff"].notna()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.15, top=0.97)
    outline.plot(ax=ax, facecolor="#FAFAFA", edgecolor="#1F2937", lw=1.0, zorder=0)
    lw = 0.2 + 0.42 * np.log10(np.maximum(sub["area_km2"].to_numpy(), 1.0))
    if (~ok).any():
        sub[~ok].plot(ax=ax, color="#D4D4D4", linewidth=lw[~ok.to_numpy()], zorder=1, rasterized=True)
    col = np.clip(sub["diff"].to_numpy(float), -LIM, LIM)
    sub[ok].plot(ax=ax, color=cmap(norm(col[ok.to_numpy()])), linewidth=lw[ok.to_numpy()], zorder=2, rasterized=True)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.text(0.01, 0.97, f"({tag}) {label}: {len(sub):,} reaches, {int(ok.sum()):,} scored", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    cax = fig.add_axes([0.3, 0.075, 0.4, 0.028])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="both")
    cb.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
    cb.set_label("Difference in NSE, days 1-5 (GNN - LSTM)", labelpad=2)
    cb.ax.grid(False)
    fig.text(0.28, 0.089, "LSTM higher", ha="right", va="center", fontsize=8)
    fig.text(0.72, 0.089, "GNN higher", ha="left", va="center", fontsize=8)
    med = float(np.nanmedian(sub["diff"]))
    ax.text(0.99, 0.03, f"median {med:+.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
        fig.savefig(FIG / f"figS5{tag}_reach_map.{ext}", facecolor="white", **kw)
    plt.close(fig)
    print(f"figS5{tag}: {len(sub):,} reaches, {int(ok.sum()):,} scored, median {med:+.3f}")
