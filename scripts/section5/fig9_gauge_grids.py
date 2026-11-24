"""Figures 9 and 10: per-gauge NSE differences by lead time, on the basin with its major rivers.
Figure 9: random hold-out (n = 98): rows = the four contrasts of Figure 6 (M0 - LSTM, M0 - M2, M1 - M4, M0 - M1; no research-question labels on the figure); columns = lead 6 h, 12 h, 18 h and days 1-5.
Figure 10: M0 - LSTM, rows = the three protocols (gauged n = 101, random hold-out n = 98, blocked n = 29); same columns.
Purple = first model higher, orange = second; one scale clipped at +-0.3 (single-lead differences vary more than the days 1-5 mean of Figure 6); the number in
a panel is gauges > 0 of gauges scored. Reads reports/section5/s5_pergauge_nse.csv and s5_pergauge_nse_leads.csv; geometry is the study area (NHDPlus V2.1, HUC2 06).
Run: python scripts/section5/fig9_gauge_grids.py   Writes figures/fig9_gauge_contrast_leads.{pdf,png} and fig10_gauge_protocol_leads.{pdf,png}.
"""
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "section3"))
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from style import DIVERGING, FIG, pretty, read

from common import INPUTS  # noqa: E402

CRS, LIM = "EPSG:5070", 0.3
day, lead = read("s5_pergauge_nse.csv"), read("s5_pergauge_nse_leads.csv")
outline = gpd.read_file(INPUTS["huc2"]).to_crs(CRS)
nodes = pd.read_parquet(INPUTS["nodes"], columns=["comid", "totdasqkm"]).set_index("comid")
lines = gpd.read_file(INPUTS["flowlines"], columns=["COMID"])
lines = lines[lines.COMID.isin(nodes.index)]
lines["a"] = nodes.totdasqkm.reindex(lines.COMID).to_numpy()
lines = lines[lines.a >= 1000].to_crs(CRS)
x0, y0, x1, y1 = outline.total_bounds
pad = 0.02 * (x1 - x0)
aspect = (y1 - y0 + 2 * pad) / (x1 - x0 + 2 * pad)
norm, cmap = TwoSlopeNorm(vmin=-LIM, vcenter=0, vmax=LIM), plt.get_cmap(DIVERGING)
COLS = ["6 h", "12 h", "18 h", "Days 1-5"]
for t in (day, lead):
    pts = gpd.GeoSeries(gpd.points_from_xy(t.lon, t.lat), crs="EPSG:4326").to_crs(CRS)
    t["x"], t["y"] = pts.x.to_numpy(), pts.y.to_numpy()


def diff(protocol, a, b, c):
    """Per-gauge difference a - b at column c, for the gauges of one protocol (NaN where either model is not scored)."""
    if c == "Days 1-5":
        t = day[day.protocol == protocol]
        return t, (t[f"nse_{a}"] - t[f"nse_{b}"]).to_numpy()
    h = c.replace(" ", "")
    t = lead[lead.protocol == protocol]
    return t, (t[f"nse_{a}_{h}"] - t[f"nse_{b}_{h}"]).to_numpy()


def grid(rows, name, label_fn):
    """rows: list of (protocol, first, second, row label)."""
    W, cw, left = 7.2, 1.61, 0.7
    pw = 0.95 * cw
    fig = plt.figure(figsize=(W, 0.35 + len(rows) * cw * aspect + 0.95))
    H = fig.get_figheight()
    for r, (protocol, a, b, rlab) in enumerate(rows):
        for c, col in enumerate(COLS):
            ax = fig.add_axes([(left + c * cw + (cw - pw) / 2) / W, (0.95 + (len(rows) - 1 - r) * cw * aspect + (cw - pw) * aspect / 2) / H, pw / W, pw * aspect / H])
            outline.plot(ax=ax, color="#F2F2F2", edgecolor="#4D4D4D", lw=0.4)
            lines.plot(ax=ax, color="#9DB9D3", lw=0.3 + 0.7 * (lines.a >= 10000))
            t, d = diff(protocol, a, b, col)
            ok = np.isfinite(d)
            if ok.any():
                order = np.argsort(np.abs(d[ok]))  # largest differences on top
                sub = t[ok].iloc[order]
                ax.scatter(sub.x, sub.y, c=np.clip(d[ok][order], -LIM, LIM), cmap=cmap, norm=norm, s=9, edgecolor="#000000", lw=0.25, zorder=3)
                ax.text(0.985, 0.03, f"{int((d[ok] > 0).sum())}/{int(ok.sum())}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
            ax.set_xlim(x0 - pad, x1 + pad)
            ax.set_ylim(y0 - pad, y1 + pad)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_linewidth(0.8)
                sp.set_edgecolor("#374151")
            if r == 0:
                ax.set_title(col, fontsize=9, pad=2)
            if c == 0:
                ax.text(-0.05, 0.5, pretty(rlab), transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=7.5, linespacing=1.15)
    cax = fig.add_axes([0.3, 0.5 / H, 0.4, 0.14 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="both")
    cb.set_ticks([-0.3, -0.15, 0, 0.15, 0.3])
    cb.set_label("Difference in NSE; numbers: gauges > 0 of gauges scored", labelpad=2, fontsize=8)
    cb.ax.grid(False)
    fig.text(0.28, 0.57 / H, "second model higher", ha="right", va="center", fontsize=7.5)
    fig.text(0.72, 0.57 / H, "first model higher", ha="left", va="center", fontsize=7.5)
    for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
        fig.savefig(FIG / f"{name}.{ext}", facecolor="white", **kw)
    print(name, "saved", fig.get_size_inches())


grid([("heldout", "M0", "LSTM", "M0 -\nLSTM"), ("heldout", "M0", "M2", "M0 -\nM2\n(message\npassing)"),
      ("heldout", "M1", "M4", "M1 -\nM4\n(routing)"), ("heldout", "M0", "M1", "M0 -\nM1\n(insertion)")], "fig9_gauge_contrast_leads", None)
grid([("gauged", "M0", "LSTM", "Gauged\n(n = 101)"), ("heldout", "M0", "LSTM", "Random hold-out\n(n = 98)"),
      ("blocked", "M0", "LSTM", "Blocked basins\n(n = 29)")], "fig10_gauge_protocol_leads", None)
