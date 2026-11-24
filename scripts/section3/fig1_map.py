"""Figure 1: study area (full width, 17.4 cm). (a) all 58,834 reaches (line width by drainage area), state boundaries and names, the 101 evaluation gauges,
CONUS inset with state boundaries; (b) the same base map with the three blocked basins shaded in three hues and their gauges in the basin's hue, other gauges gray.
Dam locations are not public and are not drawn. One gauge marker shape; counts are in figures/section3/README.md, not in the legends.
Run: python scripts/section3/fig1_map.py. Inputs (common.INPUTS): NHDPlus V2.1 flowlines, catchments_clean.parquet, tennessee_huc2.geojson, cb_2022 states,
nodes.parquet, gauges_public, and the saved evaluations (which gauges are scored).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
import shapely
from common import FIG, INPUTS, load
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CRS = "EPSG:5070"
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
NAVY, BASIN_FILL, RIVER, STATE_LINE, STATE_TEXT = "#14213D", "#E9F0FB", "#6FA0DC", "#8C8C8C", "#5E5E5E"
GAUGE, GRAY = "#2B7BE0", "#8A8A8A"
BASIN = {1: "#E8663A", 2: "#2A9D8F", 3: "#7B4FA3"}  # orange, teal, purple: distinct hues and luminance, with the blue/orange scheme of the reference map
FIG.mkdir(parents=True, exist_ok=True)
c = load()
g, ids = c.gauges, c.network.ids

lines = gpd.read_file(INPUTS["flowlines"], columns=["COMID"])
lines = lines[lines.COMID.isin(ids)].set_index("COMID").to_crs(CRS)
lines["a"] = c.nodes.totdasqkm.reindex(lines.index).to_numpy()
outline = gpd.read_file(INPUTS["huc2"]).to_crs(CRS)
states = gpd.read_file(INPUTS["states"]).to_crs(CRS)
conus = states[~states.STUSPS.isin(["AK", "HI", "PR", "VI", "GU", "MP", "AS"])]
catch = gpd.read_parquet(INPUTS["catchments"]).set_index("comid")


def outlet(comid):
    geom = lines.geometry.get(comid)
    geom = geom.geoms[-1] if geom.geom_type == "MultiLineString" else geom
    return geom.coords[-1][:2]


gxy = np.array([outlet(ids[r]) for r in g.reach])
blocks = []
for k, (o, basin) in enumerate(c.blocks, start=1):
    polys = catch.geometry.reindex(ids[basin]).dropna()
    blocks.append(gpd.GeoSeries([shapely.union_all(polys.to_numpy()).simplify(200)], crs=catch.crs).to_crs(CRS).iloc[0])

x0, y0, x1, y1 = outline.total_bounds
mx, my = 0.04 * (x1 - x0), 0.06 * (y1 - y0)
vx0, vx1, vy0, vy1 = x0 - mx, x1 + mx, y0 - my, y1 + my  # view: the basin with a margin that shows the neighbouring states
view = shapely.box(vx0, vy0, vx1, vy1)
W = 17.4 / 2.54
panel_h = W * (vy1 - vy0) / (vx1 - vx0)
fig, axes = plt.subplots(2, 1, figsize=(W, 2 * panel_h + 0.8), gridspec_kw=dict(hspace=0.12))
MANUAL = {"Kentucky": (0.47, 0.84), "Virginia": (0.80, 0.975), "North Carolina": (0.92, 0.36), "Alabama": (0.29, 0.035)}
NAMES = ["Kentucky", "Virginia", "Tennessee", "Mississippi", "Alabama", "Georgia", "North Carolina"]


def base(ax):
    conus.boundary.plot(ax=ax, color=STATE_LINE, lw=0.7, zorder=0.2)
    outline.plot(ax=ax, color=BASIN_FILL, edgecolor="none", zorder=0.4)
    # main stems stay clear, tributary texture recedes: reaches below 100 km2 are not drawn; 100-1,000 km2 thin and pale; larger reaches stronger
    for lw, lo, hi, alpha in ((0.3, 100, 1000, 0.45), (0.8, 1000, 10000, 0.85), (1.5, 10000, 1e9, 1.0)):
        lines[(lines.a >= lo) & (lines.a < hi)].plot(ax=ax, color=RIVER, lw=lw, alpha=alpha, zorder=1)
    outline.boundary.plot(ax=ax, color=NAVY, lw=1.8, zorder=3)
    for name in NAMES:
        part = conus[conus.NAME == name].geometry.iloc[0].intersection(view)
        if name == "Tennessee":
            p = part.representative_point()
            xy = (p.x, p.y)
        elif name in MANUAL:  # positions in view fractions where the automatic spot touches the basin or the view edge
            xy = (vx0 + MANUAL[name][0] * (vx1 - vx0), vy0 + MANUAL[name][1] * (vy1 - vy0))
        else:  # largest piece of the state outside a 20 km buffer around the basin
            free = part.difference(outline.geometry.union_all().buffer(20e3))
            if not free.is_empty:
                part = max(free.geoms, key=lambda q: q.area) if free.geom_type == "MultiPolygon" else free
            p = part.representative_point()
            xy = (p.x, p.y)
        ax.text(*xy, name, fontsize=7.5, style="italic", color=STATE_TEXT, ha="center", va="center", zorder=0.6,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    ax.set_xlim(vx0, vx1)
    ax.set_ylim(vy0, vy1)
    ax.set_aspect("equal")
    ax.set_axis_off()


def scale_and_north(ax):
    sx, sy = vx1 - 0.22 * (vx1 - vx0), vy0 + 0.04 * (vy1 - vy0)
    ax.plot([sx, sx + 100e3], [sy, sy], color="black", lw=1.5, solid_capstyle="butt", zorder=6)
    ax.plot([sx, sx + 50e3], [sy, sy], color="white", lw=0.7, solid_capstyle="butt", zorder=7)
    ax.text(sx, sy + 0.035 * (vy1 - vy0), "0", ha="center", va="bottom", fontsize=7.5, zorder=8, path_effects=[pe.withStroke(linewidth=6, foreground="white")])
    ax.text(sx + 100e3, sy + 0.035 * (vy1 - vy0), "100 km", ha="center", va="bottom", fontsize=7.5, zorder=8, path_effects=[pe.withStroke(linewidth=6, foreground="white")])
    nx, ny = sx - 0.05 * (vx1 - vx0), vy0 + 0.04 * (vy1 - vy0)
    ax.annotate("N", xy=(nx, ny + 0.15 * (vy1 - vy0)), xytext=(nx, ny), ha="center", va="center", fontsize=7.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.9))


ev = c.scored["gauged"]  # the 101 evaluation gauges
inb = (c.in_block > 0) & c.scored["blocked"]  # the 29 gauges scored in the blocked protocol

# ---- (a) ---------------------------------------------------------------------------------------------------------------------
ax = axes[0]
base(ax)
ax.scatter(*gxy[ev].T, s=16, marker="o", color=GAUGE, edgecolor="white", lw=0.5, zorder=5)
scale_and_north(ax)
ins = ax.inset_axes([0.0, 0.76, 0.23, 0.24])
conus.plot(ax=ins, color="#EEEEEE", edgecolor="#9A9A9A", lw=0.3)
outline.plot(ax=ins, color="#E8663A", edgecolor="#E8663A", lw=0.3)
ins.set_axis_off()
ins.set_aspect("equal")
ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=GAUGE, mec="white", mew=0.5, ms=5.5, label="evaluation gauge"),
                   Line2D([], [], color=RIVER, lw=1.6, label="river network"),
                   Line2D([], [], color=NAVY, lw=2, label="basin boundary")],
          loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=7.5, handletextpad=0.5, columnspacing=1.8)

# ---- (b) ---------------------------------------------------------------------------------------------------------------------
ax = axes[1]
base(ax)
for k, poly in enumerate(blocks, start=1):
    gpd.GeoSeries([poly], crs=CRS).plot(ax=ax, color=BASIN[k], alpha=0.42, edgecolor="none", zorder=1.5)
    gpd.GeoSeries([poly], crs=CRS).boundary.plot(ax=ax, color=BASIN[k], lw=1.3, zorder=2.5)
other = ev & ~inb
ax.scatter(*gxy[other].T, s=8, marker="o", color=GRAY, edgecolor="white", lw=0.25, zorder=4)
for k in (1, 2, 3):
    m = inb & (c.in_block == k)
    ax.scatter(*gxy[m].T, s=18, marker="o", color=BASIN[k], edgecolor="white", lw=0.5, zorder=5)
LABEL = {1: (0.11, -0.01), 2: (-0.14, 0.14), 3: (-0.02, -0.11)}  # label position relative to the basin's gauge centroid, as a fraction of the view: outside the basin
for k in (1, 2, 3):
    m = inb & (c.in_block == k)
    cx, cy = float(np.mean(gxy[m][:, 0])), float(np.mean(gxy[m][:, 1]))
    ax.annotate(f"B{k}", xy=(cx, cy), xytext=(cx + LABEL[k][0] * (vx1 - vx0), cy + LABEL[k][1] * (vy1 - vy0)), fontsize=8.5, fontweight="bold", ha="center", va="center",
                zorder=8, color="black", bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=BASIN[k], lw=1.3),
                arrowprops=dict(arrowstyle="-", color=BASIN[k], lw=1.1, shrinkA=0, shrinkB=2))
ax.text(0.0, 1.0, "(b)", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
ax.legend(handles=[Patch(facecolor=BASIN[k], alpha=0.55, edgecolor=BASIN[k], label=f"B{k}") for k in (1, 2, 3)]
          + [Line2D([], [], marker="o", ls="", color=GRAY, mec="white", mew=0.3, ms=4.5, label="other gauges")],
          loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False, fontsize=7.5, handletextpad=0.5, columnspacing=1.8)
for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
    fig.savefig(FIG / f"fig1_study_area.{ext}", bbox_inches="tight", facecolor="white", **kw)
print("(a) gauges drawn:", int(ev.sum()), "| reaches drawn:", len(lines), "| (b) gray gauges:", int(other.sum()),
      "| blocked basins B1/B2/B3:", [int((inb & (c.in_block == k)).sum()) for k in (1, 2, 3)], "| total", int(other.sum() + inb.sum()))
