"""Figure 4: basin attributes in a 3 x 2 grid of maps (full width 17.4 cm). Each NHDPlus catchment is filled with one value of its own (no DEM, no interpolation):
(a) mean catchment elevation (StreamCat elevcat; hypsometric tints, rivers of at least 5,000 km2 in blue), (b) mean annual AORC precipitation 2000-2019,
(c) river network by Strahler order (line width and color), (d) reach slope (log classes), (e) forest cover and (f) developed cover of the local catchment
(StreamCat NLCD 2011). Class breaks are fixed and printed on the colorbars. Frame, state boundaries, north arrow and scale bar follow Fig 1 (drawn in (a) only); state names in every panel; no CONUS inset.
Reads figures/section3/basin_attributes.csv (scripts/section3/basin_attributes.py). Run: python scripts/section3/fig4_basin_grid.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from common import FIG, INPUTS
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from shapely.geometry import Polygon

CRS = "EPSG:5070"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
NAVY, STATE_LINE, STATE_TEXT, RIVER = "#14213D", "#8C8C8C", "#5E5E5E", "#1F6FD1"
HALO = [pe.withStroke(linewidth=2.5, foreground="white")]

attrs = pd.read_csv(FIG / "basin_attributes.csv").set_index("comid")
catch = gpd.read_parquet(INPUTS["catchments"]).to_crs(CRS)
geoms = []
for g in catch.geometry:  # fill water-body holes with the catchment value, as in the earlier Fig 4
    parts = [Polygon(p.exterior) for p in getattr(g, "geoms", [g])]
    geoms.append(shapely.union_all(parts) if len(parts) > 1 else parts[0])
catch = gpd.GeoDataFrame(catch[["comid"]], geometry=gpd.GeoSeries(geoms, index=catch.index).simplify(60, preserve_topology=True), crs=CRS)
catch = catch.join(attrs, on="comid")
lines = gpd.read_file(INPUTS["flowlines"], columns=["COMID"])
lines = lines[lines.COMID.isin(attrs.index)].set_index("COMID").to_crs(CRS).join(attrs[["area_km2", "order"]])
outline = gpd.read_file(INPUTS["huc2"]).to_crs(CRS)
states = gpd.read_file(INPUTS["states"]).to_crs(CRS)
conus = states[~states.STUSPS.isin(["AK", "HI", "PR", "VI", "GU", "MP", "AS"])]

# ---- class breaks and palettes -------------------------------------------------------------------------------------------------------
# elevation: hypsometric tints (dark green lowland, through yellow and orange, to dark red and pale gray at the top)
ELEV_B = [0, 200, 300, 450, 650, 900, 1200, 1500, 1800]
ELEV_C = ["#1E6B3A", "#5FA044", "#B4D36B", "#F4E36E", "#F3B13E", "#E4702B", "#B5301F", "#7A1F1A"]
PREC_B = [1000, 1300, 1375, 1425, 1475, 1550, 1700, 2400]
PREC_C = ["#EAF3FB", "#C6DDF1", "#93BFE3", "#5A98D1", "#2E6FB7", "#1A4A95", "#13296B"]
SLOPE_B = [0, 0.0005, 0.002, 0.005, 0.01, 0.03, 0.1, 10.0]  # m/m
SLOPE_C = ["#FFF5D6", "#FBD58A", "#F4A261", "#E4572E", "#B02A5A", "#6A1B6D", "#2B0F4A"]
FOR_B = [0, 20, 40, 60, 80, 100.01]
FOR_C = ["#F3F6E8", "#CFE3A8", "#8FC070", "#3E9553", "#14633B"]
DEV_B = [0, 2, 5, 10, 25, 50, 100.01]
DEV_C = ["#F6F1EA", "#E7D5C4", "#D2A98C", "#B9776A", "#8E3E58", "#4A1942"]
ORDER_C = {1: "#9CC4EA", 2: "#6FA6DE", 3: "#4388D2", 4: "#2B6AB8", 5: "#1F4E94", 6: "#17376F", 7: "#10254D", 8: "#0A1530"}
ORDER_LW = {1: 0.15, 2: 0.25, 3: 0.4, 4: 0.6, 5: 0.9, 6: 1.3, 7: 1.8, 8: 2.4}

# ---- frame ---------------------------------------------------------------------------------------------------------------------------
x0, y0, x1, y1 = outline.total_bounds
mx, my = 0.04 * (x1 - x0), 0.06 * (y1 - y0)
vx0, vx1, vy0, vy1 = x0 - mx, x1 + mx, y0 - my, y1 + my
view = shapely.box(vx0, vy0, vx1, vy1)
W = 17.4 / 2.54
pw = W / 2
ph = pw * (vy1 - vy0) / (vx1 - vx0)
fig, axes = plt.subplots(3, 2, figsize=(W, 3 * (ph + 0.42)), gridspec_kw=dict(hspace=0.24, wspace=0.02, left=0.0, right=1.0, top=1.0, bottom=0.0))
axes = axes.ravel()


def add_state_names(ax):
    MANUAL = {"Kentucky": (0.47, 0.84), "Virginia": (0.80, 0.95), "North Carolina": (0.885, 0.36), "Alabama": (0.29, 0.035), "Mississippi": (0.08, 0.055)}
    for name in ["Kentucky", "Virginia", "Tennessee", "Mississippi", "Alabama", "Georgia", "North Carolina"]:
        part = conus[conus.NAME == name].geometry.iloc[0].intersection(view)
        if name in MANUAL:
            xy = (vx0 + MANUAL[name][0] * (vx1 - vx0), vy0 + MANUAL[name][1] * (vy1 - vy0))
        else:
            p = part.representative_point()
            xy = (p.x, p.y)
        ax.text(*xy, name, fontsize=6, style="italic", color=STATE_TEXT, ha="center", va="center", zorder=6, path_effects=HALO)


def base(ax, label):
    conus.boundary.plot(ax=ax, color=STATE_LINE, lw=0.5, zorder=2)
    ax.set_xlim(vx0, vx1)
    ax.set_ylim(vy0, vy1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():  # border around each map
        sp.set_linewidth(0.8)
        sp.set_color("black")
    add_state_names(ax)
    ax.text(0.012, 0.985, label, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", zorder=9)


def outline_on_top(ax):
    outline.boundary.plot(ax=ax, color=NAVY, lw=1.1, zorder=5)


def colorbar(ax, bounds, colors, label, fmt, extend=None):
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    cax = ax.inset_axes([0.14, -0.115, 0.72, 0.032])
    cb = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=bounds[: len(colors) + 1])
    cb.ax.set_xticklabels([fmt(b) for b in bounds[: len(colors) + 1]])
    cb.set_label(label, fontsize=7.5, labelpad=2)
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    cb.outline.set_linewidth(0.4)
    return cmap, norm


def classed(ax, col, bounds, colors, label, fmt):
    cmap, norm = colorbar(ax, bounds, colors, label, fmt)
    catch.plot(ax=ax, column=col, cmap=cmap, norm=norm, lw=0, zorder=1, rasterized=True, missing_kwds=dict(color="#DDDDDD"))
    base(ax, "")


# (a) elevation with main rivers
ax = axes[0]
classed(ax, "elev_m", ELEV_B, ELEV_C, "Mean catchment elevation (m)", lambda b: f"{b:,.0f}")
big = lines[lines.area_km2 >= 5000]
big.plot(ax=ax, color=RIVER, lw=0.9, zorder=4)
ax.plot([], [], color=RIVER, lw=1.2, label="Rivers $\\geq$ 5,000 km$^2$")
ax.legend(loc="upper left", bbox_to_anchor=(0.012, 0.91), fontsize=6.5, frameon=True, framealpha=0.95, edgecolor="#999999", fancybox=False, handlelength=1.6)
outline_on_top(ax)
ax.text(0.012, 0.985, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", zorder=9)
sx, sy = vx1 - 0.215 * (vx1 - vx0), vy0 + 0.022 * (vy1 - vy0)
ax.plot([sx, sx + 100e3], [sy, sy], color="black", lw=1.3, solid_capstyle="butt", zorder=8)
ax.plot([sx, sx + 50e3], [sy, sy], color="white", lw=0.6, solid_capstyle="butt", zorder=9)
for xx, t in ((sx, "0"), (sx + 100e3, "100 km")):
    ax.text(xx, sy + 0.035 * (vy1 - vy0), t, ha="center", va="bottom", fontsize=6.5, zorder=9, path_effects=[pe.withStroke(linewidth=5, foreground="white")])
nx = sx - 0.06 * (vx1 - vx0)
ax.annotate("N", xy=(nx, sy + 0.14 * (vy1 - vy0)), xytext=(nx, sy), ha="center", va="center", fontsize=6.5, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8), zorder=9)

# (b) precipitation
ax = axes[1]
classed(ax, "precip_mm_yr", PREC_B, PREC_C, "Mean annual precipitation, AORC 2000–2019 (mm)", lambda b: f"{b:,.0f}")
outline_on_top(ax)
ax.text(0.012, 0.985, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", zorder=9)

# (c) river network by Strahler order
ax = axes[2]
base(ax, "")
ax.set_facecolor("white")
outline.plot(ax=ax, color="#F2F6FB", edgecolor="none", zorder=0.4)
for o in range(1, 9):
    s = lines[lines.order == o]
    if len(s):
        s.plot(ax=ax, color=ORDER_C[o], lw=ORDER_LW[o], zorder=1 + o / 10, capstyle="round")
outline_on_top(ax)
ax.text(0.012, 0.985, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", zorder=9)
cax = ax.inset_axes([0.14, -0.115, 0.72, 0.032])
cmap_o = ListedColormap([ORDER_C[o] for o in range(1, 9)])
cb = fig.colorbar(ScalarMappable(norm=BoundaryNorm(np.arange(0.5, 9.5), 8), cmap=cmap_o), cax=cax, orientation="horizontal", ticks=np.arange(1, 9))
cb.set_label("Strahler stream order", fontsize=7.5, labelpad=2)
cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
cb.outline.set_linewidth(0.4)

# (d) slope, (e) forest, (f) developed
ax = axes[3]
classed(ax, "slope", SLOPE_B, SLOPE_C, "Reach slope (m m$^{-1}$)", lambda b: f"{b:g}" if b < 1 else "")
outline_on_top(ax)
ax.text(0.012, 0.985, "(d)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", zorder=9)
ax = axes[4]
classed(ax, "forest_pct", FOR_B, FOR_C, "Forest cover of the catchment (%)", lambda b: f"{min(b, 100):.0f}")
outline_on_top(ax)
ax.text(0.012, 0.985, "(e)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", zorder=9)
ax = axes[5]
classed(ax, "developed_pct", DEV_B, DEV_C, "Developed cover of the catchment (%)", lambda b: f"{min(b, 100):.0f}")
outline_on_top(ax)
ax.text(0.012, 0.985, "(f)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", zorder=9)

for a in axes:  # map layers as raster (the PDF stays small); text, colorbars and outlines stay vector
    for coll in a.collections:
        coll.set_rasterized(True)
for ext, kw in (("pdf", dict(dpi=300)), ("png", dict(dpi=300))):
    fig.savefig(FIG / f"fig4_basin_grid.{ext}", bbox_inches="tight", pad_inches=0.02, facecolor="white", **kw)

# share of catchments (and of basin area) per class, for the caption and the README
rows = []
for col, b in (("elev_m", ELEV_B), ("precip_mm_yr", PREC_B), ("slope", SLOPE_B), ("forest_pct", FOR_B), ("developed_pct", DEV_B)):
    v = catch[col]
    cnt = pd.cut(v, b, right=False).value_counts(sort=False)
    rows.append(f"{col}: " + "; ".join(f"[{i.left:g},{i.right:g}) {100 * n / v.notna().sum():.1f}%" for i, n in cnt.items()) + f" | missing {int(v.isna().sum())}")
print("\n".join(rows))
print("orders:", lines["order"].value_counts().sort_index().to_dict())
