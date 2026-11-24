"""Figure 4 (optional): mean elevation of every NHDPlus catchment (m), StreamCat "Elevation" attribute (`elevcat`, one value per catchment, so the
fill is catchment-level, not a DEM). Map frame, state boundaries, scale bar, north arrow and CONUS inset follow Fig 1; blue ramp from Fig 1's palette.
Water-body holes inside catchment polygons (180 holes, 0.9 km2 in total) are filled with the catchment value and are stated in the caption.
Polygons are simplified by 60 m for file size (boundaries move by less than 60 m). Full width 17.4 cm.
Run: python scripts/section3/fig4_elevation.py. Inputs: processed/graph_candidate/static_core.nc, catchments_clean.parquet, tennessee_huc2.geojson, cb_2022 states.
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
import xarray as xr
from common import FIG, INPUTS, P
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from shapely.geometry import Polygon

CRS = "EPSG:5070"
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
NAVY, STATE_LINE, STATE_TEXT = "#14213D", "#8C8C8C", "#5E5E5E"
RAMP = LinearSegmentedColormap.from_list("fig1_blues", ["#E9F0FB", "#6FA0DC", "#2B7BE0", "#14213D"])  # Fig 1 palette, lightness decreasing

# ---- data: one StreamCat elevcat value per catchment ----------------------------------------------------------------------------
STATIC = P.PROCESSED / "graph_candidate/static_core.nc"
with xr.open_dataset(STATIC) as d:
    feats = list(d.static_feature.values.astype(str))
    i = feats.index("elevcat")
    elev = pd.Series(d.static_values.values[:, i], index=d.feature_id.values)
    ok = pd.Series(d.static_valid.values[:, i].astype(bool), index=d.feature_id.values)
elev = elev[ok]
catch = gpd.read_parquet(INPUTS["catchments"]).to_crs(CRS)
catch["elev"] = elev.reindex(catch.comid.to_numpy()).to_numpy()
n_holes, holes_km2 = 0, 0.0
geoms = []
for g in catch.geometry:
    parts = []
    for p in getattr(g, "geoms", [g]):
        for r in p.interiors:  # water bodies cut out of the catchment: filled with the catchment value
            n_holes += 1
            holes_km2 += Polygon(r).area / 1e6
        parts.append(Polygon(p.exterior))
    geoms.append(shapely.union_all(parts) if len(parts) > 1 else parts[0])
catch = gpd.GeoDataFrame(catch[["comid", "elev"]], geometry=gpd.GeoSeries(geoms, index=catch.index).simplify(60, preserve_topology=True), crs=CRS)
outline = gpd.read_file(INPUTS["huc2"]).to_crs(CRS)
states = gpd.read_file(INPUTS["states"]).to_crs(CRS)
conus = states[~states.STUSPS.isin(["AK", "HI", "PR", "VI", "GU", "MP", "AS"])]

# ---- frame: as Fig 1 (basin plus margins that show the neighbouring states) ---------------------------------------------------
x0, y0, x1, y1 = outline.total_bounds
mx, my = 0.04 * (x1 - x0), 0.06 * (y1 - y0)
vx0, vx1, vy0, vy1 = x0 - mx, x1 + mx, y0 - my, y1 + my
view = shapely.box(vx0, vy0, vx1, vy1)
W = 17.4 / 2.54
panel_h = W * (vy1 - vy0) / (vx1 - vx0)
fig, ax = plt.subplots(figsize=(W, panel_h + 0.5))
fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
norm = Normalize(vmin=float(catch.elev.min()), vmax=float(catch.elev.max()))
catch.plot(ax=ax, column="elev", cmap=RAMP, norm=norm, lw=0, zorder=1)
conus.boundary.plot(ax=ax, color=STATE_LINE, lw=0.7, zorder=2)
outline.boundary.plot(ax=ax, color=NAVY, lw=1.8, zorder=3)
MANUAL = {"Kentucky": (0.47, 0.84), "Virginia": (0.80, 0.975), "North Carolina": (0.92, 0.36), "Alabama": (0.29, 0.035)}
for name in ["Kentucky", "Virginia", "Tennessee", "Mississippi", "Alabama", "Georgia", "North Carolina"]:
    part = conus[conus.NAME == name].geometry.iloc[0].intersection(view)
    if name in MANUAL:
        xy = (vx0 + MANUAL[name][0] * (vx1 - vx0), vy0 + MANUAL[name][1] * (vy1 - vy0))
    else:
        p = part.representative_point()
        xy = (p.x, p.y)
    ax.text(*xy, name, fontsize=7.5, style="italic", color=STATE_TEXT, ha="center", va="center", zorder=4,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
ax.set_xlim(vx0, vx1)
ax.set_ylim(vy0, vy1)
ax.set_aspect("equal")
ax.set_axis_off()
sx, sy = vx1 - 0.22 * (vx1 - vx0), vy0 + 0.04 * (vy1 - vy0)
ax.plot([sx, sx + 100e3], [sy, sy], color="black", lw=1.5, solid_capstyle="butt", zorder=6)
ax.plot([sx, sx + 50e3], [sy, sy], color="white", lw=0.7, solid_capstyle="butt", zorder=7)
ax.text(sx, sy + 0.035 * (vy1 - vy0), "0", ha="center", va="bottom", fontsize=7.5, zorder=8, path_effects=[pe.withStroke(linewidth=6, foreground="white")])
ax.text(sx + 100e3, sy + 0.035 * (vy1 - vy0), "100 km", ha="center", va="bottom", fontsize=7.5, zorder=8, path_effects=[pe.withStroke(linewidth=6, foreground="white")])
nx, ny = sx - 0.05 * (vx1 - vx0), vy0 + 0.04 * (vy1 - vy0)
ax.annotate("N", xy=(nx, ny + 0.15 * (vy1 - vy0)), xytext=(nx, ny), ha="center", va="center", fontsize=7.5, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.9), zorder=8)
ins = ax.inset_axes([0.0, 0.76, 0.23, 0.24])
conus.plot(ax=ins, color="#EEEEEE", edgecolor="#9A9A9A", lw=0.3)
outline.plot(ax=ins, color=NAVY, edgecolor=NAVY, lw=0.3)
ins.set_axis_off()
ins.set_aspect("equal")
# colorbar as an inset strip under the map, so the map gets the full width
cax = ax.inset_axes([0.30, -0.075, 0.40, 0.03])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=RAMP), cax=cax, orientation="horizontal")
cb.set_label("Mean catchment elevation (m)", fontsize=7.5, labelpad=2)
cb.ax.tick_params(labelsize=7.5, length=2)
cb.outline.set_linewidth(0.4)
for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
    fig.savefig(FIG / f"fig4_elevation.{ext}", bbox_inches="tight", pad_inches=0.02, facecolor="white", **kw)
print("catchments drawn:", len(catch), "| with elevation:", int(catch.elev.notna().sum()), "| range m:", round(norm.vmin, 1), round(norm.vmax, 1),
      "| water-body holes filled:", n_holes, "totalling", round(holes_km2, 2), "km2")
