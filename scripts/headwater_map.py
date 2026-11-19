"""Reach-level map of the Tennessee basin with the headwater basins outlined.

All 58,834 reaches are drawn (wider for larger drainage areas). A headwater gauge has no TVA dam and no other gauge upstream
(hydro.evaluate.regulation); its basin is the union of its upstream reaches' catchments, outlined in bold. Reaches inside a
headwater basin are dark blue, other natural reaches light blue, reaches with a TVA dam upstream orange.

  python scripts/headwater_map.py OUT.png
"""
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import shapely

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tva_flood.hydro import paths as P  # noqa: E402
from tva_flood.hydro import splits  # noqa: E402
from tva_flood.hydro.data import Arrays, Gauges, Network  # noqa: E402
from tva_flood.hydro.evaluate import regulation  # noqa: E402

OUT = Path(sys.argv[1])
CRS = "EPSG:5070"
NAVY, BLUE, LIGHT_BLUE, ORANGE, GRAY = "#14213D", "#1C5CAB", "#A9C4E4", "#E8A27C", "#52514E"

network = Network()
gauges = Gauges(network, Arrays().times)
regulated, groups = regulation(network, gauges)
headwater = dict(groups)["headwater, no dam"]
scored = splits.gauge_folds("random", gauges) > 0  # the gauges of the held-out folds
ids = network.ids
print(f"headwater gauges: {int(headwater.sum())} ({int((headwater & scored).sum())} in the held-out folds)")

# reaches inside each headwater basin
inside = np.zeros(network.n, bool)
basin_of = {}
for g in np.flatnonzero(headwater):
    members = network.basin(gauges.reach[g])
    inside[members] = True
    basin_of[g] = ids[members]
dammed = regulated > 0
category = np.where(inside, 2, np.where(dammed, 1, 0))
print(f"reaches: {network.n:,}; in headwater basins {int(inside.sum()):,}; with a dam upstream {int(dammed.sum()):,}")

# geometry
fl = gpd.read_file(P.ROOT / "raw/geography/nhdplus_v21/extracted/NHDPlusV21_MS_06_NHDSnapshot_09/NHDPlusMS/NHDPlus06/NHDSnapshot/Hydrography/NHDFlowline.shp", columns=["COMID"])
lines = fl[fl.COMID.isin(ids)].set_index("COMID").to_crs(CRS)
area = pd.read_parquet(P.PROCESSED / "graph_candidate/nodes.parquet", columns=["comid", "totdasqkm"]).set_index("comid").totdasqkm
cat_of = pd.Series(category, index=ids)
lines = lines.assign(cat=cat_of.reindex(lines.index).to_numpy(), a=area.reindex(lines.index).to_numpy())
outline = gpd.read_file(P.ROOT / "raw/geography/tennessee_huc2.geojson").to_crs(CRS)
catchments = gpd.read_parquet(P.PROCESSED / "graph_candidate/catchments_clean.parquet").set_index("comid")
basins = []
for g, members in basin_of.items():
    polys = catchments.geometry.reindex(members).dropna()
    basins.append(shapely.union_all(polys.to_numpy()).simplify(150))
basins = gpd.GeoDataFrame(geometry=basins, crs=catchments.crs).to_crs(CRS)

x0, y0, x1, y1 = outline.total_bounds
fig, ax = plt.subplots(figsize=(12, 12 * (y1 - y0) / (x1 - x0) + 0.3))
outline.plot(ax=ax, color="#F3F5F8", edgecolor=NAVY, lw=1.0)
basins.plot(ax=ax, color="#DCE8F7", edgecolor="none", zorder=1)
for lw, lo, hi in ((0.2, 0, 50), (0.55, 50, 1500), (1.5, 1500, 1e9)):
    part = lines[(lines.a >= lo) & (lines.a < hi)]
    for cat, color in ((0, LIGHT_BLUE), (1, ORANGE), (2, BLUE)):
        sel = part[part.cat == cat]
        if len(sel):
            sel.plot(ax=ax, color=color, lw=lw, zorder=2 + (cat == 2))
basins.boundary.plot(ax=ax, color=NAVY, lw=1.6, zorder=5)  # the headwater basin boundaries, prominent

def outlet(comid):
    geom = lines.geometry.get(comid)
    geom = geom.geoms[-1] if geom.geom_type == "MultiLineString" else geom
    return geom.coords[-1][:2]

pts = np.array([outlet(ids[gauges.reach[g]]) for g in np.flatnonzero(headwater)])
tag = scored[np.flatnonzero(headwater)]
ax.scatter(*pts[tag].T, s=46, color="#EB6834", edgecolor="white", lw=0.9, zorder=7)
ax.scatter(*pts[~tag].T, s=46, color="white", edgecolor="#EB6834", lw=1.4, zorder=7)
with np.load(P.HYDRO / "dams.npz", allow_pickle=True) as z:
    dam_xy = np.array([outlet(c) for c in z["comid"] if c in lines.index])
ax.scatter(*dam_xy.T, s=62, marker="^", color=GRAY, edgecolor="white", lw=0.7, zorder=6)
ax.set_aspect("equal")
ax.set_axis_off()
handles = [
    Line2D([], [], color=BLUE, lw=2.5, label=f"reach inside a headwater basin ({int(inside.sum()):,})"),
    Line2D([], [], color=LIGHT_BLUE, lw=2.5, label=f"other natural reach ({int((~inside & ~dammed).sum()):,})"),
    Line2D([], [], color=ORANGE, lw=2.5, label=f"reach with a TVA dam upstream ({int((~inside & dammed).sum()):,})"),
    Line2D([], [], color=NAVY, lw=2, label="headwater basin boundary"),
    Line2D([], [], marker="o", ls="", color="#EB6834", mec="white", ms=9, label=f"headwater gauge, in the held-out folds ({int((headwater & scored).sum())})"),
    Line2D([], [], marker="o", ls="", mfc="white", mec="#EB6834", ms=9, label=f"headwater gauge, not in a fold ({int((headwater & ~scored).sum())})"),
    Line2D([], [], marker="^", ls="", color=GRAY, mec="white", ms=9, label=f"TVA dam ({len(dam_xy)})"),
]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.17, 0.97), fontsize=9.5, frameon=True, framealpha=0.95)
ax.set_title("Tennessee River Basin: all 58,834 reaches and the headwater basins (no TVA dam, no other gauge upstream)", color=NAVY, fontsize=12.5, loc="left")
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
