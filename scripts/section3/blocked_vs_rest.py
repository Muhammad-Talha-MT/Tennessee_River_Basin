"""Section 3 table: the three blocked basins (B1-B3, scored gauges) compared with the other scored gauged-protocol gauges.
Each gauge is one unit: its upstream basin (graph `members`), drainage area from the graph (`routing.npz` basin_area). Basin means are area-weighted
by the local catchment area of each reach (`routing.npz` area). Group rows summarise the gauges of the group (median, range or mean across gauges).
Sources: elevation and slope from static_core.nc (StreamCat `elevcat` per catchment; `reach_slope` per reach); precipitation from the AORC six-hour
rain (`six_hour_nodes/*/APCP_surface`, mm per six-hour step) summed to annual totals over 2000-2019 and averaged; forest and developed shares from
StreamCat NLCD 2011 watershed values (`tennessee_landcover_2011.nc`, `ws` = upstream watershed of the gauge reach). Run via facts.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glob

import numpy as np
import pandas as pd
import xarray as xr
from common import P

SIX_HOUR = P.PROCESSED / "six_hour_nodes"
Y0, Y1 = 2000, 2019
STATIC = P.PROCESSED / "graph_candidate/static_core.nc"
LANDCOVER = P.PROCESSED / "static/tennessee_landcover_2011.nc"


def reach_precip(ids):
    """Mean annual AORC precipitation (mm) per reach over Y0-Y1; annual total = finite six-hour steps scaled to 1,461 steps per year."""
    total, steps = np.zeros(len(ids)), np.zeros(len(ids))
    for year in range(Y0, Y1 + 1):
        for f in sorted(glob.glob(str(SIX_HOUR / f"{year}/*.nc"))):
            with xr.open_dataset(f) as d:
                assert np.array_equal(d.feature_id.values, ids), "AORC node order differs from the graph"
                a = d.APCP_surface.values.astype(float)
            total += np.nansum(a, axis=0)
            steps += np.isfinite(a).sum(axis=0)
    per_step = np.divide(total, steps, out=np.full(len(ids), np.nan), where=steps > 0)
    return per_step * 1461.0  # mm per year (1,461 six-hour steps in an average year)


def compute(c):
    n = c.network
    with xr.open_dataset(STATIC) as d:
        f = list(d.static_feature.values.astype(str))
        elevcat = d.static_values.values[:, f.index("elevcat")].astype(float)
        slope = d.static_values.values[:, f.index("reach_slope")].astype(float)
    with xr.open_dataset(LANDCOVER) as d:
        lf = list(d.static_feature.values.astype(str))
        lc = d.static_values.values.astype(float)
    cols = {k: lc[:, lf.index(k)] for k in ("pctmxfst2011ws", "pctdecid2011ws", "pctconif2011ws", "pcturbop2011ws", "pcturblo2011ws", "pcturbmd2011ws", "pcturbhi2011ws")}
    forest_ws = cols["pctmxfst2011ws"] + cols["pctdecid2011ws"] + cols["pctconif2011ws"]
    developed_ws = cols["pcturbop2011ws"] + cols["pcturblo2011ws"] + cols["pcturbmd2011ws"] + cols["pcturbhi2011ws"]
    precip = reach_precip(n.ids)
    w = n.area.astype(float)  # local catchment area per reach (km2)
    ev = c.scored["gauged"]
    blk = c.scored["blocked"]
    rows = []
    for g in np.flatnonzero(ev):
        group = f"B{int(c.in_block[g])}" if blk[g] else "other"
        if blk[g]:
            group = f"B{int(c.in_block[g])}"
        mem = np.asarray(c.gauges.basins[g])
        ww = w[mem]
        def wmean(x):
            ok = np.isfinite(x[mem]) & (ww > 0)
            return float(np.sum(ww[ok] * x[mem][ok]) / np.sum(ww[ok])) if ok.any() else np.nan
        r = c.gauges.reach[g]
        rows.append(dict(gauge=c.names[g], group=group, area_km2=float(c.gauges.area[g]), elev_m=wmean(elevcat), slope_m_per_m=wmean(slope),
                         precip_mm_yr=wmean(precip), forest_pct_ws=float(forest_ws[r]), developed_pct_ws=float(developed_ws[r])))
    region = _region(c, elevcat, slope, precip, w)
    return pd.DataFrame(rows), region


def _region(c, elevcat, slope, precip, w):
    """Basin-wide context for the study-area text: area-weighted precipitation by elevation class and by west/east."""
    import geopandas as gpd
    from common import INPUTS

    cat = gpd.read_parquet(INPUTS["catchments"]).to_crs("EPSG:4326").set_index("comid")
    lon = cat.geometry.representative_point().x.reindex(c.network.ids).to_numpy()
    ok = np.isfinite(elevcat) & np.isfinite(precip) & (w > 0)
    wm = lambda x, m: float(np.sum(w[m & ok] * x[m & ok]) / np.sum(w[m & ok]))  # noqa: E731
    q = np.quantile(elevcat[ok], [0.25, 0.5, 0.75, 0.95])
    return dict(basin_precip=wm(precip, np.ones(len(w), bool)), west_precip=wm(precip, lon < -87), east_precip=wm(precip, lon > -84),
                west_elev=wm(elevcat, lon < -87), east_elev=wm(elevcat, lon > -84), quantiles=q,
                by_class=[wm(precip, elevcat < q[0]), wm(precip, (elevcat >= q[0]) & (elevcat < q[1])), wm(precip, (elevcat >= q[1]) & (elevcat < q[2])),
                          wm(precip, (elevcat >= q[2]) & (elevcat < q[3])), wm(precip, elevcat >= q[3])],
                corr=float(np.corrcoef(elevcat[ok], precip[ok])[0, 1]), elev_lt250=float(np.mean(elevcat[np.isfinite(elevcat)] < 250)),
                n_gt1500=int((elevcat > 1500).sum()), elev_max=float(np.nanmax(elevcat)))


def summarise(df):
    """Markdown rows for B1, B2, B3 and the other scored gauges."""
    out = []
    for grp, label in (("B1", "B1"), ("B2", "B2"), ("B3", "B3"), ("other", "other scored gauges (training/other)")):
        s = df[df.group == grp]
        a = s.area_km2
        out.append([label, len(s), f"{np.median(a):,.0f} ({a.min():,.0f}-{a.max():,.0f})",
                    f"{s.elev_m.mean():,.0f} / {s.elev_m.median():,.0f}", f"{s.slope_m_per_m.mean():.4f}",
                    f"{s.precip_mm_yr.mean():,.0f}", f"{s.forest_pct_ws.mean():.1f}", f"{s.developed_pct_ws.mean():.1f}"])
    return out
