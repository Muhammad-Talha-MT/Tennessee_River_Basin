"""Per-reach attribute table for the Section 3 basin-attribute grid (fig4_basin_grid.py). One row per NHDPlus reach/catchment (58,834).
Columns: comid; elev_m (StreamCat elevcat, catchment mean); precip_mm_yr (AORC six-hour rain summed to annual totals, mean 2000-2019, same function as
blocked_vs_rest.reach_precip); slope (reach_slope, m/m); order (stream_order); area_km2 (upstream_drainage_area_km2); forest_pct, developed_pct
(StreamCat NLCD 2011, local catchment "cat" values: mixed + deciduous + coniferous forest; open + low + medium + high intensity developed).
Run: python scripts/section3/basin_attributes.py -> figures/section3/basin_attributes.csv (about a few minutes: reads 20 years of AORC files).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import xarray as xr
from blocked_vs_rest import LANDCOVER, STATIC, reach_precip
from common import FIG, P
from tva_flood.hydro.data import Network

ids = Network().ids
with xr.open_dataset(STATIC) as d:
    f = list(d.static_feature.values.astype(str))
    assert np.array_equal(d.feature_id.values, ids)
    col = lambda k: d.static_values.values[:, f.index(k)].astype(float)  # noqa: E731
    ok = d.static_valid.values.astype(bool)
    elev, slope, order, area = col("elevcat"), col("reach_slope"), col("stream_order"), col("upstream_drainage_area_km2")
    elev[~ok[:, f.index("elevcat")]] = np.nan
with xr.open_dataset(LANDCOVER) as d:
    lf = list(d.static_feature.values.astype(str))
    if "feature_id" in d:
        assert np.array_equal(d.feature_id.values, ids)
    lc = lambda k: d.static_values.values[:, lf.index(k)].astype(float)  # noqa: E731
    forest = lc("pctmxfst2011cat") + lc("pctdecid2011cat") + lc("pctconif2011cat")
    developed = sum(lc(f"pcturb{k}2011cat") for k in ("op", "lo", "md", "hi"))
df = pd.DataFrame(dict(comid=ids, elev_m=elev, precip_mm_yr=reach_precip(ids), slope=slope, order=order, area_km2=area, forest_pct=forest, developed_pct=developed))
df.to_csv(FIG / "basin_attributes.csv", index=False)
print(df.describe().T.to_string())
