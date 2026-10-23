"""TVA headwater archive: gauges, observed flow, issued forecasts and gauge-basin rain.

Observed flow is TVA's six-hour mean over [t, t+6h) in UTC (checked against USGS
hourly data: zero offset). A TVA forecast value at lead L is the six-hour mean
ending at issue + L, so TVA lead 6(k+1) h matches model lead index k, and TVA
lead 0 is the last observed interval, the same cut-off as the model's history.
TVA's forecast precipitation is real to about 72 h; from 78 to 96 h it carries a
fifth of the observed rain (data.rain_error), and it is zero filled after that.
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import xarray as xr

from .schema import HORIZON
from .transforms import transform_precipitation

ATTRIBUTES = "TVA_headwater_basin_CAMELSH_StreamCat_attributes.csv"

# Outlet reaches the archive lacks (its CAMELSH match needs an NWPS-mapped USGS
# id) or gets wrong. Each is the graph flowline nearest the co-located USGS gauge
# (CAMELSH coordinates; TVA coordinates for WCKG1) whose total drainage area is
# within 2% of the gauge's; all are within 30 m. The same rule reproduces 30 of
# the archive's 31 matches; the exception is SGWN7.
REACHES = {
    "CHSA1": 19621280,  # USGS 03575000
    "FTCT1": 19668435,  # USGS 03560000
    "GERA1": 19648068,  # USGS 03572900
    "PLKT1": 19595982,  # USGS 03584020
    "SGWN7": 19743430,  # archive gives the 90 km2 reach upstream; gauge 03479000 drains 236 km2
    "TLRT1": 19722609,  # USGS 03518500
    "TWNT1": 22130697,  # USGS 03497300
    "VSTV2": 19752425,  # USGS 03473000
    "WCKG1": 19697599,  # USGS 03567340, not in CAMELSH
}


def extract(archive, directory):
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if member.startswith("__MACOSX") or member.endswith((".DS_Store", "/")):
                continue
            bundle.extract(member, directory)
    folders = [p.parent for p in Path(directory).rglob(ATTRIBUTES)]
    if len(folders) != 1:
        raise ValueError(f"Expected one {ATTRIBUTES} in {archive}")
    return folders[0]


def _has_flow(path):
    if not path.exists():
        return False
    with xr.open_dataset(path) as data:
        return "river_flow" in data and bool(np.isfinite(data.river_flow.values).any())


def gauges(folder, ids, forecasts=True):
    """Stations with a graph reach and observed flow (and, by default, TVA flow forecasts)."""
    table = pd.read_csv(folder / ATTRIBUTES)
    fixed = table.tva_nwsid.map(REACHES)
    archive = table.camelsh_outlet_comid.where(table.camelsh_comid_match_status.eq("matched"))
    table["comid"] = fixed.fillna(archive)
    table["comid_source"] = np.where(fixed.notna(), "location_and_area", "archive")
    table = table[table.comid.notna()].copy()
    table["comid"] = table.comid.astype(np.int64)
    table = table[table.comid.isin(ids)]
    keep = [
        row.Index
        for row in table.itertuples()
        if _has_flow(folder / "observed" / f"{row.tva_nwsid}.nc")
        and (not forecasts or _has_flow(folder / "forecast" / f"{row.tva_nwsid}.nc"))
    ]
    return table.loc[keep].reset_index(drop=True)


def observed_flow(folder, table, times):
    """Observed six-hour flow [time, station] on a timeline (NaN where TVA has none)."""
    observed = np.full((len(times), len(table)), np.nan)
    for s, station in enumerate(table.tva_nwsid):
        with xr.open_dataset(folder / "observed" / f"{station}.nc") as data:
            series = data.river_flow.to_series()
        observed[:, s] = series[~series.index.duplicated()].reindex(times).to_numpy()
    return observed


def align(folder, table, times):
    """Observed [time, station] on the model timeline and TVA forecasts by issue index."""
    observed = observed_flow(folder, table, times)
    forecasts, precipitation = {}, {}
    for s, station in enumerate(table.tva_nwsid):
        with xr.open_dataset(folder / "forecast" / f"{station}.nc") as data:
            leads = list(data.lead_time.values.astype(int))
            wanted = [
                leads.index(6 * (k + 1)) if 6 * (k + 1) in leads else -1 for k in range(HORIZON)
            ]
            values = data.river_flow.values.astype(np.float64)
            rain = (
                data.precipitation.values.astype(np.float64)
                if "precipitation" in data
                else np.full(values.shape, np.nan)
            )
            issues = times.get_indexer(pd.DatetimeIndex(data.forecast_time.values))
        for row, index in enumerate(issues):
            if index < 0:
                continue
            for store, source in ((forecasts, values), (precipitation, rain)):
                lead_values = np.array([source[row, w] if w >= 0 else np.nan for w in wanted])
                store.setdefault(int(index), np.full((HORIZON, len(table)), np.nan))[:, s] = (
                    lead_values
                )
    return observed, forecasts, precipitation


def basin_owner(edges, n, outlets):
    """Gauge index owning each reach: smallest upstream basin containing it, else -1."""
    from scipy.sparse.csgraph import breadth_first_order

    edges = np.asarray(edges, dtype=np.int64)
    # Edges point downstream (source -> target); walk them backwards from each outlet.
    upstream = sp.csr_matrix(
        (np.ones(len(edges), dtype=bool), (edges[:, 1], edges[:, 0])), shape=(n, n)
    )
    basins = [
        breadth_first_order(upstream, int(o), directed=True, return_predecessors=False)
        for o in outlets
    ]
    owner = np.full(n, -1, dtype=np.int64)
    for s in np.argsort([-len(b) for b in basins], kind="stable"):
        owner[basins[s]] = s
    return owner, np.array([len(b) for b in basins])


def basin_rain(
    aorc_value, aorc_flag, aorc_known, forecast, forecast_owner, use_forecast, steps, rain
):
    """Precipitation-only future inputs [lead, node, 2]: normalized value and known flag.

    Rain is known only within the first ``steps`` leads. Realized AORC rain
    (``aorc_value``/``aorc_flag`` [lead, node]) is used at reaches with
    ``aorc_known`` [node]. ``forecast`` [lead, gauge] (raw mm, NaN where missing)
    is a gauge's basin forecast, applied to every reach its basin owns
    (``forecast_owner`` [node] gauge index or -1) when ``use_forecast`` [gauge];
    it takes precedence over AORC. ``rain`` is (transform, mean, std).
    Everything else is unknown (value and flag zero).
    """
    transform, mean, std = rain
    leads, nodes = aorc_value.shape
    device, dtype = aorc_value.device, aorc_value.dtype
    window = torch.zeros((leads, 1), dtype=torch.bool, device=device)
    window[:steps] = True
    keep = window & aorc_known[None] & aorc_flag
    value = torch.where(keep, aorc_value, torch.zeros_like(aorc_value))
    flag = keep.to(dtype)
    if forecast is not None:
        raw = np.asarray(forecast, dtype=np.float64)
        valid = np.isfinite(raw) & (raw >= 0)
        z = (transform_precipitation(raw, valid, transform) - mean) / std
        z = torch.as_tensor(np.where(valid, z, 0.0), dtype=dtype, device=device)
        valid = torch.as_tensor(valid, device=device)
        owned = forecast_owner >= 0
        safe = torch.where(owned, forecast_owner, torch.zeros_like(forecast_owner))
        active = torch.as_tensor(use_forecast, device=device)[safe]
        known = window & (owned & active)[None] & valid[:, safe]
        value = torch.where(known, z[:, safe], value)
        flag = torch.where(known, torch.ones_like(flag), flag)
    return torch.stack([value, flag], dim=-1)
