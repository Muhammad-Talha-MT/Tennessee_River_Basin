"""TVA dams on the river graph: observed releases and inflows, TVA's forecasts, regulation.

Source: ``raw/tva_dams`` (TVA archive of 44 dams). ``observed/<id>.nc`` holds six-hour
means over [t, t+6h) of the total release (``dam_outflow``: spillway, sluice and turbine)
and of the inflow TVA computes from the release and the storage change (``dam_inflow``;
not a gauge, so a few percent of values are negative). ``forecast/<id>.nc`` holds TVA's
forecasts of both (32 dams, four issues a day from 2016-10-03, leads 0-330 h). Checks
behind the conventions used here:

- Timestamps are UTC: release changes match the USGS gauge just below Nolichucky
  (03466500) and Ocoee #1 (03564500) at lag 0 (r = 0.98 and 0.97, winter and summer
  alike); local time would shift them by one step.
- A forecast at lead L is the six-hour mean ending at issue + L (as TVA's headwater
  forecasts): lead 6 h is model lead 0, lead 0 the last observed interval.

Each dam sits at the reach that ends at the dam and carries its water: of the reaches
within 3 km of the dam's coordinates whose NWM retrospective mean flow (just below, as
reservoir reaches have none, from a reach of at least half the drainage area, so not a side
stream entering the reservoir) is within a factor 2 of the dam's mean release, the one whose
downstream end is nearest; within 250 m the one with more flow, since a reservoir can
have a second, dry outlet reach. (NHD merges back-to-back reservoirs into one waterbody,
so a waterbody's outlet is not the dam's. NWM runs 1.5-1.9 low in the Bear Creek basin;
Nantahala and Thorpe are powerhouses, where piped water re-enters the river.) No match is
an error. Four reaches are fixed: the archive's Nottely coordinates are Norris Dam's,
the reach ending nearest Nickajack is a 7 km2 side pond and nearest Chatuge an arm of the
lake, and Tellico releases almost nothing (below). Barkley and Great Falls are on the Cumberland, outside the graph.
Tellico releases through a canal into Fort Loudoun reservoir, so its own release is about
zero and Fort Loudoun's includes the Little Tennessee: with observed releases at both
dams the routed water still adds up.

Output ``processed/hydro/dams.npz``: dam ids, names and reaches; observed release and
inflow [step, dam] on the model timeline (m3/s, NaN missing); TVA forecasts at every issue
[issue, lead 1-40, dam] with the issues' timeline steps; each dam's next dam downstream
(``downstream``, -1: none); for every reach the first dam on its way downstream
(``dam_below``, -1: none) and the share of its basin area whose water passes a dam before
reaching it (``regulated``). ``dams.csv`` lists the matches and checks.
"""

import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

from . import paths as P
from .data import Arrays, Network, timeline_index

RAW = P.ROOT / "raw/tva_dams"
FLOWLINES = (
    P.ROOT / "raw/geography/nhdplus_v21/extracted/NHDPlusV21_MS_06_NHDSnapshot_09/NHDPlusMS/"
    "NHDPlus06/NHDSnapshot/Hydrography/NHDFlowline.shp"
)
OUTSIDE = {"BARK2", "GGRT1"}  # Cumberland basin
# Reaches set by hand (see above), each the outlet of the given reservoir reach: Nickajack's
# Tennessee River reservoir (56,354 km2), Nottely's (555 km2; the dam drains 214 sq mi), the
# Little Tennessee at Tellico Dam (6,801 km2) and Chatuge's (490 km2; the reach ending
# nearest is an arm of the lake, 358 km2).
RESERVOIR = {"NKJT1": 19696757, "NOTG1": 19680543, "TDTT1": 19720259, "CHAN7": 19674829}
SEARCH_M = 3000
FLOW_RATIO = (0.5, 2.0)
LEADS = P.HORIZON


class Graph:
    """Main downstream path, waterbodies and NWM mean flow of every reach."""

    def __init__(self, network):
        nodes = pd.read_parquet(
            P.PROCESSED / "graph_candidate/nodes.parquet",
            columns=["comid", "gnis_name", "totdasqkm", "wbareatype", "wbareacomi"],
        ).set_index("comid").reindex(network.ids)
        self.ids = network.ids
        self.area = nodes.totdasqkm.to_numpy()
        self.name = nodes.gnis_name.fillna("").to_numpy()
        self.waterbody = nodes.wbareacomi.fillna(0).astype(np.int64).to_numpy()
        self.lake = nodes.wbareatype.isin(["LakePond", "Reservoir"]).to_numpy()
        self.mean = np.load(P.HYDRO / "reach_stats.npz")["mean_raw"]  # NaN at lake reaches
        self.down = {}
        for s, t in network.edges:
            self.down.setdefault(int(s), []).append(int(t))

    def next(self, i):
        """The reach below i on the main path (largest drainage area), or None."""
        below = self.down.get(i)
        return max(below, key=lambda j: self.area[j]) if below else None

    def outlet(self, i):
        """Last reach of i's waterbody along the main path."""
        while (j := self.next(i)) is not None and self.waterbody[i] > 0 and self.waterbody[j] == self.waterbody[i]:
            i = j
        return i

    def mean_below(self, i, source=False):
        """NWM mean flow at i or at the first reach below with one (lakes have none);
        with ``source`` also that reach."""
        while not (np.isfinite(self.mean[i]) and self.mean[i] > 0):
            if (i := self.next(i)) is None:
                return (np.nan, -1) if source else np.nan
        return (float(self.mean[i]), i) if source else float(self.mean[i])


def match(graph, meta, release):
    """Reach of every dam on the graph, with how it was found and its checks."""
    import geopandas as gpd
    from shapely.geometry import Point

    lines = gpd.read_file(FLOWLINES, columns=["COMID"])
    index = pd.Index(graph.ids)
    lines = lines[lines.COMID.isin(graph.ids)].to_crs(5070)
    lines["i"] = index.get_indexer(lines.COMID)
    geometry = dict(zip(lines.i, lines.geometry))
    rows = []
    for d in meta.itertuples():
        if d.NWSID in OUTSIDE:
            continue
        point = gpd.GeoSeries([Point(d.longitude, d.latitude)], crs=4326).to_crs(5070).iloc[0]
        if d.NWSID in RESERVOIR:
            i, how, dist = graph.outlet(int(index.get_loc(RESERVOIR[d.NWSID]))), "fixed", np.nan
        else:
            near = lines.assign(dist=lines.distance(point)).query("dist < @SEARCH_M")
            candidates = []
            for j, line in zip(near.i, near.geometry):
                flow, source = graph.mean_below(int(j), source=True)
                # The flow must be the reach's own river, not the river below a reservoir that
                # a side stream enters.
                if not FLOW_RATIO[0] < release[d.NWSID] / flow < FLOW_RATIO[1] or graph.area[j] < 0.5 * graph.area[source]:
                    continue
                end = line.geoms[-1] if line.geom_type == "MultiLineString" else line
                gap = Point(end.coords[-1]).distance(point)  # NHD lines run downstream
                candidates.append((int(gap // 250), -flow, gap, int(j)))
            if not candidates:
                raise ValueError(f"{d.NWSID}: no reach within {SEARCH_M} m carries the dam's flow")
            _, _, dist, i = min(candidates)
            how = "reservoir outlet" if graph.lake[i] else "river reach"
        ratio = release[d.NWSID] / graph.mean_below(i)
        if release[d.NWSID] > 0.5 and not FLOW_RATIO[0] < ratio < FLOW_RATIO[1]:
            raise ValueError(f"{d.NWSID}: reach {graph.ids[i]} carries {1 / ratio:.2f} x the dam's mean release in NWM")
        rows.append(
            dict(
                dam=d.NWSID, name=d.Descript, group=d.FCSTGroup, reach=int(i), comid=int(graph.ids[i]),
                area_km2=round(float(graph.area[i])), how=how, distance_m=float(dist) if np.isnan(dist) else round(float(dist)),
                mean_release=round(release[d.NWSID], 1), release_over_nwm=round(float(ratio), 2),
            )
        )
    table = pd.DataFrame(rows)
    if table.reach.duplicated().any():
        raise ValueError(f"Dams share a reach: {table[table.reach.duplicated(keep=False)].dam.tolist()}")
    return table


def regulation(network, reaches):
    """For every reach the first dam on its way downstream (index into ``reaches``, itself
    included; -1: none) and the share of its basin area whose water passes a dam first."""
    rows = np.repeat(np.arange(network.n), np.diff(network.indptr))
    dam_below = np.full(network.n, -1)
    # Nested basins: the nearest dam wins. Order by reach count: a basin holding another
    # always has more reaches (its share-weighted area can be smaller below a divergence).
    for k in np.argsort(-np.diff(network.indptr)[reaches]):
        dam_below[network.members[network.indptr[reaches[k]] : network.indptr[reaches[k] + 1]]] = k
    # Reach i is at or below dam k when dam k's reach is in i's basin.
    below = np.zeros((len(reaches), network.n), dtype=bool)
    is_dam = np.full(network.n, -1)
    is_dam[reaches] = np.arange(len(reaches))
    hit = is_dam[network.members] >= 0
    below[is_dam[network.members[hit]], rows[hit]] = True
    # Upstream reach j counts for i when j's first dam lies between j and i.
    k = dam_below[network.members]
    passes = (k >= 0) & below[np.maximum(k, 0), rows]
    weight = network.share * network.area[network.members] / network.basin_area[rows]
    regulated = np.bincount(rows, weights=weight * passes, minlength=network.n)
    return dam_below, np.clip(regulated, 0, 1).astype(np.float32)


def observed(table, times):
    """Release and inflow [step, dam] (m3/s, NaN missing) on the timeline ``times``."""
    out = np.full((len(times), len(table)), np.nan, dtype=np.float32)
    inflow = out.copy()
    for k, dam in enumerate(table.dam):
        with xr.open_dataset(RAW / f"observed/{dam}.nc") as ds:
            series = ds[["dam_outflow", "dam_inflow"]].to_dataframe().reindex(times)
        out[:, k] = series.dam_outflow.to_numpy()
        inflow[:, k] = series.dam_inflow.to_numpy()
    return out, inflow


def forecasts(table, times):
    """TVA's forecasts at every issue time: issue steps and release, inflow [issue, lead, dam]
    for leads 6-240 h (model leads 0-39), NaN where a dam has none."""
    issues = set()
    for dam in table.dam:
        path = RAW / f"forecast/{dam}.nc"
        if path.exists():
            with xr.open_dataset(path) as ds:
                issues.update(pd.DatetimeIndex(ds.forecast_time.values))
    issues = pd.DatetimeIndex(sorted(t for t in issues if times[0] <= t <= times[-1]))
    release = np.full((len(issues), LEADS, len(table)), np.nan, dtype=np.float32)
    inflow = release.copy()
    for k, dam in enumerate(table.dam):
        path = RAW / f"forecast/{dam}.nc"
        if not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            if ds.sizes.get("forecast_time", 0) == 0:  # a file with no forecasts
                continue
            leads = ds.lead_time.values
            if not np.array_equal(leads[: LEADS + 1], np.arange(0, 6 * (LEADS + 1), 6)):
                raise ValueError(f"{dam}: unexpected forecast leads")
            position = issues.get_indexer(pd.DatetimeIndex(ds.forecast_time.values))
            keep = position >= 0
            release[position[keep], :, k] = ds.dam_outflow.values[keep, 1 : LEADS + 1]
            inflow[position[keep], :, k] = ds.dam_inflow.values[keep, 1 : LEADS + 1]
    steps = np.array([timeline_index(t) for t in issues], dtype=np.int64)
    return steps, release, inflow


def build():
    network = Network()
    graph = Graph(network)
    times = Arrays().times
    meta = pd.read_csv(RAW / "selected_dams.csv")
    release = {}
    for dam in meta.NWSID:
        with xr.open_dataset(RAW / f"observed/{dam}.nc") as ds:
            release[dam] = float(ds.dam_outflow.sel(time=slice("1979-02-01", "2023-01-31")).mean())
    table = match(graph, meta, release)
    reaches = table.reach.to_numpy()
    dam_below, regulated = regulation(network, reaches)
    below_dam = [dam_below[graph.next(int(r))] if graph.next(int(r)) is not None else -1 for r in reaches]
    table["downstream_dam"] = [table.dam[k] if k >= 0 else "" for k in below_dam]
    table["regulated_share"] = regulated[reaches].round(2)
    outflow, inflow = observed(table, times)
    issue_steps, plan_release, plan_inflow = forecasts(table, times)
    out = P.HYDRO / "dams.npz"
    np.savez_compressed(
        out,
        dam=table.dam.to_numpy().astype(str),
        name=table.name.to_numpy().astype(str),
        reach=reaches,
        comid=table.comid.to_numpy(),
        release=outflow,
        inflow=inflow,
        forecast_issue=issue_steps,
        forecast_release=plan_release,
        forecast_inflow=plan_inflow,
        regulated=regulated,
        dam_below=dam_below,
        downstream=np.array(below_dam, dtype=np.int64),
        start=np.array(str(times[0])),
    )
    table.to_csv(P.HYDRO / "dams.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.drop(columns=["name"]).to_string())
    print(
        f"{len(table)} dams on the graph; reaches with more than half their basin passing a dam first: "
        f"{int((regulated > 0.5).sum()):,} of {network.n:,}; any: {int((regulated > 0).sum()):,}; "
        f"{len(issue_steps):,} TVA forecast issues -> {out}"
    )


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()


if __name__ == "__main__":
    main()
