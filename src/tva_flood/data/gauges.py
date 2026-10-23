"""Observed-flow gauge catalog: TVA headwater gauges and every CAMELS-H gauge in Tennessee.

A forecast issued at t may use everything observed up to t. These gauges are that
observed information: their six-hour flow is fed into the model as history, and
fine-tuning targets are the same series. Each gauge is matched to the graph
reach nearest its coordinates whose total drainage area is within a tolerance of
the gauge's; gauges without such a reach are listed in ``excluded.csv``. One
gauge is kept per reach, preferring TVA's own series, then the closest area.

Six-hour flow is the mean over [t, t+6h) in UTC. CAMELS-H hourly values are
averaged only when all six hours are present (TVA's archive is already in
six-hour means). The catalog holds ``observed[time, gauge]`` in m3/s on the
six-hour timeline of the prepared data, NaN where unobserved.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tva_flood.common import atomic_json, sha

from . import tva

START, STOP = pd.Timestamp("1979-02-01"), pd.Timestamp("2026-01-01")
MIN_VALUES = 1460  # one year of six-hour flow


def timeline():
    return pd.date_range(START, STOP, freq="6h", inclusive="left")


def hourly_to_six_hour(series, times):
    """Means over [t, t+6h) of hourly values labelled t..t+5h; NaN unless all six exist."""
    hourly = pd.date_range(times[0], periods=6 * len(times), freq="h")
    values = series[~series.index.duplicated()].reindex(hourly).to_numpy(dtype=np.float64)
    blocks = values.reshape(len(times), 6)
    complete = np.isfinite(blocks).all(axis=1)
    return np.where(complete, blocks.mean(axis=1), np.nan)


def match_reaches(root, ids, points, max_distance, ratio):
    """Graph reach nearest each (lon, lat, area_km2) whose drainage area agrees with the gauge's."""
    import geopandas as gpd
    from shapely.geometry import Point

    nodes = pd.read_parquet(
        root / "processed/graph_candidate/nodes.parquet", columns=["comid", "totdasqkm"]
    )
    nodes = nodes[nodes.comid.isin(ids)]
    found = list((root / "raw/geography/nhdplus_v21/extracted").rglob("NHDFlowline.shp"))
    if len(found) != 1:
        raise ValueError("Expected one NHDFlowline.shp under raw/geography/nhdplus_v21")
    lines = gpd.read_file(found[0], columns=["COMID", "geometry"]).rename(
        columns={"COMID": "comid"}
    )
    lines = lines.merge(nodes, on="comid").to_crs(5070)
    result = []
    for lon, lat, area in points:
        point = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(5070).iat[0]
        near = lines[lines.distance(point) < max_distance].copy()
        near["distance"] = near.distance(point)
        near["ratio"] = near.totdasqkm / area
        near = near[near.ratio.between(*ratio)].sort_values("distance")
        if len(near):
            best = near.iloc[0]
            result.append((int(best.comid), float(best.distance), float(best.ratio)))
        else:
            result.append((None, np.nan, np.nan))
    return result


def build(a):
    root = a.root.resolve()
    prep = (a.preparation or root / "processed/pretraining_full").resolve()
    out = (a.out or root / "processed/gauges").resolve()
    if (out / "gauges.nc").exists():
        raise ValueError(f"{out}/gauges.nc exists; choose a new --out")
    with np.load(prep / "graph.npz") as graph:
        ids = graph["feature_id"].copy()
    times = timeline()
    rows, series = [], []
    with tempfile.TemporaryDirectory() as scratch:
        folder = tva.extract(a.tva_archive, Path(scratch))
        evaluated = set(tva.gauges(folder, ids, forecasts=True).tva_nwsid)
        stations = tva.gauges(folder, ids, forecasts=False)
        observed = tva.observed_flow(folder, stations, times)
    for s, row in enumerate(stations.itertuples()):
        rows.append(
            dict(
                gauge=row.tva_nwsid,
                source="tva",
                comid=int(row.comid),
                area_km2=float(row.tva_area_sqmi) * 2.58999,
                distance_m=0.0,
                area_ratio=np.nan,
                tva_eval=row.tva_nwsid in evaluated,
            )
        )
        series.append(observed[:, s])
    metadata = pd.read_csv(
        root / "raw/camelsh/metadata/tennessee_camelsh_gauges.csv", dtype={"STAID": str}
    )
    available = metadata[metadata.in_hourly_release].reset_index(drop=True)
    matches = match_reaches(
        root,
        ids,
        list(zip(available.LNG_GAGE, available.LAT_GAGE, available.DRAIN_SQKM)),
        a.max_distance_m,
        (a.min_area_ratio, a.max_area_ratio),
    )
    excluded = [
        dict(gauge=g, source="camelsh", reason="no graph reach with matching drainage area")
        for g, m in zip(available.STAID, matches)
        if m[0] is None
    ]
    excluded += [
        dict(gauge=g, source="camelsh", reason="not in the hourly CAMELS-H release")
        for g in metadata.STAID[~metadata.in_hourly_release]
    ]
    for gauge, area, (comid, distance, ratio) in zip(
        available.STAID, available.DRAIN_SQKM, matches
    ):
        if comid is None:
            continue
        with xr.open_dataset(root / f"raw/camelsh/flow/{gauge}_hourly.nc") as data:
            hourly = data.streamflow.to_series()
        rows.append(
            dict(
                gauge=gauge,
                source="camelsh",
                comid=comid,
                area_km2=float(area),
                distance_m=distance,
                area_ratio=ratio,
                tva_eval=False,
            )
        )
        series.append(hourly_to_six_hour(hourly, times))
    table = pd.DataFrame(rows)
    values = np.stack(series, axis=1).astype(np.float32)
    values[values < 0] = np.nan
    finite = np.isfinite(values)
    table["n_values"] = finite.sum(axis=0)
    table["first"] = [str(times[f][0]) if f.any() else None for f in finite.T]
    table["last"] = [str(times[f][-1]) if f.any() else None for f in finite.T]
    # One gauge per reach: TVA's series, else the closest drainage area.
    table["rank"] = (table.source != "tva").astype(int) * 10 + np.abs(
        np.log(table.area_ratio.fillna(1.0))
    )
    keep = table.sort_values("rank").drop_duplicates("comid").index
    for index in table.index.difference(keep):
        excluded.append(
            dict(gauge=table.gauge[index], source=table.source[index], reason="reach shared")
        )
    for index in keep:
        if table["n_values"][index] < MIN_VALUES:
            excluded.append(
                dict(gauge=table.gauge[index], source=table.source[index], reason="under a year")
            )
    keep = [i for i in sorted(keep) if table["n_values"][i] >= MIN_VALUES]
    table = table.loc[keep].drop(columns="rank").reset_index(drop=True)
    values = values[:, keep]
    out.mkdir(parents=True, exist_ok=True)
    dataset = xr.Dataset(
        dict(observed=(("time", "gauge"), values)),
        coords=dict(
            time=times,
            gauge=table.gauge.to_numpy(),
            comid=("gauge", table.comid.to_numpy()),
            source=("gauge", table.source.to_numpy()),
            tva_eval=("gauge", table.tva_eval.to_numpy()),
        ),
        attrs=dict(
            units="m3 s-1",
            interval="six-hour mean over [t, t+6h) UTC",
            tva_archive_sha256=sha(a.tva_archive),
        ),
    )
    dataset.to_netcdf(out / "gauges.nc", encoding=dict(observed=dict(zlib=True, complevel=1)))
    table.to_csv(out / "gauges.csv", index=False)
    pd.DataFrame(excluded).to_csv(out / "excluded.csv", index=False)
    atomic_json(
        out / "gauges.json",
        dict(
            gauges=len(table),
            tva=int((table.source == "tva").sum()),
            tva_evaluation=int(table.tva_eval.sum()),
            camelsh=int((table.source == "camelsh").sum()),
            excluded=len(excluded),
            max_distance_m=a.max_distance_m,
            area_ratio=[a.min_area_ratio, a.max_area_ratio],
        ),
    )
    print(
        f"Gauge catalog: {out}; {len(table)} gauges ({int((table.source == 'tva').sum())} TVA, "
        f"{int((table.source == 'camelsh').sum())} CAMELS-H); {len(excluded)} excluded "
        f"(see excluded.csv)"
    )


class Catalog:
    """Observed six-hour flow at every catalog gauge, on the catalog timeline."""

    def __init__(self, path):
        path = Path(path)
        with xr.open_dataset(path) as data:
            self.times = pd.DatetimeIndex(data.time.values)
            self.observed = data.observed.values.astype(np.float32)
            self.comid = data.comid.values.astype(np.int64)
            self.source = data.source.values.astype(str)
            self.tva_eval = data.tva_eval.values.astype(bool)
            self.names = data.gauge.values.astype(str)
            # CAMELS-H sites (the benchmark set); catalogs from download-usgs and
            # build-public-gauges record it, the training catalog by source.
            self.camelsh = (
                data.camelsh.values.astype(bool)
                if "camelsh" in data.coords
                else self.source == "camelsh"
            )
        self.path = path

    def __len__(self):
        return len(self.comid)

    def on(self, times):
        """Observed flow [time, gauge] on another timeline (NaN where the catalog lacks it)."""
        index = self.times.get_indexer(times)
        values = np.full((len(times), len(self)), np.nan, dtype=np.float32)
        found = index >= 0
        values[found] = self.observed[index[found]]
        return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--preparation", type=Path)
    p.add_argument(
        "--tva-archive", type=Path, default=Path("/data/tva_data/raw/tva/TVA_headwater.zip")
    )
    p.add_argument("--out", type=Path, help="Default: <root>/processed/gauges")
    p.add_argument("--max-distance-m", type=float, default=1500)
    p.add_argument("--min-area-ratio", type=float, default=0.8)
    p.add_argument("--max-area-ratio", type=float, default=1.25)
    build(p.parse_args())


if __name__ == "__main__":
    main()
