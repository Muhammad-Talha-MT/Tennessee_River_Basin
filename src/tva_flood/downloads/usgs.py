"""Observed flow for the benchmark test period, straight from USGS.

The paper scores every model against public USGS observations at the catalog's
gauges, not against TVA's archive. For each catalog gauge (TVA's are USGS
stations too) this downloads USGS instantaneous discharge (parameter 00060) from
the NWIS instantaneous-values service, averages it to whole UTC hours, then to
six-hour means over [t, t+6h) with all six hours present (as ``data.gauges``),
and converts ft3/s to m3/s.

``benchmark_gauges.nc`` is a ``data.gauges.Catalog`` with the same gauges and
reaches as the training catalog: values before ``--start`` are the catalog's,
values from ``--start`` on are USGS only. ``benchmark_gauges.csv`` lists each
gauge's USGS id and whether it is a CAMELS-H (CAMELSH) site, the benchmark set.
"""

import argparse
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import xarray as xr

from tva_flood.common import atomic_json
from tva_flood.data import tva
from tva_flood.data.gauges import hourly_to_six_hour, timeline

SERVICE = "https://waterservices.usgs.gov/nwis/iv/"
CFS = 0.028316846592


def usgs_id(value):
    """USGS station id as text: archive values lose leading zeros (3512000 -> 03512000)."""
    text = str(int(float(value)))
    if not text.startswith("0"):
        text = "0" + text
    return text.zfill(8)


def station_ids(catalog_csv, archive):
    """USGS id of every catalog gauge (CAMELS-H gauges are named by theirs)."""
    table = pd.read_csv(catalog_csv, dtype={"gauge": str})
    ids = {}
    with tempfile.TemporaryDirectory() as scratch:
        folder = tva.extract(archive, Path(scratch))
        attributes = pd.read_csv(folder / tva.ATTRIBUTES).set_index("tva_nwsid")
        for gauge in table.gauge[table.source == "tva"]:
            with xr.open_dataset(folder / "observed" / f"{gauge}.nc") as data:
                value = data.attrs.get("usgs_id")
            if value in (None, "") or not str(value).strip():
                value = attributes.camelsh_usgs_id.get(gauge)
            if value is None or pd.isna(value):
                raise ValueError(f"No USGS id for TVA gauge {gauge}")
            ids[gauge] = usgs_id(value)
    table["usgs_id"] = [ids.get(g, g) for g, s in zip(table.gauge, table.source)]
    return table


def fetch(site, start, stop):
    """Instantaneous discharge (ft3/s) as a UTC-indexed series; empty if none."""
    query = dict(
        format="json",
        sites=site,
        parameterCd="00060",
        startDT=start.strftime("%Y-%m-%dT%H:%MZ"),
        endDT=stop.strftime("%Y-%m-%dT%H:%MZ"),
        siteStatus="all",
    )
    for attempt in range(6):
        try:
            with urlopen(f"{SERVICE}?{urlencode(query)}", timeout=300) as response:
                payload = json.load(response)
            break
        except Exception as exc:  # network hiccups and throttling
            if attempt == 5:
                raise
            print(f"Retry {site} {start:%Y}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(2**attempt)
    series = []
    for record in payload["value"]["timeSeries"]:
        for block in record["values"]:
            missing = float(record["variable"]["noDataValue"])
            for v in block["value"]:
                value = float(v["value"])
                if value != missing and value >= 0:
                    series.append((v["dateTime"], value))
    if not series:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(series, columns=["time", "flow"])
    index = pd.to_datetime(frame.time, utc=True).dt.tz_localize(None)
    # Several methods or sensors at one site: average duplicates.
    return pd.Series(frame.flow.to_numpy(), index=index).groupby(level=0).mean().sort_index()


def six_hour(site, start, stop, times):
    """Six-hour means [m3/s] on ``times`` from hourly means of instantaneous values."""
    parts = []
    year = start
    while year < stop:
        upper = min(stop, pd.Timestamp(year.year + 1, 1, 1))
        parts.append(fetch(site, year, upper))
        year = upper
    raw = pd.concat([p for p in parts if len(p)]) if any(len(p) for p in parts) else None
    if raw is None:
        return np.full(len(times), np.nan)
    raw = raw[~raw.index.duplicated()]
    hourly = raw.resample("h").mean()
    return hourly_to_six_hour(hourly * CFS, times)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--gauges", type=Path, help="Default: <root>/processed/gauges/gauges.nc")
    p.add_argument(
        "--tva-archive", type=Path, default=Path("/data/tva_data/raw/tva/TVA_headwater.zip")
    )
    p.add_argument("--start", default="2019-11-01", help="First time taken from USGS")
    p.add_argument("--stop", default="2026-01-01")
    p.add_argument("--out", type=Path, help="Default: <root>/processed/benchmarks")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0, help="Only the first N gauges (testing)")
    a = p.parse_args()
    root = a.root.resolve()
    catalog_nc = (a.gauges or root / "processed/gauges/gauges.nc").resolve()
    out = (a.out or root / "processed/benchmarks").resolve()
    out.mkdir(parents=True, exist_ok=True)
    start, stop = pd.Timestamp(a.start), pd.Timestamp(a.stop)
    table = station_ids(catalog_nc.with_name("gauges.csv"), a.tva_archive)
    camelsh = pd.read_csv(
        root / "raw/camelsh/metadata/tennessee_camelsh_gauges.csv", dtype={"STAID": str}
    )
    hourly_release = set(camelsh.STAID[camelsh.in_hourly_release])
    table["camelsh"] = table.usgs_id.isin(hourly_release)
    if a.limit:
        table = table.head(a.limit)
    times = timeline()
    window = times[(times >= start) & (times < stop)]
    with ThreadPoolExecutor(a.workers) as pool:
        columns = list(pool.map(lambda s: six_hour(s, start, stop, window), table.usgs_id))
    with xr.open_dataset(catalog_nc) as data:
        catalog = data.load()
    keep = [list(catalog.gauge.values).index(g) for g in table.gauge]
    observed = catalog.observed.values[:, keep].astype(np.float32)
    position = times.get_indexer(window)
    observed[position[0] :] = np.nan
    observed[position] = np.stack(columns, axis=1)
    observed[observed < 0] = np.nan
    finite = np.isfinite(observed[position])
    table["usgs_values"] = finite.sum(axis=0)
    dataset = xr.Dataset(
        dict(observed=(("time", "gauge"), observed)),
        coords=dict(
            time=times,
            gauge=table.gauge.to_numpy(),
            comid=("gauge", table.comid.to_numpy()),
            source=("gauge", table.source.to_numpy()),
            tva_eval=("gauge", table.tva_eval.to_numpy()),
            usgs_id=("gauge", table.usgs_id.to_numpy()),
            camelsh=("gauge", table.camelsh.to_numpy()),
        ),
        attrs=dict(
            units="m3 s-1",
            interval="six-hour mean over [t, t+6h) UTC",
            usgs_from=str(start),
            source=f"before {start}: {catalog_nc}; from {start}: USGS NWIS IV 00060",
        ),
    )
    dataset.to_netcdf(
        out / "benchmark_gauges.nc", encoding=dict(observed=dict(zlib=True, complevel=1))
    )
    table.to_csv(out / "benchmark_gauges.csv", index=False)
    atomic_json(
        out / "benchmark_gauges.json",
        dict(
            gauges=len(table),
            camelsh_sites=int(table.camelsh.sum()),
            usgs_from=str(start),
            usgs_until=str(stop),
            gauges_without_usgs_data=table.gauge[table.usgs_values == 0].tolist(),
        ),
    )
    print(
        f"{out / 'benchmark_gauges.nc'}: {len(table)} gauges, {int(table.camelsh.sum())} CAMELS-H "
        f"sites; {int((table.usgs_values == 0).sum())} without USGS data from {start:%Y-%m-%d}"
    )


if __name__ == "__main__":
    main()
