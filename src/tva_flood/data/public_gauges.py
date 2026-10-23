"""Gauge catalog built from public observations only (no TVA archive values).

The training catalog (``build-gauges``) prefers TVA's own six-hour series at TVA
gauges. For the paper every model must train, select and test on public data, so
this catalog keeps the same gauges, names, order and reaches as
``benchmark_gauges.nc`` (``download-usgs``) but takes its values from:

- CAMELS-H hourly streamflow (six-hour means, all six hours present) before the
  USGS start of ``benchmark_gauges.nc`` (2019-11-01), at its CAMELS-H sites;
- USGS NWIS from that date on (as ``benchmark_gauges.nc``).

Gauges that are not CAMELS-H sites have no values before the USGS start, so they
neither train nor select; they are never scored. Output ``<out>/gauges.nc`` is a
``data.gauges.Catalog`` (with ``usgs_id`` and ``camelsh``) and ``<out>/gauges.csv``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tva_flood.common import atomic_json, sha

from .gauges import hourly_to_six_hour


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument(
        "--benchmark-gauges",
        type=Path,
        help="Default: <root>/processed/benchmarks/benchmark_gauges.nc",
    )
    p.add_argument("--out", type=Path, help="Default: <root>/processed/gauges_public")
    a = p.parse_args()
    root = a.root.resolve()
    source = (a.benchmark_gauges or root / "processed/benchmarks/benchmark_gauges.nc").resolve()
    out = (a.out or root / "processed/gauges_public").resolve()
    if (out / "gauges.nc").exists():
        p.error(f"{out / 'gauges.nc'} exists")
    with xr.open_dataset(source) as data:
        data = data.load()
    times = pd.DatetimeIndex(data.time.values)
    usgs_from = pd.Timestamp(data.attrs["usgs_from"])
    before = int(times.searchsorted(usgs_from))
    observed = data.observed.values.astype(np.float32).copy()
    observed[:before] = np.nan
    usgs = data.usgs_id.values.astype(str)
    camelsh = data.camelsh.values.astype(bool)
    for g in np.flatnonzero(camelsh):
        with xr.open_dataset(root / f"raw/camelsh/flow/{usgs[g]}_hourly.nc") as flow:
            hourly = flow.streamflow.to_series()
        observed[:before, g] = hourly_to_six_hour(hourly, times)[:before]
    observed[observed < 0] = np.nan
    names = data.gauge.values.astype(str)
    table = pd.DataFrame(
        dict(
            gauge=names,
            source=np.where(camelsh, "camelsh", "usgs"),
            comid=data.comid.values,
            usgs_id=usgs,
            camelsh=camelsh,
            tva_eval=False,
        )
    )
    finite = np.isfinite(observed)
    table["n_values"] = finite.sum(axis=0)
    table["values_before_usgs"] = finite[:before].sum(axis=0)
    table["first"] = [str(times[f][0]) if f.any() else None for f in finite.T]
    table["last"] = [str(times[f][-1]) if f.any() else None for f in finite.T]
    out.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        dict(observed=(("time", "gauge"), observed)),
        coords=dict(
            time=times,
            gauge=names,
            comid=("gauge", table.comid.to_numpy()),
            source=("gauge", table.source.to_numpy()),
            tva_eval=("gauge", np.zeros(len(names), dtype=bool)),
            usgs_id=("gauge", usgs),
            camelsh=("gauge", camelsh),
        ),
        attrs=dict(
            units="m3 s-1",
            interval="six-hour mean over [t, t+6h) UTC",
            usgs_from=str(usgs_from),
            source=f"before {usgs_from}: CAMELS-H hourly (CAMELS-H sites only); "
            f"from {usgs_from}: USGS NWIS IV 00060",
        ),
    ).to_netcdf(out / "gauges.nc", encoding=dict(observed=dict(zlib=True, complevel=1)))
    table.to_csv(out / "gauges.csv", index=False)
    atomic_json(
        out / "gauges.json",
        dict(
            gauges=len(table),
            camelsh_sites=int(camelsh.sum()),
            gauges_with_training_values=int((table.values_before_usgs > 0).sum()),
            usgs_from=str(usgs_from),
            benchmark_gauges_sha256=sha(source),
            tva_data_used=False,
        ),
    )
    print(
        f"{out / 'gauges.nc'}: {len(table)} gauges, {int(camelsh.sum())} CAMELS-H sites, "
        f"{int((table.values_before_usgs > 0).sum())} with values before {usgs_from:%Y-%m-%d}"
    )


if __name__ == "__main__":
    main()
