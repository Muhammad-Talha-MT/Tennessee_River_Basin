"""NOAA National Water Model medium-range forecasts for the benchmark, 00Z cycle, 2020-2025.

Two things per issue day, both from the public archive gs://national-water-model
(read over HTTPS, only the parts needed):

- **Streamflow** of the deterministic member (``medium_range_mem1``,
  ``channel_rt_1``) at every catalog gauge's reach (NWM feature_id = NHDPlus v2
  COMID) at leads 3, 6, ..., 240 h. NWM output is instantaneous; a six-hour mean
  over [t+6k, t+6k+6) is the trapezoid of the values at 6k, 6k+3 and 6k+6 h
  (for k = 0, the mean of 3 and 6 h, as 0 h is not in the files).
- **The rain that NWM was forced with** (``forcing_medium_range`` RAINRATE,
  hourly mean rate on the 1 km NWM grid) for leads up to ``--rain-hours``,
  summed to six-hour totals (mm) and averaged over every reach's catchment: each
  AORC cell of ``aorc_mapping`` takes the nearest NWM cell, and the AORC
  area weights give the reach mean. Our models get exactly the rain NWM had.

Versions in the period: v2.0 (to 2021-04-20, 3-hourly channel output), v2.1 (to
2023-09-20) and v3.0. One file per month in ``--out``; reruns skip finished
months. Missing archive files are recorded and left as NaN.
"""

import argparse
import io
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

BUCKET = "https://storage.googleapis.com/national-water-model"
FLOW_LEADS = np.arange(3, 241, 3)
WAIT = 1800  # seconds a single file may take before the month fails and is retried


def get(url, attempts=6):
    import requests

    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=180)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.content
        except Exception as exc:
            if attempt == attempts - 1:
                print(f"FAILED {url}: {type(exc).__name__}: {exc}", flush=True)
                return None
            time.sleep(2**attempt)


def flow_url(day, lead):
    return (
        f"{BUCKET}/nwm.{day:%Y%m%d}/medium_range_mem1/"
        f"nwm.t00z.medium_range.channel_rt_1.f{lead:03d}.conus.nc"
    )


def forcing_url(day, lead):
    return (
        f"{BUCKET}/nwm.{day:%Y%m%d}/forcing_medium_range/"
        f"nwm.t00z.medium_range.forcing.f{lead:03d}.conus.nc"
    )


class Grid:
    """Reach-mean rain from the NWM 1 km grid through the AORC area weights."""

    def __init__(self, root, ids, sample_url):
        import h5py
        from pyproj import Proj

        content = get(sample_url)
        if content is None:
            raise ValueError(f"Cannot read {sample_url}")
        with h5py.File(io.BytesIO(content), "r") as h:
            x, y = h["x"][:], h["y"][:]
            proj4 = h["RAINRATE"].attrs["proj4"].decode()
        mapping = np.load(root / "processed/aorc_mapping/grid_and_node_order.npz")
        weights = sp.load_npz(root / "processed/aorc_mapping/aorc_to_nodes_weights.npz").tocsr()
        order = pd.Index(mapping["node_comid"]).get_indexer(ids)
        if (order < 0).any():
            raise ValueError("A graph reach is missing from the AORC mapping")
        weights = weights[order]
        lon, lat = np.meshgrid(mapping["longitude"], mapping["latitude"])
        used = np.unique(weights.indices)
        # WRF-Hydro convention: geographic coordinates taken directly on the model sphere.
        cx, cy = Proj(proj4)(lon.ravel()[used], lat.ravel()[used])
        ix = np.clip(np.round((cx - x[0]) / (x[1] - x[0])).astype(int), 0, len(x) - 1)
        iy = np.clip(np.round((cy - y[0]) / (y[1] - y[0])).astype(int), 0, len(y) - 1)
        self.window = (slice(iy.min(), iy.max() + 1), slice(ix.min(), ix.max() + 1))
        width = ix.max() - ix.min() + 1
        cell = (iy - iy.min()) * width + (ix - ix.min())
        pick = sp.csr_matrix(
            (np.ones(len(used)), (used, cell)),
            shape=(weights.shape[1], (iy.max() - iy.min() + 1) * width),
        )
        self.matrix = (weights @ pick).tocsr()  # reach x window cell

    def reach_mean(self, field):
        return self.matrix @ np.nan_to_num(field).ravel()


def flow_task(day, lead, gauges):
    """Streamflow [gauge] (NaN where absent) and NWM version of one channel file."""
    import h5py

    url = flow_url(day, lead)
    content = get(url)
    if content is None:
        return None, "", url
    with h5py.File(io.BytesIO(content), "r") as h:
        index = pd.Index(h["feature_id"][:]).get_indexer(gauges)
        data = h["streamflow"]
        values = data[:]
        scale = data.attrs.get("scale_factor", [1.0])[0]
        fill = data.attrs.get("_FillValue", [None])[0]
        version = h.attrs.get("NWM_version_number", b"").decode()
    picked = np.where(index >= 0, values[np.maximum(index, 0)], fill)
    return np.where(picked == fill, np.nan, picked * scale), version, None


def rain_task(day, hour, window):
    """Hourly rain (mm) on the window of the NWM grid, or None if the file is missing."""
    import fsspec
    import h5py

    url = forcing_url(day, hour)
    # Only the compressed tiles over the basin are fetched (HTTP range requests).
    for attempt in range(6):
        try:
            import aiohttp

            # A whole-request limit: a connection that stalls must fail and retry.
            client = dict(timeout=aiohttp.ClientTimeout(total=240))
            with fsspec.open(
                url, "rb", block_size=2**21, cache_type="readahead", client_kwargs=client
            ).open() as f:
                with h5py.File(f, "r") as h:
                    data = h["RAINRATE"]
                    values = data[(0,) + window].astype(np.float64)
                    fill = data.attrs.get("_FillValue", [None])[0]
                    scale = data.attrs.get("scale_factor", [1.0])[0]
            break
        except FileNotFoundError:
            return None, url
        except Exception as exc:
            if attempt == 5:
                print(f"FAILED {url}: {type(exc).__name__}: {exc}", flush=True)
                return None, url
            time.sleep(2**attempt)
    if fill is not None:
        values[values == fill] = np.nan
    return values * scale * 3600.0, None


def month(days, gauges, grid, rain_steps, pool):
    """Flow [day, lead, gauge], six-hour rain [day, step, reach], versions and missing files."""
    flow = np.full((len(days), len(FLOW_LEADS), len(gauges)), np.nan)
    hourly = np.full((len(days), 6 * rain_steps, grid.matrix.shape[0]), np.nan, np.float32)
    versions, missing = [""] * len(days), []
    flows = {
        pool.submit(flow_task, day, int(lead), gauges): (d, i)
        for d, day in enumerate(days)
        for i, lead in enumerate(FLOW_LEADS)
    }
    rains = {
        pool.submit(rain_task, day, hour, grid.window): (d, hour)
        for d, day in enumerate(days)
        for hour in range(1, 6 * rain_steps + 1)
    }
    for future, (d, i) in flows.items():
        values, version, url = future.result(timeout=WAIT)
        if values is None:
            missing.append(url)
            continue
        flow[d, i] = values
        versions[d] = version or versions[d]
    for future, (d, hour) in rains.items():
        values, url = future.result(timeout=WAIT)
        if values is None:
            missing.append(url)
            continue
        hourly[d, hour - 1] = grid.reach_mean(values)
    totals = hourly.reshape(len(days), rain_steps, 6, -1)
    complete = np.isfinite(totals).all(axis=2)
    rain = np.where(complete, totals.sum(axis=2), np.nan)
    return flow, rain, versions, missing


def six_hour_flow(flow):
    """Six-hour mean flow [.., 40, gauge] from instantaneous values at 3, 6, ..., 240 h."""
    values = flow.reshape(flow.shape[:-2] + (40, 2, flow.shape[-1]))
    middle, end = values[..., 0, :], values[..., 1, :]
    start = np.concatenate([middle[..., :1, :], end[..., :-1, :]], axis=-2)
    mean = start / 4 + middle / 2 + end / 4
    mean[..., 0, :] = (middle[..., 0, :] + end[..., 0, :]) / 2
    return mean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--preparation", type=Path, help="For the reach order (graph.npz)")
    p.add_argument(
        "--gauges", type=Path, help="Default: <root>/processed/benchmarks/benchmark_gauges.nc"
    )
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--stop", default="2026-01-01")
    p.add_argument("--rain-hours", type=int, default=96, help="Forecast rain kept to this lead")
    p.add_argument("--out", type=Path, help="Default: <root>/processed/benchmarks/nwm_medium_range")
    p.add_argument("--workers", type=int, default=32, help="Parallel download processes")
    p.add_argument("--days", type=int, default=0, help="Only the first N days (testing)")
    a = p.parse_args()
    import xarray as xr

    root = a.root.resolve()
    prep = (a.preparation or root / "processed/pretraining_full").resolve()
    out = (a.out or root / "processed/benchmarks/nwm_medium_range").resolve()
    out.mkdir(parents=True, exist_ok=True)
    with np.load(prep / "graph.npz") as graph:
        ids = graph["feature_id"].copy()
    gauges_nc = (a.gauges or root / "processed/benchmarks/benchmark_gauges.nc").resolve()
    with xr.open_dataset(gauges_nc) as data:
        names = data.gauge.values.astype(str)
        comids = data.comid.values.astype(np.int64)
    rain_steps = a.rain_hours // 6
    days = pd.date_range(a.start, a.stop, freq="D", inclusive="left")
    if a.days:
        days = days[: a.days]
    grid = Grid(root, ids, forcing_url(days[0], 1))
    print(
        f"NWM medium range: {len(days)} issue days, {len(names)} gauges, rain to {a.rain_hours} h"
    )
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(a.workers, mp_context=context) as pool:
        for period, group in pd.Series(days).groupby(days.to_period("M")):
            target = out / f"{period}.npz"
            if target.exists():
                continue
            began = time.perf_counter()
            group = list(group)
            flow, rain, versions, missing = month(group, comids, grid, rain_steps, pool)
            save(target, group, flow, rain, versions, missing, names, comids, ids)
            print(
                f"{period}: {len(group)} days, versions {sorted(set(versions))}, "
                f"{len(missing)} missing files, {time.perf_counter() - began:.0f} s",
                flush=True,
            )
    print(f"Done: {out}", flush=True)
    # Native libraries (HDF5, aiohttp) sometimes corrupt memory while the interpreter shuts
    # down; the data is already on disk, so leave without running their cleanup.
    os._exit(0)


def save(target, days, flow, rain, versions, missing, names, comids, ids):
    partial = target.with_name(target.stem + ".part.npz")
    np.savez(
        partial,
        issue=np.array([str(d.date()) for d in days]),
        flow_leads_h=FLOW_LEADS,
        flow=flow.astype(np.float32),
        rain=rain.astype(np.float16),
        version=np.array(versions),
        gauge=names,
        comid=comids,
        reach=ids,
    )
    partial.rename(target)
    target.with_name(target.stem + ".missing.json").write_text(json.dumps(missing, indent=1))


if __name__ == "__main__":
    main()
