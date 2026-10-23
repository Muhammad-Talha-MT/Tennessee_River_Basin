"""Public rain forecasts from NOAA GEFSv12 (control member), mapped to every reach.

Replaces TVA's rain forecast with public archives, 00Z cycle, six-hour totals:

- ``--source reforecast``: GEFSv12 reforecast, 2000-2019 (s3://noaa-gefs-retrospective),
  for training and validation with real forecast rain instead of observed rain;
- ``--source operational``: operational GEFSv12, from 2020-09-23 (s3://noaa-gefs-pds),
  for the test years. Earlier operational GEFS (v11, 1 degree) is a different system
  and is not used.

Only the six-hour accumulation messages (``APCP:surface:{s}-{s+6} hour acc``) for leads
up to ``--rain-hours`` are fetched, by byte range from the GRIB2 index files. Each
AORC cell of ``aorc_mapping`` takes the nearest 0.25-degree GEFS point, and the AORC
area weights give each reach's catchment mean (mm per six hours). One file per month
in ``--out``; reruns skip finished months; missing days stay NaN.
"""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

SOURCES = dict(
    reforecast="https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast",
    operational="https://noaa-gefs-pds.s3.amazonaws.com",
)
FIRST = dict(reforecast="2000-01-01", operational="2020-09-23")


def get(url, headers=None, attempts=6):
    import requests

    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=180)
            if response.status_code in (403, 404):
                return None
            response.raise_for_status()
            return response.content
        except Exception as exc:
            if attempt == attempts - 1:
                print(f"FAILED {url}: {type(exc).__name__}: {exc}", flush=True)
                return None
            time.sleep(2**attempt)


def messages(source, day, member, rain_hours):
    """(grib url, byte start, byte stop) of every six-hour rain message for one issue day."""
    wanted = []
    if source == "reforecast":
        base = f"{SOURCES[source]}/{day:%Y}/{day:%Y%m%d}00/{member}/Days:1-10/apcp_sfc_{day:%Y%m%d}00_{member}.grib2"
        files = [base] * (rain_hours // 6)
    else:
        files = [
            f"{SOURCES[source]}/gefs.{day:%Y%m%d}/00/atmos/pgrb2sp25/ge{member}.t00z.pgrb2s.0p25.f{h:03d}"
            for h in range(6, rain_hours + 1, 6)
        ]
    index_cache = {}
    for step, url in enumerate(files):
        if url not in index_cache:
            text = get(url + ".idx")
            index_cache[url] = None if text is None else text.decode().splitlines()
        lines = index_cache[url]
        if lines is None:
            wanted.append(None)
            continue
        label = f"APCP:surface:{6 * step}-{6 * step + 6} hour acc"
        found = None
        for i, line in enumerate(lines):
            if label in line:
                start = int(line.split(":")[1])
                stop = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else ""
                found = (url, start, stop)
                break
        wanted.append(found)
    return wanted


def decode(content):
    """Values [lat, lon] and grid (lat0, lon0, dlat, dlon, nlat, nlon) of one GRIB2 message."""
    import eccodes

    handle = eccodes.codes_new_from_message(content)
    try:
        get_ = lambda key: eccodes.codes_get(handle, key)  # noqa: E731
        nlon, nlat = get_("Ni"), get_("Nj")
        lat0, lon0 = (
            get_("latitudeOfFirstGridPointInDegrees"),
            get_("longitudeOfFirstGridPointInDegrees"),
        )
        dlon, dlat = get_("iDirectionIncrementInDegrees"), get_("jDirectionIncrementInDegrees")
        if not get_("jScansPositively"):
            dlat = -dlat
        values = eccodes.codes_get_values(handle).reshape(nlat, nlon)
        missing = eccodes.codes_get(handle, "missingValue")
    finally:
        eccodes.codes_release(handle)
    values = np.where(values == missing, np.nan, values)
    return values, (lat0, lon0, dlat, dlon, nlat, nlon)


def day_task(source, day, member, rain_hours, window):
    """Six-hour rain [step, window cells] (mm) for one issue day; NaN where missing."""
    rows, cols = window
    out = np.full((rain_hours // 6, len(rows)), np.nan, dtype=np.float32)
    for step, message in enumerate(messages(source, day, member, rain_hours)):
        if message is None:
            continue
        url, start, stop = message
        content = get(url, headers={"Range": f"bytes={start}-{stop}"})
        if content is None:
            continue
        values, _ = decode(content)
        out[step] = np.maximum(values[rows, cols], 0)
    return out


class Mapping:
    """AORC cells -> nearest GEFS point; reach means through the AORC area weights."""

    def __init__(self, root, ids, grid):
        lat0, lon0, dlat, dlon, nlat, nlon = grid
        mapping = np.load(root / "processed/aorc_mapping/grid_and_node_order.npz")
        weights = sp.load_npz(root / "processed/aorc_mapping/aorc_to_nodes_weights.npz").tocsr()
        order = pd.Index(mapping["node_comid"]).get_indexer(ids)
        if (order < 0).any():
            raise ValueError("A graph reach is missing from the AORC mapping")
        weights = weights[order]
        lon, lat = np.meshgrid(mapping["longitude"], mapping["latitude"])
        used = np.unique(weights.indices)
        row = np.clip(np.round((lat.ravel()[used] - lat0) / dlat).astype(int), 0, nlat - 1)
        col = np.round(((lon.ravel()[used] % 360) - lon0) / dlon).astype(int) % nlon
        cells, cell_index = np.unique(np.stack([row, col], 1), axis=0, return_inverse=True)
        self.window = (cells[:, 0], cells[:, 1])
        pick = sp.csr_matrix(
            (np.ones(len(used)), (used, cell_index.ravel())), shape=(weights.shape[1], len(cells))
        )
        self.matrix = (weights @ pick).tocsr()  # reach x GEFS point

    def reach_mean(self, values):
        return self.matrix @ np.nan_to_num(values).T  # [reach, step]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--preparation", type=Path, help="For the reach order (graph.npz)")
    p.add_argument("--source", choices=list(SOURCES), required=True)
    p.add_argument("--start", help="Default: 2000-01-01 (reforecast) or 2020-09-23 (operational)")
    p.add_argument("--stop", help="Default: 2020-01-01 (reforecast) or 2026-01-01 (operational)")
    p.add_argument("--member", default="c00")
    p.add_argument("--rain-hours", type=int, default=96)
    p.add_argument(
        "--out", type=Path, help="Default: <root>/processed/rain_forecasts/gefs_<source>"
    )
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--days", type=int, default=0, help="Only the first N days (testing)")
    a = p.parse_args()
    root = a.root.resolve()
    prep = (a.preparation or root / "processed/pretraining_full").resolve()
    out = (a.out or root / f"processed/rain_forecasts/gefs_{a.source}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    with np.load(prep / "graph.npz") as graph:
        ids = graph["feature_id"].copy()
    start = pd.Timestamp(a.start or FIRST[a.source])
    stop = pd.Timestamp(a.stop or ("2020-01-01" if a.source == "reforecast" else "2026-01-01"))
    days = pd.date_range(start, stop, freq="D", inclusive="left")
    if a.days:
        days = days[: a.days]
    # The grid of the first available message defines the mapping.
    grid = None
    for day in days[:30]:
        message = messages(a.source, day, a.member, 6)[0]
        if message is not None:
            content = get(message[0], headers={"Range": f"bytes={message[1]}-{message[2]}"})
            grid = decode(content)[1]
            break
    if grid is None:
        raise ValueError("No GEFS message found in the first 30 days")
    mapping = Mapping(root, ids, grid)
    print(
        f"GEFS {a.source} {a.member}: {len(days)} issue days, {len(mapping.window[0])} grid points, rain to {a.rain_hours} h"
    )
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(a.workers, mp_context=context) as pool:
        for period, group in pd.Series(days).groupby(days.to_period("M")):
            target = out / f"{period}.npz"
            if target.exists():
                continue
            began = time.perf_counter()
            group = list(group)
            futures = [
                pool.submit(day_task, a.source, d, a.member, a.rain_hours, mapping.window)
                for d in group
            ]
            fields = [f.result(timeout=1800) for f in futures]  # [step, grid point]
            rain = np.stack([mapping.reach_mean(f).T for f in fields])  # [day, step, reach]
            bad = np.stack([np.isnan(f).all(axis=1) for f in fields])  # [day, step]
            rain[bad] = np.nan
            missing = [str(d.date()) for d, b in zip(group, bad) if b.any()]
            partial = out / f"{period}.part.npz"
            np.savez(
                partial,
                issue=np.array([str(d.date()) for d in group]),
                rain=rain.astype(np.float16),
                steps_h=np.arange(6, a.rain_hours + 1, 6),
                reach=ids,
                source=np.array(f"GEFSv12 {a.source} {a.member} 00Z, six-hour APCP"),
            )
            partial.rename(target)
            (out / f"{period}.missing.json").write_text(json.dumps(missing))
            print(
                f"{period}: {len(group)} days, {len(missing)} with missing steps, "
                f"{time.perf_counter() - began:.0f} s",
                flush=True,
            )
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
