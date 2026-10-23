#!/usr/bin/env python3
"""Resume monthly Tennessee NWM v3.0 discharge and native-grid AORC v1.1 downloads.

Time interval is [--start, --stop); stop is exclusive. Packed source values and
CF scaling are preserved. No six-hour aggregation, gap filling, or normalization.
"""

import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import multiprocessing
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pandas as pd
import s3fs
import zarr

from tva_flood.common import atomic_json
from tva_flood.data.schema import WEATHER

NWM = "noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
AORC = "noaa-nws-aorc-v1-1-1km/{year}.zarr"
VARIABLES = WEATHER


def retry(fn):
    for attempt in range(5):
        try:
            return fn()
        except Exception as exc:
            if attempt == 4:
                raise
            print(f"Retry {attempt + 1}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(min(2**attempt, 16))


def open_group(fs, path):
    mapper = s3fs.S3Map(root=path, s3=fs, check=False)
    group = retry(lambda: zarr.open_consolidated(mapper, mode="r"))
    fingerprint = hashlib.sha256(retry(lambda: mapper[".zmetadata"])).hexdigest()
    return group, fingerprint


def times(group):
    array = group["time"]
    raw = retry(lambda: array[:])
    dates = netCDF4.num2date(
        raw,
        array.attrs["units"],
        calendar=array.attrs.get("calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    result = pd.DatetimeIndex(dates).as_unit("ns")
    if not result.is_unique or not result.is_monotonic_increasing:
        raise RuntimeError("Source time axis is duplicated or unordered")
    if len(result) > 1 and (np.diff(result.asi8) != 3600 * 10**9).any():
        raise RuntimeError("Source time axis contains missing hours; inspect before continuing")
    return result


def intervals(start, stop):
    current = start
    while current < stop:
        following = min(current.to_period("M").end_time.normalize() + pd.Timedelta(days=1), stop)
        yield current, following
        current = following


def bounds(root):
    collection = json.loads((root / "raw/geography/tennessee_huc2.geojson").read_text())
    coordinates = []

    def walk(value):
        if isinstance(value[0], (float, int)):
            coordinates.append(value[:2])
        else:
            for child in value:
                walk(child)

    for feature in collection["features"]:
        walk(feature["geometry"]["coordinates"])
    points = np.asarray(coordinates)
    return [
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    ]


def spatial_slice(coordinate, lower, upper):
    spacing = float(np.max(np.abs(np.diff(coordinate))))
    ids = np.flatnonzero((coordinate >= lower - spacing) & (coordinate <= upper + spacing))
    if not len(ids) or not np.all(np.diff(ids) == 1):
        raise RuntimeError("Invalid spatial selection")
    return slice(int(ids[0]), int(ids[-1]) + 1)


def blocks(begin, end, length):
    current = begin
    while current < end:
        following = min((current // length + 1) * length, end)
        yield current, following
        current = following


def packed_variable(handle, name, array, dims, shape):
    attrs = dict(array.attrs)
    fill = attrs.pop("_FillValue", array.fill_value)
    attrs.pop("_ARRAY_DIMENSIONS", None)
    # Source grid_mapping may refer to a separate CRS variable. Supply our own below.
    attrs.pop("grid_mapping", None)
    attrs.pop("coordinates", None)
    chunks = tuple(min(s, c) for s, c in zip(shape, array.chunks))
    variable = handle.createVariable(
        name,
        array.dtype,
        dims,
        fill_value=fill,
        zlib=True,
        complevel=1,
        shuffle=True,
        chunksizes=chunks,
    )
    for key, value in attrs.items():
        if isinstance(value, (str, int, float, list)):
            variable.setncattr(key, value)
    variable.set_auto_maskandscale(False)
    return variable


def save_file(path, signature, minimum_free_gib, write):
    marker = path.with_suffix(".complete.json")
    if path.exists() and marker.exists():
        info = json.loads(marker.read_text())
        if info["selection"] == signature and path.stat().st_size == info["file_size_bytes"]:
            print(f"Skip completed: {path.name}", flush=True)
            return
        raise RuntimeError(f"Existing output differs from requested selection: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < minimum_free_gib * 1024**3:
        raise RuntimeError(f"Less than {minimum_free_gib} GiB free; stopped before writing {path}")
    temporary = path.with_suffix(".nc.part")
    # Incomplete monthly files are rebuilt; completed files are never downloaded again.
    write(temporary)
    with netCDF4.Dataset(temporary) as check:
        check.set_auto_maskandscale(False)
        for dimension in check.dimensions.values():
            if len(dimension) == 0:
                raise RuntimeError("Empty dimension in downloaded file")
    os.replace(temporary, path)
    atomic_json(
        marker,
        {
            "selection": signature,
            "file_size_bytes": path.stat().st_size,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Saved: {path} ({path.stat().st_size / 1024**2:.1f} MiB)", flush=True)


def time_coordinate(handle, selected):
    handle.createDimension("time", len(selected))
    variable = handle.createVariable("time", "i8", ("time",))
    variable.units = "hours since 1970-01-01 00:00:00"
    variable.calendar = "proleptic_gregorian"
    variable.standard_name = "time"
    variable[:] = selected.asi8 // (3600 * 10**9)


def download_nwm(args, fs, output):
    inventory = pd.read_parquet(args.root / "metadata/tennessee_nwm_reaches.parquet")
    inventory = inventory.sort_values("nwm_feature_index")
    if args.smoke_test:
        inventory = inventory.head(64)
    positions = inventory["nwm_feature_index"].to_numpy(dtype="int64")
    feature_ids = inventory["comid"].to_numpy(dtype="int64")
    group, fingerprint = open_group(fs, NWM)
    if not np.array_equal(retry(lambda: group["feature_id"].oindex[positions]), feature_ids):
        raise RuntimeError("Inventory node order no longer matches NWM")
    dates = times(group)
    source = group["streamflow"]
    if list(source.attrs["_ARRAY_DIMENSIONS"]) != ["time", "feature_id"]:
        raise RuntimeError("Unexpected discharge dimension order")
    for start, stop in intervals(args.start, args.stop):
        begin, end = int(dates.searchsorted(start)), int(dates.searchsorted(stop))
        if begin == end:
            print(f"No NWM data for {start} to {stop}", flush=True)
            continue
        selected = dates[begin:end]
        name = f"tennessee_nwm_{start:%Y%m%dT%H}_{stop:%Y%m%dT%H}.nc"
        signature = {
            "source": NWM,
            "source_metadata_sha256": fingerprint,
            "start": str(start),
            "stop_exclusive": str(stop),
            "feature_ids_sha256": hashlib.sha256(feature_ids.tobytes()).hexdigest(),
        }

        def write(path):
            with netCDF4.Dataset(path, "w") as handle:
                handle.source = "s3://" + NWM
                handle.description = (
                    "NWM v3.0 simulated discharge; source packed integers and CF scaling preserved"
                )
                time_coordinate(handle, selected)
                handle.createDimension("feature_id", len(feature_ids))
                handle.createVariable("feature_id", "i8", ("feature_id",))[:] = feature_ids
                variable = packed_variable(
                    handle,
                    "streamflow",
                    source,
                    ("time", "feature_id"),
                    (len(selected), len(feature_ids)),
                )
                for first, last in blocks(begin, end, source.chunks[0]):
                    raw = retry(lambda: source.oindex[slice(first, last), positions])
                    variable[first - begin : last - begin, :] = raw
                    print(f"  NWM {start:%Y-%m}: {last - begin}/{end - begin} hours", flush=True)

        save_file(output / "nwm" / name, signature, args.min_free_gib, write)


def download_aorc(args, fs, output):
    domain = bounds(args.root)
    for year in range(args.start.year, (args.stop - pd.Timedelta(nanoseconds=1)).year + 1):
        store = AORC.format(year=year)
        group, fingerprint = open_group(fs, store)
        dates = times(group)
        latitude, longitude = (
            retry(lambda: group["latitude"][:]),
            retry(lambda: group["longitude"][:]),
        )
        lat_slice = spatial_slice(latitude, domain[1], domain[3])
        lon_slice = spatial_slice(longitude, domain[0], domain[2])
        if args.smoke_test:
            lat_slice = slice(lat_slice.start, lat_slice.start + 3)
            lon_slice = slice(lon_slice.start, lon_slice.start + 3)
        lats, lons = latitude[lat_slice], longitude[lon_slice]
        start = max(args.start, pd.Timestamp(year=year, month=1, day=1))
        stop = min(args.stop, pd.Timestamp(year=year + 1, month=1, day=1))
        print(f"AORC {year}: regional grid {len(lats)} x {len(lons)}", flush=True)
        for first_date, last_date in intervals(start, stop):
            begin, end = int(dates.searchsorted(first_date)), int(dates.searchsorted(last_date))
            if begin == end:
                print(f"No AORC data for {first_date} to {last_date}", flush=True)
                continue
            selected = dates[begin:end]
            for name in VARIABLES:
                source = group[name]
                if list(source.attrs["_ARRAY_DIMENSIONS"]) != ["time", "latitude", "longitude"]:
                    raise RuntimeError(f"Unexpected dimensions for {name}")
                filename = f"tennessee_aorc_{name}_{first_date:%Y%m%dT%H}_{last_date:%Y%m%dT%H}.nc"
                signature = {
                    "source": store,
                    "source_metadata_sha256": fingerprint,
                    "variable": name,
                    "start": str(first_date),
                    "stop_exclusive": str(last_date),
                    "latitude_indices": [lat_slice.start, lat_slice.stop],
                    "longitude_indices": [lon_slice.start, lon_slice.stop],
                }

                def write(path):
                    with netCDF4.Dataset(path, "w") as handle:
                        handle.source = "s3://" + store
                        handle.description = "AORC v1.1 bounding rectangle with one grid-cell margin; native packed values retained"
                        handle.geographic_crs = "EPSG:4326"
                        time_coordinate(handle, selected)
                        handle.createDimension("latitude", len(lats))
                        handle.createDimension("longitude", len(lons))
                        lat = handle.createVariable("latitude", "f8", ("latitude",))
                        lon = handle.createVariable("longitude", "f8", ("longitude",))
                        lat.units, lon.units = "degrees_north", "degrees_east"
                        lat.standard_name, lon.standard_name = "latitude", "longitude"
                        lat[:], lon[:] = lats, lons
                        variable = packed_variable(
                            handle,
                            name,
                            source,
                            ("time", "latitude", "longitude"),
                            (len(selected), len(lats), len(lons)),
                        )
                        for first, last in blocks(begin, end, source.chunks[0]):
                            raw = retry(lambda: source[first:last, lat_slice, lon_slice])
                            variable[first - begin : last - begin, :, :] = raw
                            print(
                                f"  AORC {first_date:%Y-%m} {name}: {last - begin}/{end - begin} hours",
                                flush=True,
                            )

                save_file(
                    output / "aorc" / str(year) / filename, signature, args.min_free_gib, write
                )


# Separate processes keep NetCDF/HDF5 writers isolated; years never overlap.


def year_worker(task):
    root, start, stop, smoke, minimum, log_path = task
    args = SimpleNamespace(
        root=Path(root),
        start=pd.Timestamp(start),
        stop=pd.Timestamp(stop),
        smoke_test=smoke,
        min_free_gib=minimum,
    )
    output = args.root / "raw" / ("_smoke_test" if smoke else "")
    with (
        open(log_path, "a", buffering=1) as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        print(f"Worker started {start} .. {stop}", flush=True)
        fs = s3fs.S3FileSystem(
            anon=True, config_kwargs={"max_pool_connections": 32, "retries": {"max_attempts": 5}}
        )
        download_aorc(args, fs, output)
        print("Year task completed.", flush=True)
    return start


def local_year_complete(root, start, stop, smoke):
    # Quick local check only; source fingerprint is still verified by each worker.
    output = root / "raw" / ("_smoke_test" if smoke else "") / "aorc" / str(start.year)
    for first, last in intervals(start, stop):
        for name in VARIABLES:
            p = output / f"tennessee_aorc_{name}_{first:%Y%m%dT%H}_{last:%Y%m%dT%H}.nc"
            marker = p.with_suffix(".complete.json")
            if not p.exists() or not marker.exists():
                return False
            try:
                info = json.loads(marker.read_text())
                if p.stat().st_size != info["file_size_bytes"]:
                    return False
                selection = info["selection"]
                if selection["start"] != str(first) or selection["stop_exclusive"] != str(last):
                    return False
            except (OSError, KeyError, ValueError):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download NWM and parallel AORC; reuse original monthly resume markers."
    )
    parser.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    parser.add_argument("--start", default="1979-02-01")
    parser.add_argument("--stop", default="2023-02-01")
    parser.add_argument("--source", choices=["all", "aorc", "nwm"], default="all")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--min-free-gib", type=float, default=100)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--verify-completed-years",
        action="store_true",
        help="Also open remote metadata for locally complete years; no data redownload when signatures match.",
    )
    args = parser.parse_args()
    args.start, args.stop = pd.Timestamp(args.start), pd.Timestamp(args.stop)
    if args.start >= args.stop or args.start.tz is not None or args.stop.tz is not None:
        parser.error("Use increasing UTC timestamps without timezone suffixes")
    if args.start != args.start.floor("h") or args.stop != args.stop.floor("h"):
        parser.error("Start and stop must be whole hours")
    if args.workers < 1 or args.workers > 16:
        parser.error("Use 1–16 workers")
    if args.smoke_test and args.stop - args.start > pd.Timedelta(days=1):
        parser.error("Smoke test must be at most one day")
    # Refuse to share output paths with the original serial downloader.
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if any(Path(arg).name == "04_download_timeseries.py" for arg in argv[1:] if arg):
                raise SystemExit(
                    f"Stop original downloader PID {proc.name} before starting this script."
                )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    args.root.mkdir(parents=True, exist_ok=True)
    # This lock protects parallel-script instances; stop the original script first.
    with open(args.root / ".aorc_parallel.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another parallel AORC downloader is already running.")
        if args.source in ("all", "nwm"):
            fs = s3fs.S3FileSystem(anon=True, config_kwargs={"max_pool_connections": 32})
            output = args.root / "raw" / ("_smoke_test" if args.smoke_test else "")
            download_nwm(args, fs, output)
            if args.source == "nwm":
                return
        logs = args.root / "logs" / "aorc_parallel"
        logs.mkdir(parents=True, exist_ok=True)
        tasks = []
        for year in range(args.start.year, (args.stop - pd.Timedelta(nanoseconds=1)).year + 1):
            start = max(args.start, pd.Timestamp(year=year, month=1, day=1))
            stop = min(args.stop, pd.Timestamp(year=year + 1, month=1, day=1))
            if not args.verify_completed_years and local_year_complete(
                args.root, start, stop, args.smoke_test
            ):
                print(f"Skip locally complete year: {year}", flush=True)
                continue
            tasks.append(
                (
                    str(args.root),
                    str(start),
                    str(stop),
                    args.smoke_test,
                    args.min_free_gib,
                    str(logs / f"{year}.log"),
                )
            )
        if not tasks:
            print("All requested AORC files are locally complete.", flush=True)
            return
        print(
            f"Pending year tasks: {len(tasks)}; workers: {min(args.workers, len(tasks))}; worker logs: {logs}",
            flush=True,
        )
        print("Completed variable-month files are reused. Partial files are rebuilt.", flush=True)
        ctx = multiprocessing.get_context("spawn")
        errors = []
        completed = 0
        t0 = time.monotonic()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers, len(tasks)), mp_context=ctx
        ) as pool:
            pending = {pool.submit(year_worker, t): t for t in tasks}
            while pending:
                done, _ = concurrent.futures.wait(
                    pending, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED
                )
                if not done:
                    print(
                        f"Active/pending year tasks: {len(pending)}; elapsed {(time.monotonic() - t0) / 3600:.2f} h",
                        flush=True,
                    )
                for future in done:
                    task = pending.pop(future)
                    try:
                        future.result()
                        completed += 1
                        print(
                            f"Completed year {pd.Timestamp(task[1]).year}: {completed}/{len(tasks)} tasks",
                            flush=True,
                        )
                    except Exception as exc:
                        errors.append(task[1])
                        print(f"FAILED {task[1]}: {exc}; inspect {task[-1]}", flush=True)
        if errors:
            raise SystemExit(f"Failed year tasks: {errors}. Rerun to resume.")
        print("AORC download completed.", flush=True)


if __name__ == "__main__":
    main()
