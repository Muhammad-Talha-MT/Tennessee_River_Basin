#!/usr/bin/env python3
"""Stream raw Tennessee AORC/NWM into monthly six-hour full-network files.
Resume only unchanged completed months. No interpolation, scaling or training.
"""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import load_npz
from tqdm import tqdm

from tva_flood.common import atomic_json
from tva_flood.data.schema import WEATHER

VARIABLES = WEATHER + ["streamflow"]


def source_path(root, name, month):
    end = month + pd.offsets.MonthBegin(1)
    stamp = f"{month:%Y%m%dT%H}_{end:%Y%m%dT%H}"
    return (
        root / "raw/nwm" / f"tennessee_nwm_{stamp}.nc"
        if name == "streamflow"
        else root / "raw/aorc" / str(month.year) / f"tennessee_aorc_{name}_{stamp}.nc"
    )


def no_flow(ids):
    """Flow source with no times: every interval gets zero valid hours."""
    empty = np.empty((0, len(ids)), "float32")
    return xr.Dataset(
        {"streamflow": (("time", "feature_id"), empty, {"units": "m3 s-1"})},
        coords={"time": pd.DatetimeIndex([]), "feature_id": ids},
    )


def signature(path):
    if not path.is_file():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def worker_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--start", default="1979-02-01")
    p.add_argument("--stop", default="2023-02-01", help="Exclusive month boundary")
    p.add_argument("--block-hours", type=int, default=168)
    p.add_argument("--compression", type=int, default=1)
    p.add_argument(
        "--no-streamflow",
        action="store_true",
        help="Months without NWM flow: flow is written as all-invalid (valid hours 0)",
    )
    args = p.parse_args([v for v in sys.argv[1:] if v != "--worker"])
    start, stop = pd.Timestamp(args.start), pd.Timestamp(args.stop)
    if (
        start.day != 1
        or stop.day != 1
        or start != start.normalize()
        or stop != stop.normalize()
        or start >= stop
    ):
        p.error("Start/stop must be ordered midnight month boundaries")
    if args.compression not in range(0, 5):
        p.error("Compression must be 0..4")
    if args.block_hours < 6 or args.block_hours % 6:
        p.error("Block hours must be a positive multiple of six")
    mapping = args.root / "processed/aorc_mapping"
    weights_path = mapping / "aorc_to_nodes_weights.npz"
    order_path = mapping / "grid_and_node_order.npz"
    weights = load_npz(weights_path).tocsr()
    with np.load(order_path) as info:
        ids = info["node_comid"].copy().astype("int64")
        lat, lon = info["latitude"].copy(), info["longitude"].copy()
        available = info["forcing_available"].copy().astype(bool)
    nodes_path = args.root / "processed/graph_candidate/nodes.parquet"
    nodes = pd.read_parquet(nodes_path).sort_values("node_index")
    if not np.array_equal(nodes.comid, ids) or not np.array_equal(
        nodes.node_index, np.arange(len(ids))
    ):
        raise ValueError("Graph/mapping node ordering differs")
    sums = np.asarray(weights.sum(axis=1)).ravel()
    if (
        weights.shape != (len(ids), len(lat) * len(lon))
        or not np.isfinite(weights.data).all()
        or (weights.data < 0).any()
        or not np.allclose(sums[available], 1, atol=1e-6)
        or (sums[~available] != 0).any()
    ):
        raise ValueError("Invalid weights")
    support = weights.copy()
    support.data[:] = 1
    node_hash = hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()
    out = args.root / "processed/six_hour_nodes"
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"Processing {len(ids):,} nodes; sequential full-network reads; block hours={args.block_hours}",
        flush=True,
    )
    for month in pd.date_range(start, stop, freq="MS", inclusive="left"):
        end = month + pd.offsets.MonthBegin(1)
        stamp = f"{month:%Y%m%dT%H}_{end:%Y%m%dT%H}"
        destination = out / str(month.year) / f"tennessee_{stamp}_six_hour_nodes.nc"
        destination.parent.mkdir(parents=True, exist_ok=True)
        marker = destination.with_suffix(".complete.json")
        paths = {name: source_path(args.root, name, month) for name in VARIABLES}
        boundary_path = source_path(args.root, "APCP_surface", end)
        fingerprint = {
            "version": 1,
            "node_order_sha256": node_hash,
            "sources": [signature(path) for path in paths.values()] + [signature(boundary_path)],
            "mapping": [signature(weights_path), signature(order_path), signature(nodes_path)],
        }
        if args.no_streamflow:
            if paths["streamflow"].exists():
                raise ValueError(f"NWM flow exists for {month:%Y-%m}; do not use --no-streamflow")
            fingerprint["no_streamflow"] = True
        if destination.is_file() and marker.is_file():
            done = json.loads(marker.read_text())
            if (
                done.get("fingerprint") == fingerprint
                and done.get("file_size_bytes") == destination.stat().st_size
            ):
                print(f"Skip completed: {month:%Y-%m}", flush=True)
                continue
        times = pd.date_range(month, end, freq="6h", inclusive="left")
        partial = destination.with_suffix(".nc.part")
        checkpoint = destination.with_suffix(".progress.json")
        state = {
            "fingerprint": fingerprint,
            "block_hours": args.block_hours,
            "compression": args.compression,
            "variable_index": 0,
            "next_interval": 0,
            "audit": [],
            "stats": None,
        }
        resume = False
        if partial.is_file() and checkpoint.is_file():
            try:
                saved = json.loads(checkpoint.read_text())
                if all(
                    saved.get(k) == state[k] for k in ("fingerprint", "block_hours", "compression")
                ):
                    with xr.open_dataset(partial, engine="netcdf4") as probe:
                        if not pd.DatetimeIndex(probe.time.values).equals(
                            times
                        ) or not np.array_equal(probe.feature_id, ids):
                            raise ValueError("Checkpoint coordinates differ")
                    state = saved
                    resume = True
                    print(
                        f"Resume {month:%Y-%m}: variable {state['variable_index'] + 1}/9; next interval {state['next_interval']}",
                        flush=True,
                    )
            except (OSError, ValueError, KeyError):
                print(f"Rebuilding unreadable checkpoint: {month:%Y-%m}", flush=True)
        audit = state["audit"]
        with ExitStack() as stack:
            datasets = {
                name: no_flow(ids)
                if name == "streamflow" and args.no_streamflow
                else stack.enter_context(xr.open_dataset(path, engine="netcdf4"))
                for name, path in paths.items()
            }
            boundary = (
                stack.enter_context(xr.open_dataset(boundary_path, engine="netcdf4"))
                if boundary_path.is_file()
                else None
            )
            expected_hourly = pd.date_range(month, end, freq="h", inclusive="left")
            for name, ds in datasets.items():
                if name == "streamflow" and args.no_streamflow:
                    continue
                actual = pd.DatetimeIndex(ds.time.values)
                known_first = (
                    name == "streamflow"
                    and month == pd.Timestamp("1979-02-01")
                    and actual.equals(expected_hourly[1:])
                )
                if not actual.equals(expected_hourly) and not known_first:
                    raise ValueError(f"Unexpected time coverage: {paths[name]}")
                if name not in ds or not ds[name].attrs.get("units"):
                    raise ValueError(f"Missing variable/units: {paths[name]}")
                if name == "streamflow":
                    if not np.array_equal(ds.feature_id.values, ids) or set(ds[name].dims) != {
                        "time",
                        "feature_id",
                    }:
                        raise ValueError(
                            "NWM ordering/dimensions differ: contiguous reads require identical ordering"
                        )
                elif (
                    not np.array_equal(ds.latitude, lat)
                    or not np.array_equal(ds.longitude, lon)
                    or set(ds[name].dims) != {"time", "latitude", "longitude"}
                ):
                    raise ValueError("AORC grid/dimensions differ")
            rain_units = datasets["APCP_surface"].APCP_surface.attrs["units"]
            if rain_units not in {"kg/m^2", "kg m-2", "mm"}:
                raise ValueError(f"Unexpected rain units: {rain_units}")
            if boundary is not None:
                if (
                    not np.array_equal(boundary.latitude, lat)
                    or not np.array_equal(boundary.longitude, lon)
                    or boundary.APCP_surface.attrs.get("units") != rain_units
                ):
                    raise ValueError("Boundary rainfall grid/units differ")
                if pd.Timestamp(end) not in pd.DatetimeIndex(boundary.time.values):
                    raise ValueError("Boundary rainfall midnight absent")
            nc = stack.enter_context(netCDF4.Dataset(partial, "a" if resume else "w"))
            if not resume:
                nc.createDimension("time", len(times))
                nc.createDimension("feature_id", len(ids))
                nc.createDimension("bounds", 2)
                tv = nc.createVariable("time", "f8", ("time",))
                tv.units = "hours since 1970-01-01 00:00:00"
                tv.calendar = "standard"
                tv.bounds = "time_bounds"
                hours = netCDF4.date2num(times.to_pydatetime(), tv.units, calendar=tv.calendar)
                tv[:] = hours
                bv = nc.createVariable("time_bounds", "f8", ("time", "bounds"))
                bv.units = tv.units
                bv.calendar = tv.calendar
                bv[:] = np.column_stack([hours, hours + 6])
                nc.createVariable("feature_id", "i8", ("feature_id",))[:] = ids
                nc.createVariable("forcing_available", "u1", ("feature_id",))[:] = available
                nc.node_order_sha256 = node_hash
                nc.interval_policy = "Left-labelled [time,time+6h); rain labels time+1h..time+6h summed; other hourly samples time..time+5h averaged"
                nc.target_source = (
                    "NWM simulations for pretraining; not observed verification truth"
                )
                nc.missing_policy = "All six finite samples required; no interpolation; missing boundary rainfall masked"
                nc.operational_policy = "Retrospective AORC is not automatically available by forecast issue time; no future forcing input assignment"
            for variable_index, (name, ds) in enumerate(datasets.items()):
                if variable_index < state["variable_index"]:
                    continue
                v = nc.variables.get(name)
                if v is None:
                    v = nc.createVariable(
                        name,
                        "f4",
                        ("time", "feature_id"),
                        fill_value=np.float32(np.nan),
                        zlib=args.compression > 0,
                        complevel=args.compression,
                        chunksizes=(min(args.block_hours // 6, len(times)), min(2048, len(ids))),
                    )
                countv = nc.variables.get(name + "_valid_hours")
                if countv is None:
                    countv = nc.createVariable(
                        name + "_valid_hours",
                        "u1",
                        ("time", "feature_id"),
                        zlib=args.compression > 0,
                        complevel=args.compression,
                        chunksizes=(min(args.block_hours // 6, len(times)), min(2048, len(ids))),
                    )
                countv.units = "1"
                for key in ("units", "long_name", "standard_name"):
                    if key in ds[name].attrs:
                        v.setncattr(key, ds[name].attrs[key])
                if name == "APCP_surface":
                    v.units = "mm"
                field = (
                    ds[name].transpose("time", "feature_id")
                    if name == "streamflow"
                    else ds[name].transpose("time", "latitude", "longitude")
                )
                index = pd.DatetimeIndex(ds.time.values)
                stats = {
                    "month": f"{month:%Y-%m}",
                    "variable": name,
                    "valid_intervals": 0,
                    "masked_intervals": 0,
                    "partial_intervals": 0,
                }
                if variable_index == state["variable_index"] and state.get("stats") is not None:
                    stats = state["stats"]
                first_interval = (
                    state["next_interval"] if variable_index == state["variable_index"] else 0
                )
                for offset in tqdm(
                    range(first_interval, len(times), args.block_hours // 6),
                    desc=f"{month:%Y-%m} {name}",
                ):
                    limit = min(offset + args.block_hours // 6, len(times))
                    begin = times[offset] + (
                        pd.Timedelta(hours=1) if name == "APCP_surface" else pd.Timedelta(0)
                    )
                    wanted = pd.date_range(begin, periods=(limit - offset) * 6, freq="h")
                    positions = index.get_indexer(wanted)
                    present = positions >= 0
                    hourly = np.full((len(wanted), len(ids)), np.nan, dtype="float32")
                    if present.any():
                        selected = positions[present]
                        if not np.all(np.diff(selected) == 1):
                            raise ValueError("Noncontiguous source read")
                        values = np.asarray(
                            field.isel(time=slice(int(selected[0]), int(selected[-1]) + 1)).values,
                            dtype="float32",
                        )
                        invalid = ~np.isfinite(values)
                        if name in {"APCP_surface", "streamflow"}:
                            invalid |= values < 0
                        if name == "streamflow":
                            values[invalid] = np.nan
                            hourly[present] = values
                        else:
                            values = values.reshape(len(selected), -1)
                            invalid = invalid.reshape(len(selected), -1)
                            mapped = np.asarray(weights @ np.where(invalid, 0, values).T).T
                            if invalid.any():
                                mapped[np.asarray(support @ invalid.astype("float32").T).T > 0] = (
                                    np.nan
                                )
                            mapped[:, ~available] = np.nan
                            hourly[present] = mapped
                    if name == "APCP_surface" and end in wanted and boundary is not None:
                        values = np.asarray(
                            boundary.APCP_surface.sel(time=end)
                            .transpose("latitude", "longitude")
                            .values,
                            dtype="float32",
                        ).reshape(1, -1)
                        invalid = ~np.isfinite(values) | (values < 0)
                        mapped = np.asarray(weights @ np.where(invalid, 0, values).T).T
                        mapped[np.asarray(support @ invalid.astype("float32").T).T > 0] = np.nan
                        mapped[:, ~available] = np.nan
                        hourly[wanted == end] = mapped
                    blocks = hourly.reshape(limit - offset, 6, len(ids))
                    finite = np.isfinite(blocks)
                    counts = finite.sum(axis=1).astype("uint8")
                    result = np.where(finite, blocks, 0).sum(axis=1)
                    if name != "APCP_surface":
                        result /= 6
                    result[counts != 6] = np.nan
                    v[offset:limit] = result
                    countv[offset:limit] = counts
                    stats["valid_intervals"] += int((counts == 6).sum())
                    stats["masked_intervals"] += int((counts < 6).sum())
                    stats["partial_intervals"] += int(((counts > 0) & (counts < 6)).sum())
                    nc.sync()
                    state.update(
                        variable_index=variable_index, next_interval=limit, stats=stats, audit=audit
                    )
                    atomic_json(checkpoint, state)
                audit.append(stats)
                state.update(
                    variable_index=variable_index + 1, next_interval=0, stats=None, audit=audit
                )
                nc.sync()
                atomic_json(checkpoint, state)
        with xr.open_dataset(partial, engine="netcdf4") as check:
            if not pd.DatetimeIndex(check.time.values).equals(times) or not np.array_equal(
                check.feature_id, ids
            ):
                raise ValueError("Serialized coordinate mismatch")
        partial.replace(destination)
        done = {
            "fingerprint": fingerprint,
            "file_size_bytes": destination.stat().st_size,
            "six_hour_intervals": len(times),
            "nodes": len(ids),
            "boundary_rainfall_file_present": boundary is not None,
            "audit": audit,
        }
        atomic_json(marker, done)
        checkpoint.unlink(missing_ok=True)
        pd.DataFrame(audit).to_csv(destination.with_suffix(".audit.csv"), index=False)
        print(f"Saved: {destination}", flush=True)
    print(
        "Requested months completed. Build dated training windows separately; no training run.",
        flush=True,
    )


def main():
    if "--worker" in sys.argv:
        worker_main()
        return
    p = argparse.ArgumentParser(
        description="Parallel full-network six-hour preprocessing with per-block resume. Stop older preprocessing before starting."
    )
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--start", default="1979-02-01")
    p.add_argument("--stop", default="2023-02-01", help="Exclusive month boundary")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--no-streamflow",
        action="store_true",
        help="Months without NWM flow (after January 2023): flow is written all-invalid",
    )
    p.add_argument(
        "--block-hours",
        type=int,
        default=168,
        help="Multiple of six; keep unchanged when resuming partial months",
    )
    p.add_argument(
        "--compression",
        type=int,
        default=1,
        help="0..4; keep unchanged when resuming partial months",
    )
    args = p.parse_args()
    start, stop = pd.Timestamp(args.start), pd.Timestamp(args.stop)
    if (
        args.workers < 1
        or args.block_hours < 6
        or args.block_hours % 6
        or args.compression not in range(5)
    ):
        p.error("Invalid workers, block hours or compression")
    if start >= stop or any(t.day != 1 or t != t.normalize() for t in (start, stop)):
        p.error("Start/stop must be ordered midnight month boundaries")
    logs = args.root / "logs/six_hour_parallel"
    logs.mkdir(parents=True, exist_ok=True)
    lock = (logs / "run.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("Another parallel preprocessing run holds this root lock")
    children = set()
    import threading

    guard = threading.Lock()
    stopped = threading.Event()

    def halt(signum, frame):
        stopped.set()
        with guard:
            for child in list(children):
                if child.poll() is None:
                    child.terminate()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    environment = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[key] = "1"

    def run(month):
        if stopped.is_set():
            return str(month), "cancelled", 0
        end = month + pd.offsets.MonthBegin(1)
        command = [
            sys.executable,
            "-m",
            "tva_flood.data.aggregate",
            "--worker",
            "--root",
            str(args.root),
            "--start",
            str(month.date()),
            "--stop",
            str(end.date()),
            "--block-hours",
            str(args.block_hours),
            "--compression",
            str(args.compression),
            *(["--no-streamflow"] if args.no_streamflow else []),
        ]
        log = logs / f"{month:%Y-%m}.log"
        begun = time.monotonic()
        with log.open("a") as handle:
            handle.write("\n--- Worker start ---\n")
            handle.flush()
            with guard:
                if stopped.is_set():
                    return str(month), "cancelled", 0
                child = subprocess.Popen(
                    command, stdout=handle, stderr=subprocess.STDOUT, env=environment
                )
                children.add(child)
            try:
                code = child.wait()
            finally:
                with guard:
                    children.discard(child)
        return f"{month:%Y-%m}", "OK" if code == 0 else f"FAIL({code})", time.monotonic() - begun

    months = list(pd.date_range(start, stop, freq="MS", inclusive="left"))
    print(f"{len(months)} months; {args.workers} workers; logs: {logs}", flush=True)
    print("Resume is per saved time block; unchanged completed months are skipped.", flush=True)
    failures = []
    executor = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = [executor.submit(run, month) for month in months]
        for number, future in enumerate(as_completed(futures), 1):
            month, status, elapsed = future.result()
            print(f"{number}/{len(months)} {month}: {status}; {elapsed / 60:.1f} min", flush=True)
            if status != "OK":
                failures.append(month)
    except KeyboardInterrupt:
        stopped.set()
        with guard:
            for child in list(children):
                if child.poll() is None:
                    child.terminate()
        for future in futures:
            future.cancel()
        print("Stopped. Rerun the same command to resume saved blocks.", flush=True)
        raise SystemExit(130)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        lock.close()
    if failures:
        print(f"Failed months: {failures}; inspect worker logs and rerun.", flush=True)
        raise SystemExit(1)
    print(
        "All requested months completed. Training/validation/test splits remain a separate step.",
        flush=True,
    )


if __name__ == "__main__":
    main()
