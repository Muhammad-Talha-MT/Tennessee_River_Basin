#!/usr/bin/env python3
"""Full Tennessee preparation: graph, manifest and resumable normalization.

Uses the shared static-attribute module. Uses existing processed monthly data.
Targets: training Feb 1979--Dec 2018, validation 2019; later years untouched.
Precipitation defaults to log1p of six-hour totals in mm, then standardization.
Use --replace-preparation to replace a previous configuration in the same --out.
This prepares inputs; it does not launch a training experiment.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tva_flood.common import atomic_json, hash_array

from . import statics as statics_module
from .schema import HISTORY, HORIZON, WEATHER
from .transforms import transform_precipitation


def read_graph(root):
    path = root / "processed/graph_candidate/nodes.parquet"
    inventory = pd.read_parquet(root / "metadata/tennessee_nwm_reaches.parquet")
    if path.exists():
        nodes = pd.read_parquet(path).sort_values("node_index")
        if not np.array_equal(nodes.node_index, np.arange(len(nodes))):
            raise ValueError("Graph node_index must be contiguous")
        ids = nodes.comid.to_numpy(dtype=np.int64)
    else:
        ids = inventory.sort_values("nwm_feature_index").comid.to_numpy(dtype=np.int64)
    if len(np.unique(ids)) != len(ids) or set(ids) != set(inventory.comid):
        raise ValueError("Full graph must include every inventory COMID exactly once")
    edge_path = root / "processed/graph_candidate/edges.parquet"
    if edge_path.exists() and path.exists():
        frame = pd.read_parquet(edge_path)
        if not {"fromcomid", "tocomid"} <= set(frame):
            raise ValueError("Candidate edges missing COMIDs")
    else:
        frame = pd.read_parquet(root / "metadata/tennessee_nhd_connectivity.parquet")
        frame = frame[frame.fromcomid.isin(ids) & frame.tocomid.isin(ids)]
    index = pd.Index(ids)
    source = index.get_indexer(frame.fromcomid)
    target = index.get_indexer(frame.tocomid)
    if (source < 0).any() or (target < 0).any():
        raise ValueError("Edge outside full graph")
    if "source" in frame and "target" in frame and edge_path.exists() and path.exists():
        if not (np.array_equal(source, frame.source) and np.array_equal(target, frame.target)):
            raise ValueError("Candidate edge indices disagree with node COMIDs")
    edges = np.column_stack([source, target]).astype(np.int64)
    unique = np.unique(edges, axis=0)
    print(
        f"Graph: {len(ids):,} nodes, {len(unique):,} unique internal edges; {len(edges) - len(unique)} duplicate records collapsed"
    )
    return ids, unique


def scan_archive(root, ids, start, stop, check_ids=True):
    records = []
    for number, first in enumerate(pd.date_range(start, stop, freq="MS", inclusive="left")):
        last = first + pd.offsets.MonthBegin(1)
        stamp = f"{first:%Y%m%dT%H}_{last:%Y%m%dT%H}"
        path = (
            root
            / "processed/six_hour_nodes"
            / str(first.year)
            / f"tennessee_{stamp}_six_hour_nodes.nc"
        )
        with xr.open_dataset(path, engine="netcdf4", cache=False) as ds:
            times = pd.DatetimeIndex(ds.time.values)
            expected = pd.date_range(first, last, freq="6h", inclusive="left")
            if not times.equals(expected):
                raise ValueError(f"Time mismatch: {path}")
            actual = pd.Index(ds.feature_id.values)
            if not actual.is_unique or (check_ids and set(actual) != set(ids)):
                raise ValueError(f"COMID set mismatch: {path}")
            for v in WEATHER + ["streamflow"]:
                for key in (v, v + "_valid_hours"):
                    if ds[key].dims != ("time", "feature_id"):
                        raise ValueError(f"Unexpected dimensions for {key}: {path}")
            records.append(
                dict(
                    path=str(path.relative_to(root)),
                    start=str(first),
                    steps=len(times),
                    bytes=path.stat().st_size,
                    mtime_ns=path.stat().st_mtime_ns,
                    ordered_ids_sha256=hash_array(actual.to_numpy(dtype="<i8")),
                )
            )
        if (number + 1) % 48 == 0:
            print(f"Validated {number + 1} monthly headers", flush=True)
    return records


def fit_normalization(
    root, records, out, identity, train_stop, block_steps, precip_transform="log1p"
):
    partial = out / "normalization_progress.json"
    state = dict(identity=identity, completed=0, count=[0] * 9, mean=[0.0] * 9, m2=[0.0] * 9)
    if partial.exists():
        state = json.loads(partial.read_text())
        if state["identity"] != identity:
            raise ValueError("Normalization sources/settings changed. Use a fresh --out.")
    training = [r for r in records if pd.Timestamp(r["start"]) < train_stop]
    for month in range(state["completed"], len(training)):
        r = training[month]
        with xr.open_dataset(root / r["path"], engine="netcdf4", cache=False) as ds:
            for v, variable in enumerate(WEATHER + ["streamflow"]):
                for left in range(0, r["steps"], block_steps):
                    selection = dict(time=slice(left, left + block_steps))
                    x = ds[variable].isel(**selection).values
                    mask = np.isfinite(x) & (
                        ds[variable + "_valid_hours"].isel(**selection).values == 6
                    )
                    x = x[mask].astype(np.float64)
                    if variable == "APCP_surface":
                        x = transform_precipitation(x, mode=precip_transform)
                    if variable == "streamflow":
                        if (x < 0).any():
                            raise ValueError(f"Negative valid streamflow in {r['path']}")
                        x = np.log1p(x)
                    if len(x) == 0:
                        continue
                    n, avg = len(x), float(x.mean())
                    m2 = float(np.square(x - avg).sum())
                    old = state["count"][v]
                    total = old + n
                    delta = avg - state["mean"][v]
                    state["mean"][v] += delta * n / total
                    state["m2"][v] += m2 + delta * delta * old * n / total
                    state["count"][v] = total
        state["completed"] = month + 1
        atomic_json(partial, state)
        print(
            f"Normalization: {month + 1}/{len(training)} training months ({r['start'][:7]})",
            flush=True,
        )
    count = np.asarray(state["count"])
    if (count == 0).any():
        raise ValueError("A training feature has no valid observations")
    std = np.sqrt(np.maximum(np.asarray(state["m2"]) / count, 0))
    std[std < 1e-6] = 1
    result = dict(
        variables=WEATHER + ["log1p_streamflow"],
        mean=state["mean"],
        precip_transform=precip_transform,
        std=std.tolist(),
        count=state["count"],
        identity=identity,
    )
    atomic_json(out / "dynamic_normalization.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--out", type=Path)
    p.add_argument("--train-start", default="1979-02-01")
    p.add_argument("--train-stop", default="2019-01-01")
    p.add_argument("--validation-stop", default="2020-01-01")
    p.add_argument("--time-block", type=int, default=128)
    p.add_argument("--precip-transform", choices=["none", "log1p"], default="log1p")
    p.add_argument(
        "--replace-preparation",
        action="store_true",
        help="Replace existing preparation metadata/statistics if configuration changed; same-config runs resume",
    )
    p.add_argument(
        "--headers-only",
        action="store_true",
        help="Build manifest without scanning values; not training-ready",
    )
    args = p.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "processed/pretraining_full").resolve()
    start, stop, val_stop = map(
        pd.Timestamp, [args.train_start, args.train_stop, args.validation_stop]
    )
    if any(
        t.tzinfo is not None or t != t.to_period("M").start_time for t in (start, stop, val_stop)
    ):
        p.error("Dates must be timezone-naive month starts")
    if (
        not pd.Timestamp("1979-02-01")
        <= start
        < stop
        <= pd.Timestamp("2019-01-01")
        < val_stop
        <= pd.Timestamp("2020-01-01")
    ):
        p.error(
            "Training must stop by Jan 2019; validation is in 2019; later tests remain reserved"
        )
    if args.time_block < 1:
        p.error("time-block must be positive")
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / ".preparation.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(
            f"Another preparation process is using {out}. Wait for it to finish or stop that process before replacing."
        ) from None
    ids, edges = read_graph(root)
    statics = statics_module
    static_path = root / "processed/static/tennessee_static_core.nc"
    with xr.open_dataset(static_path, engine="netcdf4") as ds:
        if set(ds.feature_id.values) != set(ids) or not pd.Index(ds.feature_id.values).is_unique:
            raise ValueError("Static and full graph COMID sets differ")
        names = ds.static_feature.values.tolist()
        if len(names) != 24 or not np.array_equal(
            ds.static_valid.astype(bool), np.isfinite(ds.static_values)
        ):
            raise ValueError("Unexpected static schema or mask")
    # Temporal split: all these network nodes are training nodes. For geographic
    # transfer, build a separate preparation using an explicit held-out-node policy.
    _, normalization = statics.load_static_inputs(static_path, ids, fit_ids=ids)
    records = scan_archive(root, ids, start, val_stop)
    identity_config = dict(
        train=[str(start), str(stop)],
        validation=["2019-01-01 00:00:00", str(val_stop)],
        node_hash=hash_array(ids.astype("<i8")),
        edge_hash=hash_array(edges.astype("<i8")),
        static_sha256=statics.sha(static_path),
        months=records,
        schema=2,
        precip_transform=args.precip_transform,
    )
    identity = hashlib.sha256(json.dumps(identity_config, sort_keys=True).encode()).hexdigest()
    manifest_path = out / "dataset_manifest.json"
    stale = []
    for name in (
        "dataset_manifest.json",
        "normalization_progress.json",
        "dynamic_normalization.json",
    ):
        path = out / name
        if path.exists() and json.loads(path.read_text()).get("identity") != identity:
            stale.append(path)
    if stale and not args.replace_preparation:
        raise ValueError(
            "Existing preparation differs. Use --replace-preparation to replace it in this --out, or choose a new --out."
        )
    if stale:
        # Mark preparation unusable before resetting stats; retain the lock inode.
        if manifest_path.exists():
            old_manifest = json.loads(manifest_path.read_text())
            old_manifest["training_ready"] = False
            atomic_json(manifest_path, old_manifest)
        for name in ("normalization_progress.json", "dynamic_normalization.json"):
            (out / name).unlink(missing_ok=True)
        print(
            f"Replacing preparation in {out}; recalculating normalization from month 1", flush=True
        )
    print(
        f"Precipitation: {args.precip_transform} on six-hour mm totals, then training-only standardization",
        flush=True,
    )
    total_steps = sum(r["steps"] for r in records if pd.Timestamp(r["start"]) < stop)
    validation_steps = int((val_stop - pd.Timestamp("2019-01-01")) / pd.Timedelta(hours=6))
    manifest = dict(
        identity=identity,
        nodes=len(ids),
        edges=len(edges),
        months=records,
        schema=2,
        precip_transform=args.precip_transform,
        precip_reference_mm=1.0,
        node_hash=identity_config["node_hash"],
        edge_hash=identity_config["edge_hash"],
        static_sha256=identity_config["static_sha256"],
        history=HISTORY,
        horizon=HORIZON,
        dynamic_features=WEATHER,
        static_features=names,
        input_features=16 + 2 * len(names),
        splits=dict(train=[str(start), str(stop)], validation=["2019-01-01", str(val_stop)]),
        train_windows=max(0, total_steps - HISTORY - HORIZON + 1),
        validation_windows=max(0, validation_steps - HORIZON + 1),
        inputs="Historical weather+mask; static descriptors+mask. No future observations.",
        targets="Six-hour NWM mean streamflow, log1p then shared training normalization.",
        graph_note="NHDPlus topology; NWM routing equivalence still unverified.",
        landcover="Dated snapshots excluded from invariant core inputs.",
        training_ready=False,
    )
    # npz contains only O(N+E) IDs and edge arrays.
    temp = out / "graph.part.npz"
    np.savez_compressed(temp, feature_id=ids, source=edges[:, 0], target=edges[:, 1])
    os.replace(temp, out / "graph.npz")
    atomic_json(out / "static_normalization.json", normalization)
    atomic_json(manifest_path, manifest)
    if args.headers_only:
        print("HEADERS CHECKED. Run without --headers-only to fit training normalization.")
        return
    fit_normalization(root, records, out, identity, stop, args.time_block, args.precip_transform)
    manifest["training_ready"] = True
    atomic_json(manifest_path, manifest)
    print(
        f"FULL DATA READY: nodes={len(ids):,}; input features={manifest['input_features']}; train windows={manifest['train_windows']:,}; validation windows={manifest['validation_windows']:,}"
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
