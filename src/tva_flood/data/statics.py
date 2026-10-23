#!/usr/bin/env python3
"""Download StreamCat Region06 and attach statics to the existing pretraining data.

Run in the existing tva_gnn environment. Does not change monthly dynamic files.
Dependencies: numpy, pandas, pyarrow, xarray, netCDF4 (already used by pipeline).
Core: NHDPlus geometry + StreamCat soil, geology and terrain. Dated NLCD
snapshots are downloaded separately; they are not backfilled into 1979 inputs.
"""

import argparse
import fcntl
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import xarray as xr

from tva_flood.common import sha

API = "https://api.epa.gov/StreamCat/streams/metrics"
DEFINITIONS = "https://ordspub.epa.gov/ords/ow/streamcat/streamcat/variable_info"
GROUPS = {
    "STATSGO_Set1": ["clay", "sand"],
    "STATSGO_Set2": ["om", "perm", "rckdep", "wtdep"],
    "GeoChemPhys3": ["hydrlcond"],
    "Elevation": ["elev"],
}
# Use descriptive definitions where the API's units field incorrectly says Percent.
UNITS = dict(
    clay="percent",
    sand="percent",
    om="percent_by_weight",
    perm="cm/hour",
    rckdep="cm",
    wtdep="cm",
    hydrlcond="micrometer/second",
    elev="m",
)
LC = [
    "pctow",
    "pctice",
    "pcturbop",
    "pcturblo",
    "pcturbmd",
    "pcturbhi",
    "pctbl",
    "pctdecid",
    "pctconif",
    "pctmxfst",
    "pctshrb",
    "pctgrs",
    "pcthay",
    "pctcrop",
    "pctwdwet",
    "pcthbwet",
]
# source, output, unit, multiplier, minimum; smoothed NHDPlus elevations are cm.
NHD = [
    ("lengthkm", "reach_length_km", "km", 1, 0),
    ("slope", "reach_slope", "m/m", 1, 0),
    ("maxelevsmo", "reach_upstream_elevation_m", "m", 0.01, None),
    ("minelevsmo", "reach_downstream_elevation_m", "m", 0.01, None),
    ("streamorde", "stream_order", "1", 1, 1),
    ("areasqkm", "local_catchment_area_km2", "km2", 1, 0),
    ("totdasqkm", "upstream_drainage_area_km2", "km2", 1, 0),
    ("arbolatesu", "upstream_channel_length_km", "km", 1, 0),
]


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def request(url):
    error = None
    for attempt in range(4):
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "GraphRivers-static-preparation/1.0"}),
                timeout=180,
            ) as r:
                if r.status != 200 or r.headers.get("warning"):
                    raise RuntimeError(
                        f"Incomplete API response: {r.status}, {r.headers.get('warning')}"
                    )
                return r.read()
        except Exception as exc:
            error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Download failed after 4 attempts: {url}: {error}")


def validate_ids(values, label):
    x = pd.to_numeric(pd.Series(values), errors="raise").to_numpy()
    if not np.isfinite(x).all() or (x <= 0).any() or not np.equal(x, np.floor(x)).all():
        raise ValueError(f"Invalid COMIDs in {label}")
    x = x.astype(np.int64)
    if len(np.unique(x)) != len(x):
        raise ValueError(f"Duplicate COMIDs in {label}")
    return x


def read_payload(data, metrics):
    doc = json.loads(data)
    if doc.get("hasMore") or doc.get("has_more"):
        raise RuntimeError("API paginated the response; refusing a partial regional table")
    if any(link.get("rel") == "next" for link in doc.get("links", [])):
        raise RuntimeError("API returned a next page; refusing a partial regional table")
    frame = pd.DataFrame(doc["items"])
    frame.columns = frame.columns.str.lower()
    frame["comid"] = validate_ids(frame["comid"], "StreamCat response")
    wanted = [m + scope for m in metrics for scope in ("cat", "ws")]
    missing = set(wanted) - set(frame.columns)
    if missing or frame.empty:
        raise RuntimeError(f"API omitted requested metrics: {sorted(missing)}")
    return frame.set_index("comid")


def download_group(cache, name, metrics, offline):
    params = dict(name=",".join(metrics), region="Region06", aoi="cat,ws", showpctfull="true")
    url = API + "?" + urlencode(params)
    path = cache / (name + "_Region06.json")
    meta_path = path.with_suffix(".source.json")
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("url") != url or meta.get("sha256") != sha(path):
            raise RuntimeError(f"Cache mismatch: {path}; move it aside and rerun")
        data = path.read_bytes()
    else:
        if offline:
            raise FileNotFoundError(f"Offline cache missing: {path}")
        print(f"Downloading {name}: {', '.join(metrics)}", flush=True)
        data = request(url)
        frame = read_payload(data, metrics)
        atomic_bytes(path, data)
        meta = dict(
            url=url,
            sha256=sha(path),
            rows=len(frame),
            retrieved_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(meta_path, meta)
    frame = read_payload(data, metrics)
    print(f"Ready {name}: {len(frame):,} StreamCat records", flush=True)
    return name, frame, meta


def coverage_column(frame, table, scope):
    # NLCD metrics may belong to the RipBuf100 parent even for ordinary Cat/Ws AOIs.
    suffix = "_" + scope + "pctfull"
    options = [c for c in frame if c.endswith(suffix) and c.startswith(table.lower())]
    if len(options) != 1:
        raise ValueError(f"Expected one coverage column for {table}/{scope}; found {options}")
    return options[0]


def assemble(inventory, tables, threshold, landcover_year=None):
    ids = inventory.index.to_numpy(dtype=np.int64)
    values, coverage, names, units, sources = [], [], [], [], []

    def add(name, x, pct, unit, source, minimum=None, maximum=None):
        x = np.asarray(x, dtype=np.float64)
        pct = np.asarray(pct, dtype=np.float64)
        valid = np.isfinite(x) & ~np.isin(x, [-9999, -9998, -999, -99999])
        valid &= np.isfinite(pct) & (pct >= threshold) & (pct <= 100.001)
        if minimum is not None:
            valid &= x >= minimum
        if maximum is not None:
            valid &= x <= maximum
        values.append(np.where(valid, x, np.nan).astype(np.float32))
        coverage.append(pct.astype(np.float32))
        names.append(name)
        units.append(unit)
        sources.append(source)

    if landcover_year is None:
        for col, name, unit, factor, minimum in NHD:
            raw = pd.to_numeric(inventory[col], errors="raise").to_numpy(dtype=float, copy=True)
            raw[np.isin(raw, [-9999, -9998, -999, -99999])] = np.nan
            pct = np.where(np.isfinite(raw), 100.0, np.nan)
            if col == "areasqkm":
                present = inventory.has_catchment.eq(True).to_numpy()
                pct[~present] = np.nan
            add(name, raw * factor, pct, unit, "NHDPlusV2:" + col, minimum)
        groups = GROUPS
    else:
        groups = {f"NLCD{landcover_year}": [m + str(landcover_year) for m in LC]}
    for group, metrics in groups.items():
        frame = tables[group].reindex(ids)
        for m in metrics:
            for scope in ("cat", "ws"):
                name = m + scope
                pc = coverage_column(frame, group, scope)
                pct = pd.to_numeric(frame[pc], errors="raise").to_numpy(dtype=float, copy=True)
                # A reach without a local catchment has no valid local landscape summary.
                if scope == "cat":
                    pct[~inventory.has_catchment.eq(True).to_numpy()] = np.nan
                x = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float, copy=True)
                is_percent = m.startswith("pct") or m in ("clay", "sand", "om")
                add(
                    name,
                    x,
                    pct,
                    "percent" if landcover_year else UNITS[m],
                    f"StreamCat:{group}:{name}",
                    None if m == "elev" else 0,
                    100.001 if is_percent else None,
                )
    a = np.column_stack(values)
    ds = xr.Dataset(
        data_vars={
            "static_values": (("feature_id", "static_feature"), a),
            "static_valid": (("feature_id", "static_feature"), np.isfinite(a).astype(np.uint8)),
            "source_coverage_pct": (("feature_id", "static_feature"), np.column_stack(coverage)),
            "feature_units": ("static_feature", units),
            "feature_source": ("static_feature", sources),
        },
        coords={"feature_id": ids, "static_feature": names},
        attrs=dict(
            schema_version="1",
            nhd_version="NHDPlusV2",
            region="06",
            minimum_source_coverage_pct=float(threshold),
            missing_policy="NaN raw values; static_valid=0. Never drop graph nodes.",
            normalization="Raw units. Fit normalization on training nodes only.",
            time_policy="Terrain/soil/geology treated as invariant baseline descriptors.",
        ),
    )
    if landcover_year is not None:
        ds.attrs.update(
            snapshot_year=int(landcover_year),
            time_policy="Dated snapshot; not part of the core static inputs. No backward fill.",
        )
    return ds


def save_dataset(ds, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    encoding = {
        n: dict(zlib=True, complevel=4)
        for n in ("static_values", "static_valid", "source_coverage_pct")
    }
    ds.to_netcdf(tmp, engine="netcdf4", encoding=encoding)
    with xr.open_dataset(tmp, engine="netcdf4") as check:
        if not np.array_equal(check.feature_id.values, ds.feature_id.values):
            raise AssertionError("NetCDF round-trip changed COMID ordering")
        if not np.array_equal(
            check.static_valid.values.astype(bool), np.isfinite(check.static_values.values)
        ):
            raise AssertionError("Invalid static mask after write")
    os.replace(tmp, path)


def load_static_inputs(path, ids, *, fit_ids=None, normalization=None):
    """Return [nodes, 2*features] float32 values+mask, and scaler dictionary.

    Supply fit_ids=training COMIDs on training; reuse returned normalization on
    validation/test. Never fit on held-out nodes in a spatial-transfer experiment.
    Concatenate returned array along input's last axis after broadcasting over
    batch/time, or feed its values+mask to a separate static encoder.
    """
    with xr.open_dataset(path, engine="netcdf4") as opened:
        ds = opened.load()
    ids = validate_ids(ids, "requested nodes")
    names = ds.static_feature.values.tolist()
    # Log-compress nonnegative area/length and conductivity features before scaling.
    log = np.asarray(
        [
            ("area_km2" in n or "length_km" in n or n.startswith(("perm", "hydrlcond")))
            for n in names
        ]
    )

    def transformed(subset):
        x = ds.sel(feature_id=subset).static_values.values.astype(np.float64)
        x[:, log] = np.log1p(x[:, log])
        return x

    if normalization is None:
        if fit_ids is None:
            raise ValueError("Explicit fit_ids (training COMIDs) required to fit a scaler")
        fit = transformed(validate_ids(fit_ids, "training nodes"))
        valid = np.isfinite(fit)
        count = valid.sum(axis=0)
        mean = np.nansum(fit, axis=0) / np.maximum(count, 1)
        variance = np.nansum((fit - mean) ** 2, axis=0) / np.maximum(count, 1)
        std = np.sqrt(variance)
        std[std < 1e-6] = 1
        normalization = dict(
            features=names,
            mean=mean.tolist(),
            std=std.tolist(),
            count=count.tolist(),
            log1p=log.tolist(),
        )
    if normalization["features"] != names or normalization["log1p"] != log.tolist():
        raise ValueError("Static normalization schema mismatch")
    x = transformed(ids)
    mask = np.isfinite(x) & (np.asarray(normalization["count"]) > 0)
    x = np.where(mask, (x - normalization["mean"]) / normalization["std"], 0)
    result = np.concatenate((x, mask.astype(float)), axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite normalized static inputs")
    return result, normalization


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--min-coverage", type=float, default=99.0)
    p.add_argument("--landcover-years", type=int, nargs="*", default=[2001, 2006, 2011, 2016])
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    if not 0 <= args.min_coverage <= 100 or not 1 <= args.workers <= 4:
        p.error("min-coverage must be 0..100; workers must be 1..4")
    root = args.root.resolve()
    cache, out = root / "raw/streamcat", root / "processed/static"
    cache.mkdir(parents=True, exist_ok=True)
    lock = open(cache / ".download.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory_path = root / "metadata/tennessee_nwm_reaches.parquet"
    inventory = pd.read_parquet(inventory_path)
    inventory["comid"] = validate_ids(inventory.comid, str(inventory_path))
    inventory = inventory.set_index("comid")
    missing = ({x[0] for x in NHD} | {"has_catchment"}) - set(inventory.columns)
    if missing:
        raise ValueError(f"Inventory missing {sorted(missing)}; rerun existing script 03")
    references = sorted((root / "processed/six_hour_nodes").glob("*/*six_hour_nodes.nc"))
    if references:
        with xr.open_dataset(references[0], engine="netcdf4") as ds:
            ids = validate_ids(ds.feature_id.values, str(references[0]))
        if set(ids) != set(inventory.index):
            raise ValueError("Dynamic archive and inventory have different COMID sets")
        inventory = inventory.loc[ids]
    elif "nwm_feature_index" in inventory:
        inventory = inventory.sort_values("nwm_feature_index")
    jobs = dict(GROUPS)
    for year in args.landcover_years:
        jobs[f"NLCD{year}"] = [m + str(year) for m in LC]
    definitions = cache / "variable_info.csv"
    if not definitions.exists():
        if args.offline:
            raise FileNotFoundError(definitions)
        atomic_bytes(definitions, request(DEFINITIONS))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(lambda item: download_group(cache, *item, args.offline), jobs.items())
        )
    tables = {name: table for name, table, meta in results}
    # Different source rasters can contain different missing values, but should
    # expose the same regional COMID inventory. Fail on truncation or omissions.
    expected_ids = set(tables["STATSGO_Set1"].index)
    for name, table in tables.items():
        if set(table.index) != expected_ids:
            raise RuntimeError(f"Regional COMID set mismatch for {name}; inspect cached responses")
    core = assemble(inventory, tables, args.min_coverage)
    if (core.static_valid.sum("feature_id").values == 0).any():
        raise RuntimeError("A core static feature has no valid values; inspect source/schema")
    core_path = out / "tennessee_static_core.nc"
    save_dataset(core, core_path)
    paths = [core_path]
    for year in args.landcover_years:
        ds = assemble(inventory, tables, args.min_coverage, year)
        path = out / f"tennessee_landcover_{year}.nc"
        save_dataset(ds, path)
        paths.append(path)
    pilot_outputs = []
    graph_paths = sorted((root / "processed/pilot_graph").glob("*/nodes.parquet"))
    candidate = root / "processed/graph_candidate/nodes.parquet"
    if candidate.exists():
        graph_paths.append(candidate)
    for node_path in graph_paths:
        nodes = pd.read_parquet(node_path).sort_values("node_index")
        ids = validate_ids(nodes.comid, str(node_path))
        if not np.array_equal(nodes.node_index.values, np.arange(len(nodes))):
            raise ValueError(f"Noncontiguous node_index: {node_path}")
        subset = core.sel(feature_id=ids)
        subset["node_index"] = ("feature_id", np.arange(len(ids), dtype=np.int64))
        path = node_path.parent / "static_core.nc"
        save_dataset(subset, path)
        paths.append(path)
        pilot_outputs.append(dict(path=str(path.relative_to(root)), nodes=len(ids)))
    summary = pd.DataFrame(
        {
            "feature": core.static_feature.values,
            "unit": core.feature_units.values,
            "valid_nodes": core.static_valid.sum("feature_id").values,
            "total_nodes": len(inventory),
        }
    )
    summary["valid_pct"] = 100 * summary.valid_nodes / len(inventory)
    atomic_bytes(out / "static_coverage.csv", summary.to_csv(index=False).encode())
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        nodes=len(inventory),
        features=core.sizes["static_feature"],
        min_coverage=args.min_coverage,
        dynamic_files="processed/six_hour_nodes/*/*six_hour_nodes.nc",
        join_coordinate="feature_id (NHDPlusV2 COMID)",
        static_values="static_values",
        static_mask="static_valid",
        inventory_sha256=sha(inventory_path),
        source_definitions_sha256=sha(definitions),
        reference_dynamic_file=str(references[0]) if references else None,
        sources={name: meta for name, table, meta in results},
        streamcat_nodes_missing=int((~inventory.index.isin(expected_ids)).sum()),
        graph_outputs=pilot_outputs,
        files={str(path.relative_to(root)): sha(path) for path in paths},
        landcover_policy="Snapshots separate; no backfill before observation year.",
        units_note="perm and hydrlcond units follow METRIC_DESCRIPTION; source METRIC_UNITS is inconsistent.",
        training_status="Data prepared. Existing dynamic-only trainer does not consume statics automatically.",
    )
    write_json(out / "static_manifest.json", report)
    print(summary.to_string(index=False), flush=True)
    print(f"STATIC DATA READY: {len(inventory):,} nodes x {core.sizes['static_feature']} features")
    print(
        f"Core: {core_path}\nGraph copies: {len(pilot_outputs)}; NLCD snapshots: {args.landcover_years}"
    )


if __name__ == "__main__":
    main()
