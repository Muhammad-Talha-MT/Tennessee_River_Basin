#!/usr/bin/env python3
"""Match all Tennessee NHDPlus V2 reaches to NWM v3.0 metadata; no discharge download."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import s3fs
import zarr

from tva_flood.common import atomic_json

STORE = "noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"


def source_file(root, component, basename):
    manifest = json.loads((root / "metadata" / "nhdplus_download.json").read_text())
    records = [r for r in manifest["components"] if r["component"] == component]
    if len(records) != 1:
        raise RuntimeError(f"Expected one manifest entry for {component}")
    directory = Path(records[0]["extracted_directory"])
    paths = [
        directory / n for n in records[0]["members"] if Path(n).name.lower() == basename.lower()
    ]
    if len(paths) != 1 or not paths[0].exists():
        raise RuntimeError(f"Missing or ambiguous source file: {basename}")
    return paths[0]


def read_table(path):
    table = pyogrio.read_dataframe(path, read_geometry=False)
    table.columns = [c.lower() for c in table.columns]
    return table


def integer_column(table, column):
    values = pd.to_numeric(table[column], errors="raise")
    if values.isna().any() or (values != np.floor(values)).any():
        raise RuntimeError(f"Noninteger or missing IDs in {column}")
    return values.astype("int64")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    args = parser.parse_args()
    root = args.root
    output = root / "metadata"
    output.mkdir(parents=True, exist_ok=True)
    print("Reading Tennessee flowlines and catchment IDs ...", flush=True)
    flow_path = source_file(root, "NHDSnapshot", "NHDFlowline.shp")
    reaches = gpd.read_file(flow_path, engine="pyogrio")
    reaches.columns = [c.lower() for c in reaches.columns]
    reaches["comid"] = integer_column(reaches, "comid")
    if reaches["comid"].duplicated().any():
        raise RuntimeError("Duplicate flowline COMIDs")
    if reaches.crs is None:
        raise RuntimeError("Flowline CRS is missing")
    for basename in ("PlusFlowlineVAA.dbf", "elevslope.dbf"):
        table = read_table(source_file(root, "NHDPlusAttributes", basename))
        table["comid"] = integer_column(table, "comid")
        if table["comid"].duplicated().any():
            raise RuntimeError(f"Duplicate COMIDs in {basename}")
        columns = ["comid"] + [c for c in table.columns if c not in reaches.columns]
        reaches = reaches.merge(table[columns], on="comid", how="left", validate="one_to_one")
    catchments = read_table(source_file(root, "NHDPlusCatchment", "Catchment.shp"))
    catchment_ids = integer_column(catchments, "featureid")
    reaches["has_catchment"] = reaches["comid"].isin(catchment_ids)
    connectivity = read_table(source_file(root, "NHDPlusAttributes", "PlusFlow.dbf"))
    for c in ("fromcomid", "tocomid"):
        connectivity[c] = integer_column(connectivity, c)

    print("Reading NWM v3.0 reach coordinate and metadata (no streamflow values) ...", flush=True)
    fs = s3fs.S3FileSystem(anon=True)
    group = zarr.open_consolidated(s3fs.S3Map(root=STORE, s3=fs, check=False), mode="r")
    feature_ids = np.asarray(group["feature_id"][:], dtype="int64")
    index = pd.Index(feature_ids)
    if not index.is_unique:
        raise RuntimeError("NWM feature_id coordinate is not unique")
    reaches["nwm_feature_index"] = index.get_indexer(reaches["comid"])
    reaches["in_nwm_v30"] = reaches["nwm_feature_index"] >= 0
    streamflow = group["streamflow"]
    dims = list(streamflow.attrs["_ARRAY_DIMENSIONS"])
    if dims != ["time", "feature_id"]:
        raise RuntimeError(f"Unexpected NWM discharge dimensions: {dims}")
    time_array = group["time"]
    time_units = time_array.attrs["units"]
    if not time_units.startswith("hours since "):
        raise RuntimeError(f"Unexpected time units: {time_units}")
    origin = pd.Timestamp(time_units.removeprefix("hours since "))
    first = origin + pd.Timedelta(hours=int(time_array[0]))
    last = origin + pd.Timedelta(hours=int(time_array[-1]))
    matched = reaches.loc[reaches["in_nwm_v30"]].sort_values("nwm_feature_index").copy()
    unmatched = reaches.loc[~reaches["in_nwm_v30"]].copy()
    if matched.empty:
        raise RuntimeError("No Tennessee reaches matched NWM v3.0")
    inventory = pd.DataFrame(reaches.drop(columns="geometry"))
    matched_inventory = pd.DataFrame(matched.drop(columns="geometry"))
    inventory.to_csv(output / "tennessee_reach_inventory.csv", index=False)
    matched_inventory.to_parquet(output / "tennessee_nwm_reaches.parquet", index=False)
    pd.DataFrame(unmatched.drop(columns="geometry")).to_csv(
        output / "tennessee_unmatched_reaches.csv", index=False
    )
    reaches.to_parquet(output / "tennessee_flowlines.parquet", index=False)
    # Preserve every source connection, including connections crossing the VPU boundary.
    connectivity.to_parquet(output / "tennessee_nhd_connectivity.parquet", index=False)
    source_chunk_ids = np.unique(matched["nwm_feature_index"].to_numpy() // streamflow.chunks[1])
    selected_ids = set(matched["comid"].tolist())
    internal = connectivity["fromcomid"].isin(selected_ids) & connectivity["tocomid"].isin(
        selected_ids
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "nhd_region": "06",
        "nwm_store": "s3://" + STORE,
        "nwm_version": "3.0",
        "nhd_flowlines": len(reaches),
        "catchment_records": len(catchments),
        "matched_nwm_reaches": len(matched),
        "unmatched_nhd_reaches": len(unmatched),
        "matched_reaches_without_catchments": int((~matched["has_catchment"]).sum()),
        "nhd_connections_between_matched_reaches": int(internal.sum()),
        "source_discharge_shape": list(streamflow.shape),
        "source_discharge_chunks": list(streamflow.chunks),
        "source_discharge_dtype": str(streamflow.dtype),
        "source_discharge_attributes": dict(streamflow.attrs),
        "source_feature_chunks_touched": len(source_chunk_ids),
        "source_feature_chunk_ids": source_chunk_ids.tolist(),
        "first_available_time": str(first),
        "last_available_time": str(last),
        "regional_discharge_float32_uncompressed_gib": len(matched)
        * streamflow.shape[0]
        * 4
        / 1024**3,
        "note": "ID matching does not validate NWM routing equivalence. NHD topology is preserved for later checks.",
    }
    atomic_json(output / "reach_inventory_report.json", report)
    print(f"NHD flowlines: {len(reaches):,}")
    print(f"Matched NWM reaches: {len(matched):,}")
    print(f"Unmatched NHD reaches: {len(unmatched):,}")
    print(f"Matched reaches without catchments: {report['matched_reaches_without_catchments']:,}")
    print(f"Archive dates: {first} to {last}")
    print(
        f"Full-record regional discharge as float32: {report['regional_discharge_float32_uncompressed_gib']:.1f} GiB before compression"
    )
    print(f"NWM reach chunks touched: {len(source_chunk_ids)}")
    print(f"Done. Report: {output / 'reach_inventory_report.json'}")


if __name__ == "__main__":
    main()
