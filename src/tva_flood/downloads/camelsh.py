#!/usr/bin/env python3
"""Download CAMELS-H attributes and observed hourly discharge for Tennessee gauges.

Uses pinned metadata record 15066778 and observed-flow record 16729675.
Meteorological forcing remains AORC. Zip range reads avoid downloading other gauges.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd
import pandas as pd
import py7zr
from remotezip import RemoteZip
from shapely import points

from tva_flood.common import atomic_json


def retry(fn):
    for attempt in range(5):
        try:
            return fn()
        except Exception as exc:
            if attempt == 4:
                raise
            print(f"Retry: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(2**attempt)


def record(number):
    def load():
        with urlopen(f"https://zenodo.org/api/records/{number}", timeout=120) as response:
            return json.load(response)

    return retry(load)


def file_entry(metadata, name):
    matching = [x for x in metadata["files"] if x["key"].lower() == name.lower()]
    if len(matching) != 1:
        raise RuntimeError(f"Expected one {name} in Zenodo record")
    return matching[0]


def md5sum(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry, directory):
    path = directory / entry["key"]
    checksum = entry["checksum"].removeprefix("md5:")
    if path.exists():
        if path.stat().st_size == entry["size"] and md5sum(path) == checksum:
            return path
        raise RuntimeError(
            f"Existing download failed verification: {path}; move it aside and rerun"
        )
    temporary = path.with_suffix(path.suffix + ".part")

    def transfer():
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset == entry["size"]:
            return
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with urlopen(Request(entry["links"]["self"], headers=headers), timeout=180) as response:
            if offset and response.status == 200:
                offset = 0
            elif offset and not response.headers.get("Content-Range", "").startswith(
                f"bytes {offset}-"
            ):
                raise RuntimeError("Invalid resume response")
            with temporary.open("ab" if offset else "wb") as handle:
                shutil.copyfileobj(response, handle, length=8 * 1024**2)
        if temporary.stat().st_size != entry["size"]:
            raise RuntimeError("Incomplete download")

    retry(transfer)
    if md5sum(temporary) != checksum:
        raise RuntimeError(f"Checksum mismatch: {temporary}")
    os.replace(temporary, path)
    return path


def attributes(archive, destination):
    marker = destination / "complete.json"
    if marker.exists():
        return
    staging = destination.with_name(destination.name + ".extracting")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    with py7zr.SevenZipFile(archive) as package:
        for name in package.getnames():
            path = Path(name.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError("Unsafe archive path")
        package.extractall(staging)
    atomic_json(staging / "complete.json", {"archive_md5": md5sum(archive)})
    if destination.exists():
        raise RuntimeError(f"Incomplete attributes directory exists: {destination}")
    os.replace(staging, destination)


def extract_member(package, member, directory):
    info = package.getinfo(member)
    path = directory / Path(member).name
    marker = path.with_suffix(".complete.json")
    if path.exists() and marker.exists():
        previous = json.loads(marker.read_text())
        if previous["zip_crc32"] == info.CRC and path.stat().st_size == info.file_size:
            return path
    temporary = path.with_suffix(".nc.part")

    def transfer():
        # ZipExtFile checks the member CRC when the complete member is read.
        with package.open(member) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024**2)
        if temporary.stat().st_size != info.file_size:
            raise RuntimeError(f"Incomplete member: {member}")

    retry(transfer)
    os.replace(temporary, path)
    atomic_json(
        marker,
        {"member": member, "zip_crc32": info.CRC, "record": 16729675, "size_bytes": info.file_size},
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    parser.add_argument(
        "--full-zip",
        action="store_true",
        help="Fallback: download and verify the 4.2 GB national flow archive",
    )
    parser.add_argument(
        "--max-gauges", type=int, default=None, help="Limit downloads for a small test"
    )
    args = parser.parse_args()
    if args.max_gauges is not None and args.max_gauges < 1:
        parser.error("--max-gauges must be positive")
    output = args.root / "raw" / "camelsh"
    downloads, flow, meta = output / "downloads", output / "flow", output / "metadata"
    for folder in (downloads, flow, meta):
        folder.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 10 * 1024**3:
        raise RuntimeError("Less than 10 GiB free; stopped before download")
    print("Fetching CAMELS-H metadata and basin attributes ...", flush=True)
    metadata, discharge = record(15066778), record(16729675)
    atomic_json(meta / "zenodo_15066778.json", metadata)
    atomic_json(meta / "zenodo_16729675.json", discharge)
    archive = download(file_entry(metadata, "attributes.7z"), downloads)
    attributes(archive, output / "attributes")
    table_paths = list((output / "attributes").rglob("attributes_gageii_BasinID.csv"))
    if len(table_paths) != 1:
        raise RuntimeError("Missing or ambiguous basin-location table")
    locations = pd.read_csv(table_paths[0], dtype={"STAID": str, "HUC02": str})
    locations["STAID"] = locations["STAID"].str.strip().str.zfill(8)
    boundary = gpd.read_file(args.root / "raw/geography/tennessee_huc2.geojson").to_crs(4326)
    polygon = boundary.geometry.union_all()
    locations = locations.loc[
        polygon.covers(points(locations["LNG_GAGE"], locations["LAT_GAGE"]))
    ].copy()
    if locations.empty or locations["STAID"].duplicated().any():
        raise RuntimeError("Empty or duplicate Tennessee gauge selection")
    selected_ids = sorted(locations["STAID"].tolist())
    locations.to_csv(meta / "tennessee_camelsh_gauges.csv", index=False)
    (meta / "tennessee_camelsh_ids.txt").write_text("\n".join(selected_ids) + "\n")
    if args.max_gauges:
        selected_ids = selected_ids[: args.max_gauges]
    print(
        f"Tennessee gauges: {len(locations)}; requested this run: {len(selected_ids)}", flush=True
    )
    entry = file_entry(discharge, "Hourly2.zip")
    if args.full_zip:
        package = zipfile.ZipFile(download(entry, downloads))
    else:
        package = retry(lambda: RemoteZip(entry["links"]["self"], timeout=180))
    completed, missing = [], []
    with package:
        mapping = {}
        for name in package.namelist():
            match = re.fullmatch(r"(\d{8,15})_hourly\.nc", Path(name).name, flags=re.I)
            if match:
                if match.group(1) in mapping:
                    raise RuntimeError("Duplicate gauge files in ZIP")
                mapping[match.group(1)] = name
        locations["in_hourly_release"] = locations["STAID"].isin(mapping)
        locations.to_csv(meta / "tennessee_camelsh_gauges.csv", index=False)
        for index, gauge in enumerate(selected_ids, start=1):
            if gauge not in mapping:
                missing.append(gauge)
                continue
            extract_member(package, mapping[gauge], flow)
            completed.append(gauge)
            if index % 10 == 0 or index == len(selected_ids):
                print(f"  Checked/downloaded {index}/{len(selected_ids)} gauges", flush=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_tennessee_gauges": len(locations),
        "requested_ids": selected_ids,
        "flow_files_present": completed,
        "not_in_hourly_release": missing,
        "attributes_record": 15066778,
        "observed_flow_record": 16729675,
        "nominal_flow_period": "1980-01-01 through 2024-12-31 UTC; actual coverage varies by gauge",
        "note": "File presence does not establish valid observations in training or test dates.",
    }
    atomic_json(meta / "camelsh_download_report.json", report)
    print(f"Done. Flow files: {len(completed)}; not in hourly release: {len(missing)}", flush=True)
    print(f"Output: {output}", flush=True)


if __name__ == "__main__":
    main()
