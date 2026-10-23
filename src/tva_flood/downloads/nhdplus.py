#!/usr/bin/env python3
"""Download and extract EPA's core NHDPlus V2.1 vector archives for Tennessee (06)."""

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from tva_flood.common import atomic_json

BUCKET = "https://dmap-data-commons-ow.s3.amazonaws.com"
PREFIX = "NHDPlusV21/Data/NHDPlusMS/NHDPlus06/"
COMPONENTS = ("NHDSnapshot", "NHDPlusAttributes", "NHDPlusCatchment")
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def discover():
    entries = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": PREFIX, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        for attempt in range(5):
            try:
                with urlopen(BUCKET + "/?" + urlencode(params), timeout=120) as response:
                    root = ET.parse(response).getroot()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        for item in root.findall("s:Contents", NS):
            entries.append(
                {
                    "key": item.findtext("s:Key", namespaces=NS),
                    "size_bytes": int(item.findtext("s:Size", namespaces=NS)),
                    "etag": item.findtext("s:ETag", namespaces=NS).strip('"'),
                    "last_modified": item.findtext("s:LastModified", namespaces=NS),
                }
            )
        if root.findtext("s:IsTruncated", namespaces=NS) != "true":
            break
        token = root.findtext("s:NextContinuationToken", namespaces=NS)
        if not token:
            raise RuntimeError("Truncated listing without continuation token")
    selected = []
    for component in COMPONENTS:
        pattern = re.compile(r"NHDPlusV21_MS_06_" + component + r"_(\d+)\.7z$")
        candidates = [
            (int(m.group(1)), x) for x in entries if (m := pattern.fullmatch(Path(x["key"]).name))
        ]
        if not candidates:
            raise RuntimeError(f"Missing component: {component}")
        entry = dict(max(candidates, key=lambda x: x[0])[1])
        entry.update(component=component, url=BUCKET + "/" + quote(entry["key"], safe="/"))
        selected.append(entry)
    return selected


def download(entry, folder):
    destination = folder / Path(entry["key"]).name
    partial = destination.with_suffix(destination.suffix + ".part")
    expected = entry["size_bytes"]
    if destination.exists():
        if destination.stat().st_size != expected:
            raise RuntimeError(
                f"Existing file has wrong size: {destination}; move it aside and retry"
            )
        print(f"  Already downloaded: {destination.name}", flush=True)
        return destination
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            raise RuntimeError(f"Partial file exceeds expected size: {partial}")
        if offset == expected:
            os.replace(partial, destination)
            return destination
        headers = {"User-Agent": "TVA-hydrology-data/1.0", "If-Match": '"' + entry["etag"] + '"'}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(entry["url"], headers=headers), timeout=180) as response:
                if offset and response.status == 200:
                    offset = 0  # Server ignored Range; replace rather than append.
                elif offset and response.status != 206:
                    raise RuntimeError(f"Unexpected response status: {response.status}")
                elif offset and not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                ):
                    raise RuntimeError("Range response starts at the wrong offset")
                with partial.open("ab" if offset else "wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            if partial.stat().st_size != expected:
                raise RuntimeError("Download size mismatch")
            os.replace(partial, destination)
            print(f"  Downloaded: {destination.name}", flush=True)
            return destination
        except Exception as exc:
            if attempt == 4:
                raise
            print(f"  Retrying after {type(exc).__name__}: {exc}", flush=True)
            time.sleep(2**attempt)


def extract(archive, entry, folder):
    import py7zr

    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    checksum = digest.hexdigest()
    destination = folder / archive.stem
    marker = destination / "extraction_complete.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("sha256") == checksum and all(
            (destination / n).exists() for n in previous["members"]
        ):
            print(f"  Already extracted: {entry['component']}", flush=True)
            return previous
    staging = folder / (archive.stem + ".extracting")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    with py7zr.SevenZipFile(archive, mode="r") as package:
        members = package.getnames()
        for name in members:
            candidate = Path(name.replace("\\", "/"))
            if candidate.is_absolute() or ".." in candidate.parts:
                raise RuntimeError(f"Unsafe archive path: {name}")
        bad = package.testzip()
        if bad is not None:
            raise RuntimeError(f"Archive CRC check failed: {bad}")
    with py7zr.SevenZipFile(archive, mode="r") as package:
        package.extractall(path=staging)
    if not all((staging / name).exists() for name in members):
        raise RuntimeError("Extraction missing archive members")
    info = dict(
        entry,
        archive=str(archive),
        extracted_directory=str(destination),
        sha256=checksum,
        members=members,
    )
    atomic_json(staging / marker.name, info)
    if destination.exists():
        raise RuntimeError(
            f"Incomplete extraction directory exists: {destination}; move it aside and retry"
        )
    os.replace(staging, destination)
    print(f"  Extracted: {entry['component']} ({len(members)} members)", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    parser.add_argument(
        "--inspect", action="store_true", help="List archive sizes without downloading"
    )
    args = parser.parse_args()
    entries = discover()
    for entry in entries:
        print(f"{Path(entry['key']).name}: {entry['size_bytes'] / 1024**2:.1f} MiB", flush=True)
    print(
        f"Total compressed: {sum(x['size_bytes'] for x in entries) / 1024**2:.1f} MiB", flush=True
    )
    if args.inspect:
        return
    try:
        import py7zr  # noqa: F401 -- availability check before downloading
    except ImportError:
        raise SystemExit("Install py7zr in your active environment: python -m pip install py7zr")
    archives = args.root / "raw" / "geography" / "nhdplus_v21" / "archives"
    extracted = archives.parent / "extracted"
    metadata = args.root / "metadata"
    for folder in (archives, extracted, metadata):
        folder.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in entries:
        archive = download(entry, archives)
        records.append(extract(archive, entry, extracted))
    atomic_json(
        metadata / "nhdplus_download.json",
        {
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "region": "06",
            "source_page": "https://www.epa.gov/waterdata/nhdplus-tennessee-data-vector-processing-unit-06",
            "components": records,
        },
    )
    print("\nShapefiles and tables:")
    for path in sorted(extracted.rglob("*")):
        if path.suffix.lower() in (".shp", ".dbf"):
            print(path.relative_to(args.root))
    print(f"\nDone. Source metadata: {metadata / 'nhdplus_download.json'}")


if __name__ == "__main__":
    main()
