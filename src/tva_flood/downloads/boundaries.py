#!/usr/bin/env python3
"""Download USGS WBD HUC2 06 and all its HUC8 polygons. Standard library only."""

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SERVICE = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer"


def fetch(url, params):
    request = Request(
        url + "?" + urlencode(params), headers={"User-Agent": "TVA-hydrology-data/1.0"}
    )
    for attempt in range(5):
        try:
            with urlopen(request, timeout=180) as response:
                data = json.load(response)
            if "error" in data:
                raise RuntimeError(json.dumps(data["error"]))
            return data
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def download_layer(field, where, output):
    # Find layers and field names from current service metadata.
    service = fetch(SERVICE, {"f": "json"})
    matching = []
    for layer in service["layers"]:
        if layer.get("geometryType") != "esriGeometryPolygon":
            continue
        digits = field.removeprefix("huc")
        if not layer["name"].startswith(digits + "-digit"):
            continue
        metadata = fetch(f"{SERVICE}/{layer['id']}", {"f": "json"})
        fields = {f["name"].lower(): f["name"] for f in metadata["fields"]}
        if field in fields:
            matching.append((layer["id"], fields[field], metadata))
    if len(matching) != 1:
        raise RuntimeError(f"Expected one {field} layer; found {len(matching)}")
    layer_id, actual_field, metadata = matching[0]
    where = where.replace(field, actual_field)
    query_url = f"{SERVICE}/{layer_id}/query"
    id_response = fetch(query_url, {"f": "json", "where": where, "returnIdsOnly": "true"})
    ids = sorted(id_response.get("objectIds") or [])
    if not ids:
        raise RuntimeError(f"No features for {where}")
    features = []
    for start in range(0, len(ids), 10):
        batch = ids[start : start + 10]
        response = fetch(
            query_url,
            {
                "f": "geojson",
                "objectIds": ",".join(map(str, batch)),
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
            },
        )
        if response.get("exceededTransferLimit") or len(response.get("features", [])) != len(batch):
            raise RuntimeError("Incomplete feature response; refusing to save")
        features.extend(response["features"])
        print(f"  {field}: {len(features)}/{len(ids)} polygons", flush=True)
    codes = []
    for feature in features:
        properties = {k.lower(): v for k, v in feature["properties"].items()}
        code = str(properties[field])
        if not code.startswith("06") or len(code) != int(field[3:]):
            raise RuntimeError(f"Unexpected {field}: {code}")
        geometry = feature.get("geometry")
        if (
            not geometry
            or geometry["type"] not in ("Polygon", "MultiPolygon")
            or not geometry.get("coordinates")
        ):
            raise RuntimeError(f"Missing polygon for {code}")
        codes.append(code)
    if len(set(codes)) != len(codes):
        raise RuntimeError("Duplicate hydrologic units")
    features.sort(key=lambda f: str({k.lower(): v for k, v in f["properties"].items()}[field]))
    write_json(output, {"type": "FeatureCollection", "features": features})
    write_json(output.with_suffix(".source.json"), metadata)
    return features, {
        "layer": layer_id,
        "where": where,
        "count": len(features),
        "file": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    args = parser.parse_args()
    geographic = args.root / "raw" / "geography"
    metadata = args.root / "metadata"
    geographic.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    print("Downloading Tennessee River basin: HUC2 06", flush=True)
    basin, basin_info = download_layer("huc2", "huc2 = '06'", geographic / "tennessee_huc2.geojson")
    if len(basin) != 1:
        raise RuntimeError("Expected one Tennessee basin polygon")
    units, units_info = download_layer(
        "huc8", "huc8 LIKE '06%'", geographic / "tennessee_huc8.geojson"
    )
    inventory = metadata / "tennessee_huc8_inventory.csv"
    temporary = inventory.with_suffix(".csv.part")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["huc8", "name"])
        writer.writeheader()
        for feature in units:
            p = {k.lower(): v for k, v in feature["properties"].items()}
            writer.writerow({"huc8": p["huc8"], "name": p.get("name", "")})
    os.replace(temporary, inventory)
    write_json(
        metadata / "boundary_download.json",
        {
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "source": SERVICE,
            "crs": "EPSG:4326",
            "region": "06",
            "downloads": [basin_info, units_info],
        },
    )
    print(f"Done: 1 basin and {len(units)} HUC8 polygons.")
    print(f"Inventory: {inventory}")
    print(f"Polygons: {geographic}")


if __name__ == "__main__":
    main()
