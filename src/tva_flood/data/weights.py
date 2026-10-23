#!/usr/bin/env python3
"""Build area-weighted AORC mappings without reading meteorological values.

Rows: every graph node in nodes.parquet node_index order.
Columns: latitude-major, longitude-minor cells in the source file axis order.
Nodes lacking a polygon have zero rows and forcing_available=False.
"""

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from scipy.sparse import coo_matrix, save_npz
from tqdm import tqdm


def axis_edges(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid {name} coordinates")
    delta = np.diff(values)
    if not ((delta > 0).all() or (delta < 0).all()):
        raise ValueError(f"Nonmonotonic {name} coordinates")
    if not np.allclose(delta, np.median(delta), rtol=1e-5, atol=1e-8):
        raise ValueError(f"Irregular {name} spacing")
    return np.r_[
        values[0] - delta[0] / 2, (values[:-1] + values[1:]) / 2, values[-1] + delta[-1] / 2
    ]


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for arr in arrays:
        digest.update(np.asarray(arr).tobytes())
    return digest.hexdigest()


def build_weights(nodes, catchments, latitude, longitude, batch_size=500):
    n = len(nodes)
    if not np.array_equal(nodes.node_index.to_numpy(), np.arange(n)):
        raise ValueError("Graph node_index must be consecutive and ordered")
    if not nodes.comid.is_unique or not catchments.comid.is_unique:
        raise ValueError("Duplicate node or catchment COMID")
    if not catchments.comid.isin(nodes.comid).all():
        raise ValueError("Catchments include COMIDs absent from the graph")
    if catchments.crs is None:
        raise ValueError("Catchment CRS missing")
    catchments = catchments.to_crs(5070).copy()
    if (
        not catchments.geometry.notna().all()
        or not catchments.is_valid.all()
        or catchments.is_empty.any()
    ):
        raise ValueError("Catchment geometries must be valid and nonempty")
    if not catchments.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Nonpolygonal catchment geometry")
    polygon_areas = catchments.area.to_numpy()
    if not np.isfinite(polygon_areas).all() or (polygon_areas <= 0).any():
        raise ValueError("Invalid catchment area")
    node_lookup = pd.Series(nodes.node_index.to_numpy(), index=nodes.comid)
    catchment_rows = catchments.comid.map(node_lookup).to_numpy(dtype=np.int64)

    lat_edges = axis_edges(latitude, "latitude")
    lon_edges = axis_edges(longitude, "longitude")
    nlat, nlon = len(latitude), len(longitude)
    ncells = nlat * nlon
    cell_geometries = np.empty(ncells, dtype=object)
    # Densify geographic edges before projection so projected cell boundaries
    # approximate the parallels/meridians, rather than long straight chords.
    max_segment = min(np.abs(np.diff(lat_edges)).min(), np.abs(np.diff(lon_edges)).min()) / 4
    for start in tqdm(range(0, ncells, 20000), desc="Projecting grid cells"):
        stop = min(start + 20000, ncells)
        flat = np.arange(start, stop)
        iy, ix = flat // nlon, flat % nlon
        a, b = lon_edges[ix], lon_edges[ix + 1]
        c, d = lat_edges[iy], lat_edges[iy + 1]
        boxes = shapely.box(np.minimum(a, b), np.minimum(c, d), np.maximum(a, b), np.maximum(c, d))
        boxes = shapely.segmentize(boxes, max_segment)
        cell_geometries[start:stop] = np.asarray(gpd.GeoSeries(boxes, crs=4326).to_crs(5070).array)
    tree = shapely.STRtree(cell_geometries)
    polygons = np.asarray(catchments.geometry.array)
    rows, columns, areas = [], [], []
    for start in tqdm(range(0, len(polygons), batch_size), desc="Intersecting catchments"):
        batch = polygons[start : start + batch_size]
        local, cell = tree.query(batch, predicate="intersects")
        if len(cell) == 0:
            continue
        intersections = shapely.intersection(batch[local], cell_geometries[cell])
        overlap = shapely.area(intersections)
        positive = np.isfinite(overlap) & (overlap > 0)
        rows.append(catchment_rows[start + local[positive]])
        columns.append(cell[positive])
        areas.append(overlap[positive])
    if not areas:
        raise ValueError("No positive catchment/grid overlap")
    overlap_matrix = coo_matrix(
        (np.concatenate(areas), (np.concatenate(rows), np.concatenate(columns))),
        shape=(n, ncells),
        dtype=np.float64,
    ).tocsr()
    overlap_matrix.sum_duplicates()
    overlap_sum = np.asarray(overlap_matrix.sum(axis=1)).ravel()
    node_areas = np.zeros(n, dtype=np.float64)
    node_areas[catchment_rows] = polygon_areas
    available = node_areas > 0
    coverage = np.full(n, np.nan)
    coverage[available] = overlap_sum[available] / node_areas[available]
    # Preserve raw coverage diagnostics before normalizing tiny numerical errors.
    passed = available & np.isfinite(coverage) & (np.abs(coverage - 1) <= 0.001)
    inverse = np.zeros(n, dtype=np.float64)
    inverse[passed] = 1 / overlap_sum[passed]
    weights = overlap_matrix.multiply(inverse[:, None]).tocsr().astype(np.float32)
    weight_sum = np.asarray(weights.sum(axis=1)).ravel()
    audit = nodes[["node_index", "comid"]].copy()
    audit["polygon_present"] = available
    audit["polygon_area_km2"] = np.where(available, node_areas / 1e6, np.nan)
    audit["grid_coverage_fraction"] = coverage
    audit["forcing_available"] = passed
    audit["overlapping_cells"] = np.diff(overlap_matrix.indptr)
    audit["weight_sum"] = weight_sum
    return weights, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    graph = args.root / "processed/graph_candidate"
    out = args.root / "processed/aorc_mapping"
    out.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_parquet(graph / "nodes.parquet").sort_values("node_index")
    catchments = gpd.read_parquet(graph / "catchments_clean.parquet")
    sources = sorted((args.root / "raw/aorc").glob("*/tennessee_aorc_APCP_surface_*.nc"))
    if not sources:
        raise FileNotFoundError("No AORC precipitation files")
    with xr.open_dataset(sources[0], engine="netcdf4") as ds:
        latitude = np.asarray(ds.latitude.values, dtype=np.float64)
        longitude = np.asarray(ds.longitude.values, dtype=np.float64)
        if set(ds.APCP_surface.dims) != {"time", "latitude", "longitude"}:
            raise ValueError("Unexpected AORC dimensions")
    weights, audit = build_weights(nodes, catchments, latitude, longitude, args.batch_size)
    audit.to_csv(out / "coverage_audit.csv", index=False)
    failed = audit.polygon_present & ~audit.forcing_available
    audit.loc[failed].to_csv(out / "coverage_failures.csv", index=False)
    if failed.any():
        raise RuntimeError(
            f"{int(failed.sum())} catchments fail coverage; "
            "inspect coverage_failures.csv. No weights saved."
        )
    available = audit.forcing_available.to_numpy()
    if not np.allclose(audit.loc[available, "weight_sum"], 1, atol=1e-6):
        raise RuntimeError("Normalized weights do not sum to one")
    if not np.isfinite(weights.data).all() or (weights.data < 0).any():
        raise RuntimeError("Nonfinite or negative weights")
    save_npz(out / "aorc_to_nodes_weights.npz", weights)
    np.savez_compressed(
        out / "grid_and_node_order.npz",
        latitude=latitude,
        longitude=longitude,
        node_comid=nodes.comid.to_numpy(dtype=np.int64),
        forcing_available=available,
    )
    report = {
        "graph_nodes": len(nodes),
        "mapped_nodes": int(available.sum()),
        "nodes_without_catchments": int((~audit.polygon_present).sum()),
        "grid_shape": [len(latitude), len(longitude)],
        "matrix_shape": list(weights.shape),
        "nonzero_weights": int(weights.nnz),
        "minimum_polygon_coverage": float(audit.grid_coverage_fraction.min()),
        "maximum_polygon_coverage": float(audit.grid_coverage_fraction.max()),
        "maximum_weight_sum_error": float(np.abs(audit.loc[available, "weight_sum"] - 1).max()),
        "grid_source": str(sources[0]),
        "grid_sha256": fingerprint(latitude.astype("<f8"), longitude.astype("<f8")),
        "node_order_sha256": fingerprint(nodes.comid.to_numpy(dtype="<i8")),
        "column_order": "C-order flatten of (latitude, longitude), original axis order",
        "row_order": "node_index from graph_candidate/nodes.parquet",
        "area_crs": "EPSG:5070",
        "coverage_tolerance": 0.001,
        "geographic_cell_edges": "densified to at most quarter-cell segments",
        "missing_catchment_policy": "zero weight rows; forcing_available=False; unresolved",
    }
    (out / "mapping_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved mapping: {out}", flush=True)


if __name__ == "__main__":
    main()
