"""Build missing graph artifacts for a new dataset; existing graphs are immutable."""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tva_flood.common import atomic_json, hash_array


def build_tables(inventory, connectivity):
    nodes = inventory.sort_values("nwm_feature_index").reset_index(drop=True).copy()
    if nodes.empty or not nodes.comid.is_unique or nodes.comid.isna().any():
        raise ValueError("Inventory COMIDs must be nonempty, unique and nonmissing")
    if not nodes.nwm_feature_index.is_unique or (nodes.nwm_feature_index < 0).any():
        raise ValueError("Inventory must have unique valid NWM feature indices")
    nodes["node_index"] = np.arange(len(nodes), dtype=np.int64)
    selected = connectivity.fromcomid.isin(nodes.comid) & connectivity.tocomid.isin(nodes.comid)
    edges = connectivity.loc[selected, ["fromcomid", "tocomid"]].drop_duplicates().copy()
    lookup = pd.Index(nodes.comid)
    edges["source"] = lookup.get_indexer(edges.fromcomid)
    edges["target"] = lookup.get_indexer(edges.tocomid)
    edges = edges.sort_values(["source", "target"]).reset_index(drop=True)
    return nodes, edges


def main():
    import geopandas as gpd
    import shapely

    from .inventory import integer_column, source_file

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    out = a.out or a.root / "processed/graph_candidate"
    if out.exists():
        p.error(
            "Graph directory already exists. Reuse it; rebuilding requires a new --out and downstream preparation."
        )
    nodes, edges = build_tables(
        pd.read_parquet(a.root / "metadata/tennessee_nwm_reaches.parquet"),
        pd.read_parquet(a.root / "metadata/tennessee_nhd_connectivity.parquet"),
    )
    catchments = gpd.read_file(
        source_file(a.root, "NHDPlusCatchment", "Catchment.shp"), engine="pyogrio"
    )
    catchments.columns = [c.lower() for c in catchments.columns]
    catchments["comid"] = integer_column(catchments, "featureid")
    catchments = catchments.loc[catchments.comid.isin(nodes.comid), ["comid", "geometry"]].copy()
    if catchments.crs is None or catchments.geometry.isna().any():
        raise ValueError("Catchment CRS or geometry missing")
    catchments = catchments.to_crs(5070)
    repaired = int((~catchments.is_valid).sum())
    catchments.geometry = shapely.make_valid(catchments.geometry)
    if not catchments.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Repair produced nonpolygon geometry; review source catchments")
    catchments = catchments.dissolve(by="comid", as_index=False)
    if catchments.is_empty.any() or not catchments.is_valid.all() or (catchments.area <= 0).any():
        raise ValueError("Invalid or empty catchments remain")
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".graph-", dir=out.parent))
    try:
        nodes.to_parquet(temp / "nodes.parquet", index=False)
        edges.to_parquet(temp / "edges.parquet", index=False)
        catchments.to_parquet(temp / "catchments_clean.parquet", index=False)
        atomic_json(
            temp / "graph_report.json",
            dict(
                nodes=len(nodes),
                edges=len(edges),
                catchments=len(catchments),
                repaired_geometries=repaired,
                missing_catchments=int((~nodes.comid.isin(catchments.comid)).sum()),
                node_order_sha256=hash_array(nodes.comid.to_numpy(dtype="<i8")),
                topology="NHDPlus; NWM routing equivalence unverified",
            ),
        )
        os.rename(temp, out)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    print(f"Graph saved: {out}; {len(nodes):,} nodes, {len(edges):,} edges")
