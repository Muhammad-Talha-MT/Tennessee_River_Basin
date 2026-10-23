"""Optional offline tests for the download/geospatial dependency group."""

import sys

import numpy as np
import pandas as pd
import pytest


def test_graph_command_and_area_weights(tmp_path, monkeypatch):
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")
    pytest.importorskip("s3fs")
    pytest.importorskip("zarr")
    from tva_flood.data import graph, inventory
    from tva_flood.data.weights import build_weights

    meta = tmp_path / "metadata"
    meta.mkdir()
    pd.DataFrame({"comid": [10, 20], "nwm_feature_index": [0, 1]}).to_parquet(
        meta / "tennessee_nwm_reaches.parquet"
    )
    pd.DataFrame({"fromcomid": [10], "tocomid": [20]}).to_parquet(
        meta / "tennessee_nhd_connectivity.parquet"
    )
    path = tmp_path / "catchment.geojson"
    gpd.GeoDataFrame(
        {"featureid": [10]}, geometry=[shapely.box(-85.2, 34.8, -84.8, 35.2)], crs=4326
    ).to_file(path, driver="GeoJSON")
    monkeypatch.setattr(inventory, "source_file", lambda *args: path)
    monkeypatch.setattr(sys, "argv", ["tva-flood", "--root", str(tmp_path)])
    graph.main()
    output = tmp_path / "processed/graph_candidate"
    nodes = pd.read_parquet(output / "nodes.parquet")
    catch = gpd.read_parquet(output / "catchments_clean.parquet")
    weights, audit = build_weights(nodes, catch, np.array([35.0, 36.0]), np.array([-85.0, -84.0]))
    np.testing.assert_allclose(weights.toarray(), [[1, 0, 0, 0], [0, 0, 0, 0]], atol=1e-6)
    assert audit.forcing_available.tolist() == [True, False]
    with pytest.raises(SystemExit):
        graph.main()
