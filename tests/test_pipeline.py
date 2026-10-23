"""Offline numerical and tiny end-to-end tests; no GPU or remote downloads."""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from tva_flood.data.dataset import CachedWindows
from tva_flood.data.schema import WEATHER
from tva_flood.data.statics import GROUPS, NHD, assemble, save_dataset
from tva_flood.data.transforms import transform_precipitation
from tva_flood.evaluation.metrics import metrics, sums
from tva_flood.models.sparse import sparse_supports


def run(*args):
    result = subprocess.run(
        [sys.executable, "-m", "tva_flood", *map(str, args)],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    (root / "metadata").mkdir()
    inv = pd.DataFrame({c: [1.0, 2.0, 3.0] for c, *_ in NHD})
    inv["comid"] = [10, 20, 30]
    inv["has_catchment"] = True
    inv["nwm_feature_index"] = [0, 1, 2]
    inv.to_parquet(root / "metadata/tennessee_nwm_reaches.parquet", index=False)
    pd.DataFrame({"fromcomid": [10, 20], "tocomid": [20, 30]}).to_parquet(
        root / "metadata/tennessee_nhd_connectivity.parquet", index=False
    )
    tables = {}
    for group, features in GROUPS.items():
        table = pd.DataFrame(index=[10, 20, 30])
        for scope in ("cat", "ws"):
            for feature in features:
                table[feature + scope] = [2.0, 3.0, 4.0]
            table[group.lower() + "_" + scope + "pctfull"] = 100.0
        tables[group] = table
    save_dataset(
        assemble(inv.set_index("comid"), tables, 99),
        root / "processed/static/tennessee_static_core.nc",
    )
    for month, first in enumerate(
        pd.date_range("2018-02-01", "2019-02-01", freq="MS", inclusive="left")
    ):
        last = first + pd.offsets.MonthBegin(1)
        times = pd.date_range(first, last, freq="6h", inclusive="left")
        ids = np.roll([10, 20, 30], month % 3)
        data = {}
        for i, name in enumerate(WEATHER + ["streamflow"]):
            # Node-specific values expose ordering bugs; future timestamps vary.
            a = (np.arange(len(times))[:, None] % 11 + ids[None, :] / 10 + i).astype("float32")
            counts = np.full(a.shape, 6, dtype="uint8")
            counts[3::17, 0] = 0
            a[3::17, 0] = np.nan
            data[name] = (("time", "feature_id"), a)
            data[name + "_valid_hours"] = (("time", "feature_id"), counts)
        path = (
            root
            / "processed/six_hour_nodes"
            / str(first.year)
            / f"tennessee_{first:%Y%m%dT%H}_{last:%Y%m%dT%H}_six_hour_nodes.nc"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset(data, coords={"time": times, "feature_id": ids}).to_netcdf(
            path, engine="netcdf4"
        )
    run(
        "prepare",
        "--root",
        root,
        "--train-start",
        "2018-02-01",
        "--train-stop",
        "2018-04-01",
        "--validation-stop",
        "2019-02-01",
    )
    return root


def test_precipitation_masks_and_negative_rejection():
    np.testing.assert_allclose(transform_precipitation([0.0, 3.0, np.nan]), [0.0, np.log(4.0), 0.0])
    with pytest.raises(ValueError):
        transform_precipitation([-1.0])


def test_sparse_supports_are_row_normalized_both_directions():
    edges = np.array([[0, 1], [1, 2], [0, 1]])
    down, up = (support.to_dense() for support in sparse_supports(3, edges))
    # Receiver-by-sender: each sender's outgoing weights (self loop included) sum to one.
    torch.testing.assert_close(down.sum(0), torch.ones(3))
    torch.testing.assert_close(up.sum(0), torch.ones(3))
    assert down[1, 0] > 0 and down[0, 1] == 0 and up[0, 1] > 0 and up[1, 0] == 0


def test_metrics_additivity():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[1, 0], [1, 1]], dtype=bool)
    a = sums(np.log1p(y + 0.5), y + 0.5, np.log1p(y), y, mask, 0.0, 1.0)
    parts = sum(
        sums(np.log1p(y[i] + 0.5), y[i] + 0.5, np.log1p(y[i]), y[i], mask[i], 0.0, 1.0)
        for i in range(2)
    )
    np.testing.assert_allclose(a, parts)
    assert metrics(a)["mae_m3_s"] == 0.5
    assert metrics(np.zeros(7))["nse"] is None


def test_windows_split_boundaries_and_comid_alignment(prepared):
    d = CachedWindows(prepared, prepared / "processed/pretraining_full", "train")
    assert d.times[d.issues[-1] + 39] < pd.Timestamp("2018-04-01")
    # Reorder source monthly IDs by COMID, rather than relying on array positions.
    v = CachedWindows(prepared, prepared / "processed/pretraining_full", "validation")
    issue = int(v.issues[0])
    _, _, flow, _ = v.window(issue, issue + 1)
    month = int(np.searchsorted(v.offsets, issue, side="right") - 1)
    with xr.open_dataset(prepared / v.records[month]["path"]) as ds:
        raw = ds.streamflow.sel(feature_id=v.ids).isel(time=issue - v.offsets[month]).values
    expected = (np.log1p(raw) - v.mean[8]) / v.std[8]
    np.testing.assert_allclose(flow[0], expected, rtol=1e-6)


def test_graph_order_and_boundary_edges():
    from tva_flood.data.graph import build_tables

    nodes, edges = build_tables(
        pd.DataFrame({"comid": [30, 10, 20], "nwm_feature_index": [2, 0, 1]}),
        pd.DataFrame({"fromcomid": [10, 10, 20, 30], "tocomid": [20, 20, 30, 999]}),
    )
    assert nodes.comid.tolist() == [10, 20, 30]
    assert edges[["source", "target"]].values.tolist() == [[0, 1], [1, 2]]


def aggregate_inputs(tmp_path, flow=True):
    from scipy.sparse import csr_matrix, save_npz

    from tva_flood.data.aggregate import source_path

    mapping = tmp_path / "processed/aorc_mapping"
    mapping.mkdir(parents=True)
    graph = tmp_path / "processed/graph_candidate"
    graph.mkdir()
    pd.DataFrame({"node_index": [0], "comid": [10]}).to_parquet(graph / "nodes.parquet")
    save_npz(mapping / "aorc_to_nodes_weights.npz", csr_matrix([[1.0]]))
    np.savez(
        mapping / "grid_and_node_order.npz",
        node_comid=[10],
        latitude=[35.0],
        longitude=[-85.0],
        forcing_available=[True],
    )
    first = pd.Timestamp("2018-01-01")
    last = pd.Timestamp("2018-02-01")
    times = pd.date_range(first, last, freq="h", inclusive="left")
    for name in WEATHER + (["streamflow"] if flow else []):
        dims = ("time", "feature_id") if name == "streamflow" else ("time", "latitude", "longitude")
        coords = {
            "time": times,
            **(
                {"feature_id": [10]}
                if name == "streamflow"
                else {"latitude": [35.0], "longitude": [-85.0]}
            ),
        }
        values = np.arange(len(times), dtype="float32").reshape(
            (len(times),) + (1,) * (len(dims) - 1)
        )
        ds = xr.Dataset(
            {name: (dims, values, {"units": "mm" if name == "APCP_surface" else "test"})},
            coords=coords,
        )
        path = source_path(tmp_path, name, first)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path)
    boundary = source_path(tmp_path, "APCP_surface", last)
    boundary.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {
            "APCP_surface": (
                ("time", "latitude", "longitude"),
                np.array([[[1000.0]]]),
                {"units": "mm"},
            )
        },
        coords={"time": [last], "latitude": [35.0], "longitude": [-85.0]},
    ).to_netcdf(boundary)
    return first, last, times


def test_aggregate_without_streamflow_marks_flow_invalid(tmp_path):
    first, last, times = aggregate_inputs(tmp_path, flow=False)
    args = ["aggregate", "--root", tmp_path, "--start", str(first.date())]
    args += ["--stop", str(last.date()), "--workers", "1"]
    result = subprocess.run(
        [sys.executable, "-m", "tva_flood", *map(str, args)],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert result.returncode != 0  # a missing flow file is an error unless requested
    run(*args, "--no-streamflow")
    output = next((tmp_path / "processed/six_hour_nodes").rglob("*.nc"))
    with xr.open_dataset(output) as ds:
        assert (ds.streamflow_valid_hours.values == 0).all()
        assert np.isnan(ds.streamflow.values).all()
        assert ds.APCP_surface.values[0, 0] == sum(range(1, 7))
        assert (ds.APCP_surface_valid_hours.values == 6).all()
    source_path = __import__("tva_flood.data.aggregate", fromlist=["x"]).source_path
    flow = source_path(tmp_path, "streamflow", first)
    flow.parent.mkdir(parents=True, exist_ok=True)
    flow.write_bytes(b"")
    refused = subprocess.run(
        [sys.executable, "-m", "tva_flood", *map(str, args), "--no-streamflow"],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert refused.returncode != 0


def test_aggregate_subprocess_and_precipitation_boundary(tmp_path):
    first, last, times = aggregate_inputs(tmp_path)
    args = [
        "aggregate",
        "--root",
        tmp_path,
        "--start",
        str(first.date()),
        "--stop",
        str(last.date()),
        "--workers",
        "1",
    ]
    run(*args)
    output = next((tmp_path / "processed/six_hour_nodes").rglob("*.nc"))
    before = output.stat().st_mtime_ns
    with xr.open_dataset(output) as ds:
        assert ds.APCP_surface.values[0, 0] == sum(range(1, 7))
        assert ds.streamflow.values[0, 0] == 2.5
        assert ds.APCP_surface.values[-1, 0] == sum(range(len(times) - 5, len(times))) + 1000
        assert (ds.APCP_surface_valid_hours.values == 6).all()
    run(*args)
    assert output.stat().st_mtime_ns == before
