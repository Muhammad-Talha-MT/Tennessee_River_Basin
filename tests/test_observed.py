"""Observed-flow gauges, forecast-mode inputs, fine-tuning and evaluation on a tiny fixture."""

import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from test_encoder import tiny
from test_pipeline import prepared, run  # noqa: F401,F811 -- shared offline fixture

from tva_flood.data import gauges, tva
from tva_flood.data.dataset import lineage
from tva_flood.data.inputs import ForecastInputs
from tva_flood.data.ssl import FLOW, FLOW_VALID, EncoderWindows
from tva_flood.models.sparse import sparse_supports

ARCH = ["--hidden", "4", "--temporal-layers", "2", "--graph-layers", "1"]
RAIN = ["--decoder-layers", "2", "--decoder-graph-layers", "1", "--rain-hours", "24"]


def six_hour_series(prepared, name, comid):  # noqa: F811
    series = []
    for nc in sorted((prepared / "processed/six_hour_nodes").rglob("*.nc")):
        with xr.open_dataset(nc) as data:
            value = data[name].sel(feature_id=comid).to_series()
            ok = data[name + "_valid_hours"].sel(feature_id=comid).to_series() == 6
            series.append(value.where(ok))
    return pd.concat(series).astype("float64")


def tva_archive(prepared, path):  # noqa: F811
    """Synthetic TVA archive at COMID 10: observed = NWM flow; TVA forecasts = observed
    flow; TVA forecast precipitation = the realized AORC precipitation there."""
    observed = six_hour_series(prepared, "streamflow", 10)
    rain = six_hour_series(prepared, "APCP_surface", 10)
    folder = path / "build" / "TVA_headwater"
    (folder / "observed").mkdir(parents=True)
    (folder / "forecast").mkdir()
    pd.DataFrame(
        dict(
            tva_nwsid=["AAAA1"],
            camelsh_comid_match_status=["matched"],
            camelsh_outlet_comid=[10.0],
            tva_area_sqmi=[10.0],
        )
    ).to_csv(folder / tva.ATTRIBUTES, index=False)
    xr.Dataset(
        dict(river_flow=("time", observed.to_numpy())), coords=dict(time=observed.index)
    ).to_netcdf(folder / "observed/AAAA1.nc")
    issues = observed.index[::2]  # every 12 hours, as TVA sometimes issues
    leads = np.arange(0, 246, 6)
    values, precipitation = (
        np.array(
            [
                [series.get(t + pd.Timedelta(hours=int(lead) - 6), np.nan) for lead in leads]
                for t in issues
            ]
        )
        for series in (observed, rain)
    )
    xr.Dataset(
        dict(
            river_flow=(("forecast_time", "lead_time"), values),
            precipitation=(("forecast_time", "lead_time"), precipitation),
        ),
        coords=dict(forecast_time=issues, lead_time=leads),
    ).to_netcdf(folder / "forecast/AAAA1.nc")
    archive = path / "tva.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for file in folder.rglob("*"):
            bundle.write(file, file.relative_to(path / "build"))
    return archive


def catalog_file(prepared, path):  # noqa: F811
    """Two gauges: TVA's AAAA1 at COMID 10 and a CAMELS-H gauge at COMID 20."""
    flow = six_hour_series(prepared, "streamflow", 10)
    other = six_hour_series(prepared, "streamflow", 20) * 1.1
    values = np.stack([flow.to_numpy(), other.to_numpy()], axis=1).astype(np.float32)
    values[5:9, 1] = np.nan
    times = flow.index
    path.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        dict(observed=(("time", "gauge"), values)),
        coords=dict(
            time=times,
            gauge=["AAAA1", "0200000"],
            comid=("gauge", [10, 20]),
            source=("gauge", ["tva", "camelsh"]),
            tva_eval=("gauge", [True, False]),
        ),
    ).to_netcdf(path / "gauges.nc")
    pd.DataFrame(dict(gauge=["AAAA1", "0200000"], comid=[10, 20])).to_csv(
        path / "gauges.csv", index=False
    )
    return path / "gauges.nc"


def test_hourly_to_six_hour_means_need_all_six_hours():
    times = pd.date_range("2020-01-01", periods=3, freq="6h")
    hourly = pd.Series(
        np.arange(18, dtype=float), index=pd.date_range("2020-01-01", periods=18, freq="h")
    )
    hourly = hourly.drop(hourly.index[8])  # the second interval loses an hour
    result = gauges.hourly_to_six_hour(hourly, times)
    assert result[0] == pytest.approx(2.5) and np.isnan(result[1])
    assert result[2] == pytest.approx(14.5)


def test_catalog_reads_and_realigns_to_another_timeline(prepared, tmp_path):  # noqa: F811
    catalog = gauges.Catalog(catalog_file(prepared, tmp_path / "gauges"))
    assert (
        len(catalog) == 2
        and catalog.comid.tolist() == [10, 20]
        and catalog.tva_eval.tolist()
        == [
            True,
            False,
        ]
    )
    times = catalog.times[3:6].append(pd.DatetimeIndex(["1999-01-01"]))
    values = catalog.on(times)
    assert np.array_equal(values[:3], catalog.observed[3:6], equal_nan=True)
    assert np.isnan(values[3]).all()


def test_stations_use_corrected_reaches(tmp_path, monkeypatch):
    monkeypatch.setattr(tva, "REACHES", {"BBBB1": 11, "CCCC1": 12})
    pd.DataFrame(
        dict(
            tva_nwsid=["AAAA1", "BBBB1", "CCCC1", "DDDD1"],
            camelsh_comid_match_status=["matched", np.nan, "matched", np.nan],
            camelsh_outlet_comid=[10.0, np.nan, 99.0, np.nan],
        )
    ).to_csv(tmp_path / tva.ATTRIBUTES, index=False)
    for kind in ("observed", "forecast"):
        (tmp_path / kind).mkdir()
        for station in ("AAAA1", "BBBB1", "CCCC1", "DDDD1"):
            xr.Dataset(dict(river_flow=("time", [1.0, 2.0]))).to_netcdf(
                tmp_path / kind / f"{station}.nc"
            )
    table = tva.gauges(tmp_path, np.array([10, 11, 12, 99]))
    assert table.tva_nwsid.tolist() == ["AAAA1", "BBBB1", "CCCC1"]
    assert table.comid.tolist() == [10, 11, 12]
    assert table.comid_source.tolist() == ["archive", "location_and_area", "location_and_area"]
    (tmp_path / "forecast" / "BBBB1.nc").unlink()
    assert tva.gauges(tmp_path, np.array([10, 11, 12]), forecasts=False).tva_nwsid.tolist() == [
        "AAAA1",
        "BBBB1",
        "CCCC1",
    ]


def test_basin_rain_applies_each_source_only_inside_its_basin_and_window():
    # 0 -> 1 -> 2 and 3 -> 2: gauges at 1 (basin {0, 1}) and 2 (basin {0, 1, 2, 3}).
    edges = np.array([[0, 1], [1, 2], [3, 2]])
    owner, sizes = tva.basin_owner(edges, 5, [1, 2])
    assert owner.tolist() == [0, 0, 1, 1, -1] and sizes.tolist() == [2, 4]
    value, flag = torch.full((40, 5), 0.7), torch.ones(40, 5, dtype=torch.bool)
    forecast = np.full((40, 2), 3.0)
    forecast[:, 1] = 8.0
    forecast[2, 0] = np.nan
    rain = ("log1p", 0.5, 2.0)
    owners = torch.as_tensor(owner)
    both = np.array([True, True])
    qpf = tva.basin_rain(
        value, flag, torch.zeros(5, dtype=torch.bool), forecast, owners, both, 10, rain
    )
    assert qpf.shape == (40, 5, 2)
    assert not qpf[10:].any() and not qpf[:, 4].any()
    np.testing.assert_allclose(qpf[0, 0, 0], (np.log1p(3.0) - 0.5) / 2.0, rtol=1e-6)
    np.testing.assert_allclose(qpf[0, 3, 0], (np.log1p(8.0) - 0.5) / 2.0, rtol=1e-6)
    assert qpf[2, 0, 1] == 0 and qpf[2, 2, 1] == 1  # a missing forecast stays unknown
    # AORC fills reaches without a forecast; a forecast takes precedence where both exist.
    known = torch.tensor([True, True, True, True, False])
    mixed = tva.basin_rain(value, flag, known, forecast, owners, np.array([True, False]), 10, rain)
    np.testing.assert_allclose(mixed[0, 0, 0], (np.log1p(3.0) - 0.5) / 2.0, rtol=1e-6)
    assert torch.all(mixed[:10, 2:4, 0] == 0.7) and torch.all(mixed[:10, 2:4, 1] == 1)
    assert not mixed[:, 4].any()


def test_forecast_inputs_show_flow_only_at_gauges_and_rain_by_spec(prepared, tmp_path):  # noqa: F811
    catalog = gauges.Catalog(catalog_file(prepared, tmp_path / "gauges"))
    archive = tva_archive(prepared, tmp_path)
    prep = prepared / "processed/pretraining_full"
    data = EncoderWindows(prepared, prep, "validation", 40, future_rain=True)
    with np.load(prep / "graph.npz") as graph:
        edges = np.column_stack([graph["source"], graph["target"]])
    folder = tva.extract(archive, tmp_path / "extracted")
    inputs = ForecastInputs(data, catalog, folder, edges, sparse_supports(3, edges), halo=1)
    assert inputs.station_gauge.tolist() == [0] and inputs.is_tva.tolist() == [True, False]
    graph = inputs.graph([0, 1])
    assert graph["nodes"].tolist() == [0, 1, 2] and graph["local"].tolist() == [0, 1]
    item = int(np.flatnonzero([int(i) in inputs.qpf for i in data.issues])[10])
    sample = inputs.load(data, item, graph)
    flow = sample["dynamic"][..., FLOW]
    assert flow[:, 2].eq(0).all() and sample["dynamic"][:, 2, FLOW_VALID].eq(0).all()
    assert sample["dynamic"][:, 0, FLOW_VALID].eq(1).all()
    issue = sample["issue"]
    expected = (np.log1p(inputs.observed[issue - 120 : issue, 0]) - inputs.mean) / inputs.std
    np.testing.assert_allclose(flow[:, 0].numpy(), expected, rtol=1e-5)
    hidden = inputs.load(data, item, graph, hidden=np.array([True, False]))
    assert (
        hidden["dynamic"][:, 0, FLOW_VALID].eq(0).all()
        and hidden["dynamic"][:, 1, FLOW_VALID].eq(1).any()
    )
    steps = 4
    nothing = inputs.future_rain(sample, graph, None, steps)
    assert not nothing.any()
    # TVA's forecast here equals the realized rain at COMID 10, so qpf == aorc on that basin.
    qpf = inputs.future_rain(sample, graph, "qpf", steps)
    aorc = inputs.future_rain(sample, graph, "aorc", steps)
    assert qpf[:steps, 0].any() and not qpf[steps:].any() and not qpf[:, 1:].any()
    torch.testing.assert_close(qpf[:steps, 0], aorc[:steps, 0], rtol=1e-4, atol=1e-5)
    # A gauge mask adds realized rain over those basins (reach 20's gauge has no forecast).
    proxy = inputs.future_rain(sample, graph, np.array([False, True]), steps)
    assert proxy[:steps, 1].any() and not proxy[steps:].any()
    torch.testing.assert_close(proxy[:steps, 0], qpf[:steps, 0], rtol=1e-4, atol=1e-5)


def test_rain_error_flood_selection_and_weight_average(prepared, tmp_path):  # noqa: F811
    catalog = catalog_file(prepared, tmp_path / "gauges")
    archive = tva_archive(prepared, tmp_path)
    run(
        *tiny("train-forecast", prepared, tmp_path / "stage1", *ARCH, *RAIN),
        "--history-gauges",
        catalog,
        "--rain-error",
    )
    assert json.loads((tmp_path / "stage1/run_config.json").read_text())["rain_error"]
    command = [
        "finetune-observed",
        "--root",
        prepared,
        "--device",
        "cpu",
        "--gauges",
        catalog,
        "--tva-archive",
        archive,
        "--init",
        tmp_path / "stage1/best.pt",
        "--out",
        tmp_path / "tuned",
        "--updates",
        "4",
        "--eval-every",
        "2",
        "--batch-size",
        "2",
        "--gauges-per-update",
        "2",
        "--validation-issues",
        "0",
        "--min-count",
        "2",
        "--warmup-updates",
        "0",
        "--rain-error",
        "--under-weight",
        "2",
        "--ema-decay",
        "0.5",
        "--csi-weight",
        "1",
        "--flood-quantile",
        "0.9",
        "--flood-fraction",
        "0.5",
        "--select-from",
        "2018-03-20",
    ]
    run(*command)
    progress = pd.read_csv(tmp_path / "tuned/validation_progress.csv")
    combined = progress.nse_selection + progress.csi_selection.fillna(0)
    assert np.allclose(progress.selection, combined)
    assert {"tva_csi_selection", "model_csi_24h", "tva_csi_24h"} <= set(progress.columns)
    last = torch.load(tmp_path / "tuned/last.pt", weights_only=False)
    saved = torch.load(tmp_path / "tuned/checkpoints/update_0000004.pt", weights_only=False)
    # Checkpoints hold the moving average; last.pt the raw weights and the average.
    for key, value in last["average"].items():
        assert torch.equal(saved["model"][key], value)
    assert any(not torch.equal(last["model"][k], v) for k, v in last["average"].items())
    settings = last["config"]["observed_finetune"]
    assert settings["rain_error"] != "none" and "moving average" in settings["weights"]
    assert settings["flood_sampling"] != "uniform"
    assert settings["validation_period"].startswith("2018-03-20")
    run(*command)  # a completed run resumes and stops without changes
    assert len(pd.read_csv(tmp_path / "tuned/validation_progress.csv")) == 3


def test_forecast_mode_pipeline_trains_finetunes_and_evaluates(prepared, tmp_path):  # noqa: F811
    catalog = catalog_file(prepared, tmp_path / "gauges")
    archive = tva_archive(prepared, tmp_path)
    run(
        *tiny("train-forecast", prepared, tmp_path / "stage1", *ARCH, *RAIN),
        "--history-gauges",
        catalog,
    )
    config = json.loads((tmp_path / "stage1/run_config.json").read_text())
    assert config["history_gauge_comids"] == [10, 20] and config["rain_hours"] == 24
    assert config["decoder"]["future_channels"] == 2 and config["subgraph_halo_hops"] == 2
    best = tmp_path / "stage1/best.pt"
    selected = torch.load(best, weights_only=False)["metrics"]["selection"]
    run(*tiny("train-forecast", prepared, tmp_path / "rescored", "--evaluate-checkpoint", best))
    rescored = json.loads(
        (tmp_path / "rescored/evaluations/update_0000000/metrics.json").read_text()
    )
    assert rescored["selection"] == pytest.approx(selected, rel=1e-12)
    run(
        "evaluate-hydrology",
        "--root",
        prepared,
        "--device",
        "cpu",
        "--min-count",
        "2",
        "--split",
        "validation",
        "--checkpoint",
        best,
        "--out",
        tmp_path / "hydro",
        "--focus-comids",
        "10",
    )
    assert json.loads((tmp_path / "hydro/evaluation_manifest.json").read_text())["history_gauges"]

    command = [
        "finetune-observed",
        "--root",
        prepared,
        "--device",
        "cpu",
        "--gauges",
        catalog,
        "--tva-archive",
        archive,
        "--init",
        best,
        "--out",
        tmp_path / "tuned",
        "--updates",
        "4",
        "--eval-every",
        "2",
        "--batch-size",
        "2",
        "--gauges-per-update",
        "2",
        "--validation-issues",
        "8",
        "--min-count",
        "2",
        "--log-every",
        "1",
        "--warmup-updates",
        "0",
    ]
    run(*command)
    progress = pd.read_csv(tmp_path / "tuned/validation_progress.csv")
    assert progress["update"].tolist() == [0, 2, 4]
    assert np.isfinite(progress.selection).all() and np.isfinite(progress.tva_selection).all()
    tuned = torch.load(tmp_path / "tuned/best.pt", weights_only=False)
    assert tuned["update"] == progress.loc[progress.selection.idxmax(), "update"]
    assert tuned["config"]["history_gauge_comids"] == [10, 20]
    assert tuned["config"]["observed_finetune"]["tva_evaluation_gauges"] == 1
    log = [json.loads(line) for line in (tmp_path / "tuned/training_log.jsonl").open()]
    assert [row["update"] for row in log] == [1, 2, 3, 4]
    run(*command)  # a completed run resumes and stops without changes
    assert len(pd.read_csv(tmp_path / "tuned/validation_progress.csv")) == 3
    changed = [c if c != "4" else "5" for c in map(str, command)]
    result = subprocess.run(
        [sys.executable, "-m", "tva_flood", *changed],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert result.returncode != 0 and "different arguments" in result.stderr

    run(
        "evaluate-observed",
        "--root",
        prepared,
        "--device",
        "cpu",
        "--min-count",
        "2",
        "--split",
        "validation",
        "--gauges",
        catalog,
        "--tva-archive",
        archive,
        "--checkpoint",
        f"tuned={tmp_path / 'tuned/best.pt'}",
        "--out",
        tmp_path / "observed",
    )
    stations = pd.read_csv(tmp_path / "observed/per_station.csv")
    assert {"tuned+qpf", "tuned+aorc", "tuned+unknown", "persistence", "tva_forecast"} <= set(
        stations.method
    )
    assert "nwm_retrospective" in set(stations.method)
    by = stations.set_index(["method", "lead_hours"])
    # TVA's forecast rain equals the realized rain here, so both variants agree.
    for metric in ("nse", "rmse"):
        np.testing.assert_allclose(
            by.loc["tuned+qpf", metric], by.loc["tuned+aorc", metric], rtol=1e-4
        )
    manifest = json.loads((tmp_path / "observed/evaluation_manifest.json").read_text())
    assert (
        manifest["split"] == "validation" and manifest["checkpoints"]["tuned"]["rain_hours"] == 24
    )


def test_test_preparation_reuses_normalization_and_scores_checkpoints(
    prepared,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    from tva_flood.data import prepare_test

    monkeypatch.setattr(prepare_test, "FIRST_TEST", pd.Timestamp("2018-12-01"))
    source = prepared / "processed/pretraining_full"
    test = tmp_path / "test_preparation"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare-test", "--root", str(prepared), "--out", str(test)]
        + ["--start", "2018-12-01", "--stop", "2019-02-01"],
    )
    prepare_test.main()
    manifest = json.loads((test / "dataset_manifest.json").read_text())
    parent = json.loads((source / "dataset_manifest.json").read_text())
    assert manifest["parent_identity"] == parent["identity"] != manifest["identity"]
    assert lineage(manifest) == parent["identity"]
    assert list(manifest["splits"]) == ["test"] and manifest["months"][0]["start"][:7] == "2018-11"
    for name in ("dynamic_normalization.json", "static_normalization.json", "graph.npz"):
        assert (test / name).read_bytes() == (source / name).read_bytes()
    data = EncoderWindows(prepared, test, "test", 40)
    assert len(data) == manifest["test_windows"] > 0
    assert data.times[data.issues[0]] >= pd.Timestamp("2018-12-01")
    with pytest.raises(ValueError, match="Split must be"):
        EncoderWindows(prepared, test, "validation", 40)
    cache = tmp_path / "test_arrays"
    run("build-array-cache", "--root", prepared, "--preparation", test, "--out", cache)
    mapped = EncoderWindows(prepared, test, "test", 40, array_cache=cache)
    assert torch.equal(mapped[0]["dynamic"], data[0]["dynamic"])
    catalog = catalog_file(prepared, tmp_path / "gauges")
    archive = tva_archive(prepared, tmp_path)
    # A checkpoint trained on the source preparation scores on the test preparation.
    run(
        *tiny("train-forecast", prepared, tmp_path / "stage1", *ARCH, *RAIN),
        "--history-gauges",
        catalog,
    )
    run(
        "evaluate-observed",
        "--root",
        prepared,
        "--preparation",
        test,
        "--array-cache",
        cache,
        "--split",
        "test",
        "--issues-from",
        "2019-01-01",
        "--device",
        "cpu",
        "--min-count",
        "2",
        "--gauges",
        catalog,
        "--tva-archive",
        archive,
        "--checkpoint",
        f"rain={tmp_path / 'stage1/best.pt'}",
        "--out",
        tmp_path / "observed",
    )
    report = json.loads((tmp_path / "observed/evaluation_manifest.json").read_text())
    assert report["split"] == "test" and report["first_issue"] >= "2019-01-01"
    assert "test, issues 2019-01-" in (tmp_path / "observed/report.md").read_text()
