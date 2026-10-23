"""Exact patches, new masks, array cache identity and resumable encoder training."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from test_pipeline import prepared, run  # noqa: F401,F811 -- shared offline fixture

from tva_flood.data.dataset import CachedWindows
from tva_flood.data.ssl import EncoderWindows, causal_persistence, mask_history
from tva_flood.data.subgraph import SubgraphSampler, expand, undirected_adjacency
from tva_flood.models.sparse import sparse_supports
from tva_flood.models.ssl import ForecastModel, SSLReconstructor, TemporalGraphEncoder


def river(nodes=80, seed=0):
    """Random tree: every reach drains to one lower-numbered reach."""
    rng = np.random.default_rng(seed)
    return np.array([[child, int(rng.integers(0, child))] for child in range(1, nodes)])


def test_patch_core_outputs_equal_full_graph_outputs():
    torch.manual_seed(0)
    edges = river()
    supports = sparse_supports(80, edges)
    encoder = TemporalGraphEncoder(supports, hidden=8, temporal_layers=2, graph_layers=3).eval()
    dynamic, static = torch.randn(2, 12, 80, 18), torch.randn(80, 48)
    with torch.no_grad():
        full = encoder(dynamic, static)
        torch.testing.assert_close(
            encoder(dynamic, static, last_only=True)[:, 0], full[:, -1], rtol=0, atol=0
        )
        sampler = SubgraphSampler(80, edges, supports, halo=3)
        for seed in range(5):
            graph = sampler.sample(np.random.default_rng(seed), 15)
            nodes, core = graph["nodes"], graph["core"]
            assert core.sum() == 15 and len(nodes) > 15
            local = encoder(dynamic[:, :, nodes], static[nodes], graph["supports"])
            torch.testing.assert_close(local[:, :, core], full[:, :, nodes[core]])
        # Too small a halo is detectably inexact: the test can fail.
        short = SubgraphSampler(80, edges, supports, halo=1).sample(np.random.default_rng(1), 15)
        local = encoder(dynamic[:, :, short["nodes"]], static[short["nodes"]], short["supports"])
        assert not torch.allclose(
            local[:, :, short["core"]], full[:, :, short["nodes"][short["core"]]]
        )


def test_patches_are_connected_and_exactly_sized():
    edges = river(200, seed=3)
    sampler = SubgraphSampler(200, edges, sparse_supports(200, edges), halo=2)
    core = sampler.patch(np.random.default_rng(4), 37)
    assert core.sum() == 37
    adjacency = undirected_adjacency(200, edges)[core][:, core]
    reached = np.zeros(37, dtype=bool)
    reached[0] = True
    assert expand(adjacency, reached, 37).all()


def series(nodes=6, steps=30):
    dynamic = torch.randn(2, steps, nodes, 18)
    valid = torch.ones(2, steps, nodes, dtype=torch.bool)
    valid[:, 2:5, 1] = False
    dynamic[..., 17] = valid.float()
    return dynamic, valid


@pytest.mark.parametrize("mode", ["tail", "patch"])
def test_new_masks_are_repeatable_and_cannot_leak(mode):
    dynamic, valid = series()
    adjacency = undirected_adjacency(6, river(6))
    options = dict(mode=mode, adjacency=adjacency, tail_min=3, tail_max=8, patch_hops=(1, 2))
    masked, selected = mask_history(dynamic, valid, seed=3, **options)
    again, selected_again = mask_history(dynamic, valid, seed=3, **options)
    assert torch.equal(masked, again) and torch.equal(selected, selected_again)
    assert not (selected & ~valid).any() and selected.flatten(1).any(dim=1).all()
    assert torch.count_nonzero(masked[..., 16:18][selected | ~valid]) == 0
    assert torch.equal(masked[..., :16], dynamic[..., :16])
    perturbed = dynamic.clone()
    perturbed[..., 16][selected] = 999
    assert torch.equal(mask_history(perturbed, valid, seed=3, **options)[0], masked)
    if mode == "tail":
        steps = selected.any(dim=2)
        for row in steps:
            first = int(row.nonzero()[0, 0])
            assert row[first:].all() and not row[:first].any() and 3 <= 30 - first <= 8
    else:
        assert torch.equal(selected, selected[:, :1].expand_as(selected) & valid)


def test_tail_can_hide_weather_only_inside_the_tail():
    dynamic, valid = series()
    masked, selected = mask_history(
        dynamic, valid, seed=1, mode="tail", tail_min=5, tail_max=5, hide_weather=True
    )
    assert not masked[:, -5:, :, :18].any()
    torch.testing.assert_close(masked[:, :-5, :, :16], dynamic[:, :-5, :, :16])


def test_causal_persistence_uses_only_earlier_visible_values():
    flow = torch.tensor([[[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]]])
    visible = torch.tensor([[[True, False], [False, False], [True, False], [False, False]]])
    result = causal_persistence(flow, visible)
    assert result[0, :, 0].tolist() == [1.0, 1.0, 3.0, 3.0]
    assert result[0, :, 1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_array_cache_and_encoder_inputs_match_monthly_arrays(prepared, tmp_path):  # noqa: F811
    prep = prepared / "processed/pretraining_full"
    cache = tmp_path / "arrays"
    run("build-array-cache", "--root", prepared, "--out", cache, "--workers", "2")
    run("build-array-cache", "--root", prepared, "--out", cache, "--workers", "2")
    original = CachedWindows(prepared, prep, "validation")
    for horizon in (0, 40):
        plain = EncoderWindows(prepared, prep, "validation", horizon)
        mapped = EncoderWindows(prepared, prep, "validation", horizon, array_cache=cache)
        for item in range(len(plain)):
            a, b = plain[item], mapped[item]
            assert a.keys() == b.keys()
            for key in a:
                if key != "issue":
                    assert torch.equal(a[key], b[key]), key
            subset = mapped.load(item, [2, 0])
            for key in a:
                if key != "issue":
                    assert torch.equal(subset[key], a[key][:, [2, 0]]), key
    forecast = EncoderWindows(prepared, prep, "validation", 40)
    for item in (0, len(forecast) - 1):
        issue, new = int(forecast.issues[item]), forecast[item]
        x, xm, q, qm = original.window(issue - 120, issue + 40)
        # Monthly flow is already zero wherever it is invalid.
        dynamic = np.concatenate([x, xm, q[..., None], qm[..., None]], -1)[:120]
        assert torch.equal(torch.as_tensor(dynamic, dtype=torch.float32), new["dynamic"])
        assert torch.equal(torch.as_tensor(q[120:]), new["target"])
        assert torch.equal(torch.as_tensor(qm[120:]), new["target_valid"])
    meta = json.loads((cache / "array_cache.json").read_text())
    meta["months"][0]["bytes"] += 1
    (cache / "array_cache.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="different prepared data"):
        EncoderWindows(prepared, prep, "validation", 0, array_cache=cache)


def tiny(task, root, out, *extra):
    return [
        task,
        "--root",
        root,
        "--out",
        out,
        "--device",
        "cpu",
        "--updates",
        "6",
        "--eval-every",
        "3",
        "--batch-size",
        "2",
        "--subgraph-nodes",
        "2",
        "--validation-windows",
        "3",
        "--warmup-updates",
        "0",
        "--log-every",
        "1",
        *extra,
    ]


def test_ssl_resume_is_exact_and_transfers_to_forecasting(prepared, tmp_path):  # noqa: F811
    arch = ["--hidden", "4", "--temporal-layers", "2", "--graph-layers", "1", "--dropout", "0.1"]
    run(*tiny("ssl-pretrain", prepared, tmp_path / "continuous", *arch))
    for _ in range(3):
        run(*tiny("ssl-pretrain", prepared, tmp_path / "resumed", *arch, "--max-steps", "2"))
    finished = [
        torch.load(tmp_path / n / "last.pt", weights_only=False) for n in ("continuous", "resumed")
    ]
    for key, value in finished[0]["model"].items():
        assert torch.equal(value, finished[1]["model"][key]), key
    assert finished[0]["progress"] == finished[1]["progress"]
    assert finished[0]["progress"]["completed"] and finished[0]["progress"]["update"] == 6
    progress = pd.read_csv(tmp_path / "continuous/validation_progress.csv")
    assert progress["update"].tolist() == [0, 3, 6]
    assert sorted(p.name for p in (tmp_path / "continuous/checkpoints").iterdir()) == [
        "update_0000000.pt",
        "update_0000003.pt",
        "update_0000006.pt",
    ]
    kinds = pd.read_csv(tmp_path / "continuous/evaluations/update_0000006/by_kind.csv")
    assert set(kinds.kind) == {"block", "tail", "tail_noweather", "reach", "patch", "all"}
    log = [json.loads(line) for line in open(tmp_path / "continuous/training_log.jsonl")]
    assert all(row["nodes_per_patch"] == 3 for row in log)  # 2 core + 1 halo reach

    ssl = tmp_path / "continuous/best.pt"
    run(*tiny("train-forecast", prepared, tmp_path / "forecast_ssl", "--init", ssl))
    run(
        *tiny("train-forecast", prepared, tmp_path / "forecast_scratch", "--architecture-from", ssl)
    )
    source = torch.load(ssl, weights_only=False)["model"]
    initial = torch.load(
        tmp_path / "forecast_ssl/checkpoints/update_0000000.pt", weights_only=False
    )
    scratch = torch.load(
        tmp_path / "forecast_scratch/checkpoints/update_0000000.pt", weights_only=False
    )
    assert initial["config"]["architecture"] == scratch["config"]["architecture"]
    encoder_keys = [k for k in source if k.startswith("encoder.")]
    assert all(torch.equal(initial["model"][k], source[k]) for k in encoder_keys)
    assert not all(torch.equal(scratch["model"][k], source[k]) for k in encoder_keys)
    summary = pd.read_csv(tmp_path / "forecast_ssl/evaluations/update_0000006/summary.csv")
    assert set(summary.scenario) == {"full", "none"}
    # At initialization the visible-history forecast is exactly persistence.
    start = pd.read_csv(tmp_path / "forecast_ssl/evaluations/update_0000000/summary.csv")
    model = start[(start.scenario == "full") & (start.group == "model_matched")]
    base = start[(start.scenario == "full") & (start.group == "persistence_matched")]
    np.testing.assert_allclose(model.normalized_log_mse, base.normalized_log_mse)
    best = tmp_path / "forecast_ssl/best.pt"
    run(*tiny("train-forecast", prepared, tmp_path / "rescored", "--evaluate-checkpoint", best))
    rescored = json.loads(
        (tmp_path / "rescored/evaluations/update_0000000/metrics.json").read_text()
    )
    selected = torch.load(best, weights_only=False)["metrics"]["selection"]
    assert rescored["selection"] == pytest.approx(selected, rel=1e-12)
    assert not (tmp_path / "rescored/last.pt").exists()
    bad = tiny("train-forecast", prepared, tmp_path / "bad", "--init", ssl, "--hidden", "8")
    result = subprocess.run(
        [sys.executable, "-m", "tva_flood", *map(str, bad)], capture_output=True, text=True
    )
    assert result.returncode != 0 and "inherited architecture" in result.stderr


def test_forecast_and_reconstruction_accept_patch_supports():
    edges = river(20)
    supports = sparse_supports(20, edges)
    sampler = SubgraphSampler(20, edges, supports, halo=2)
    graph = sampler.sample(np.random.default_rng(0), 5)
    nodes = len(graph["nodes"])
    encoder = TemporalGraphEncoder(supports, hidden=4, temporal_layers=1, graph_layers=2)
    dynamic, static = torch.randn(1, 8, nodes, 18), torch.randn(nodes, 48)
    assert SSLReconstructor(encoder)(dynamic, static, graph["supports"]).shape == (1, 8, nodes)
    assert ForecastModel(encoder, 40)(dynamic, static, graph["supports"]).shape == (1, 40, nodes)


def test_hydrologic_metrics_match_direct_series_computation():
    from tva_flood.evaluation.hydrology import Accumulator, merge_peaks, reach_metrics

    rng = np.random.default_rng(0)
    target = torch.as_tensor(rng.gamma(2.0, 5.0, size=(30, 40, 4)))
    forecast = target * torch.as_tensor(rng.uniform(0.7, 1.3, size=target.shape)) + 1
    mask = torch.ones(target.shape, dtype=torch.bool)
    mask[3:7, :, 2] = False
    whole, halves = Accumulator(4, "cpu"), [Accumulator(4, "cpu"), Accumulator(4, "cpu")]
    for step in range(30):
        whole.add(forecast[step], target[step], mask[step])
        halves[step % 2].add(forecast[step], target[step], mask[step])
    peak, at_peak = merge_peaks(
        [h.target_peak for h in halves], [h.forecast_at_target_peak for h in halves]
    )
    assert torch.equal(peak, whole.target_peak)
    assert torch.equal(at_peak, whole.forecast_at_target_peak)
    torch.testing.assert_close(halves[0].sums + halves[1].sums, whole.sums)
    result = reach_metrics(**whole.numpy(), min_count=10)
    for lead, reach in ((0, 0), (39, 2)):
        keep = mask[:, lead, reach].numpy()
        s, o = forecast[keep, lead, reach].numpy(), target[keep, lead, reach].numpy()
        nse = 1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
        r = np.corrcoef(s, o)[0, 1]
        kge = 1 - np.sqrt(
            (r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2
        )
        np.testing.assert_allclose(result["nse"][lead, reach], nse, rtol=1e-10)
        np.testing.assert_allclose(result["kge"][lead, reach], kge, rtol=1e-10)
        np.testing.assert_allclose(result["rmse"][lead, reach], np.sqrt(((s - o) ** 2).mean()))
        np.testing.assert_allclose(result["peak_error"][lead, reach], (s.max() - o.max()) / o.max())
        at = s[o.argmax()]
        np.testing.assert_allclose(
            result["peak_error_at_target_time"][lead, reach], (at - o.max()) / o.max()
        )
    constant = reach_metrics(
        np.array([[50.0, 50.0, 50.0, 50.0, 50.0, 50.0]]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0]),
        min_count=10,
    )
    assert not constant["scored"][0] and np.isnan(constant["nse"][0])


def test_hydrology_command_on_fixture(prepared, tmp_path):  # noqa: F811
    arch = ["--hidden", "4", "--temporal-layers", "2", "--graph-layers", "1"]
    run(*tiny("train-forecast", prepared, tmp_path / "forecast", *arch))
    run(
        "evaluate-hydrology",
        "--root",
        prepared,
        "--device",
        "cpu",
        "--min-count",
        "2",
        "--checkpoint",
        tmp_path / "forecast/best.pt",
        "--out",
        tmp_path / "hydro",
        "--focus-comids",
        "10",
    )
    summary = pd.read_csv(tmp_path / "hydro/summary.csv")
    assert set(zip(summary.scenario, summary.method)) == {
        ("full", "forecast"),
        ("full", "persistence"),
        ("none", "forecast"),
    }
    assert set(summary.lead_hours) == {6, 24, 72, 120, 240}
    manifest = json.loads((tmp_path / "hydro/evaluation_manifest.json").read_text())
    assert manifest["all_windows"] and manifest["optimizer_steps"] == 0
    assert (tmp_path / "hydro/per_reach_lead_240h.parquet").exists()
    assert (tmp_path / "hydro/report.md").read_text().startswith("# Hydrologic evaluation")


def test_weather_decoder_starts_at_persistence_is_lead_causal_and_patch_exact():
    from tva_flood.models.ssl import WeatherForecastModel

    torch.manual_seed(5)
    edges = river(60, seed=2)
    supports = sparse_supports(60, edges)
    encoder = TemporalGraphEncoder(supports, hidden=8, temporal_layers=2, graph_layers=2)
    model = WeatherForecastModel(
        encoder, 40, decoder_layers=3, decoder_graph_layers=2, future_channels=2
    ).eval()
    dynamic, static = torch.randn(2, 10, 60, 18), torch.randn(60, 48)
    dynamic[..., 17] = 1
    dynamic[:, :, 7, 16:18] = 0
    future = torch.randn(2, 40, 60, 2)
    with torch.no_grad():
        result = model(dynamic, static, future)
        expected = dynamic[:, -1, :, 16][:, None].expand(-1, 40, -1)
        keep = torch.ones(60, dtype=torch.bool)
        keep[7] = False
        torch.testing.assert_close(result[..., keep], expected[..., keep], rtol=0, atol=0)
        # A change in lead 20 weather cannot alter any earlier lead.
        model.correction.weight.data.normal_()
        before = model(dynamic, static, future)
        changed = future.clone()
        changed[:, 20:] += 5
        after = model(dynamic, static, changed)
        torch.testing.assert_close(after[:, :20], before[:, :20], rtol=0, atol=0)
        assert not torch.equal(after[:, 20:], before[:, 20:])
        sampler = SubgraphSampler(60, edges, supports, halo=4)
        graph = sampler.sample(np.random.default_rng(0), 12)
        nodes, core = graph["nodes"], graph["core"]
        local = model(dynamic[:, :, nodes], static[nodes], future[:, :, nodes], graph["supports"])
        torch.testing.assert_close(local[..., core], before[..., nodes[core]])


def test_future_rain_is_the_realized_precipitation_after_issue(prepared):  # noqa: F811
    prep = prepared / "processed/pretraining_full"
    data = EncoderWindows(prepared, prep, "validation", 40, future_rain=True)
    item = data.load(0)
    weather, weather_valid, _, _ = data.node_window(item["issue"], item["issue"] + 40)
    expected = np.concatenate([weather[..., :1], weather_valid[..., :1].astype(np.float32)], -1)
    assert torch.equal(item["future"], torch.from_numpy(expected))
    with pytest.raises(ValueError, match="forecast windows"):
        EncoderWindows(prepared, prep, "validation", 0, future_rain=True)


def test_future_masking_hides_coverage_deterministically():
    from tva_flood.data.ssl import mask_future

    future = torch.rand(40, 30, 2) + 0.5
    adjacency = undirected_adjacency(30, river(30))
    nothing = mask_future(future, seed=1, adjacency=adjacency, none_prob=1.0, region_prob=0.0)
    assert not nothing.any()
    everything = mask_future(
        future, seed=1, adjacency=adjacency, none_prob=0.0, region_prob=0.0, cutoff_prob=0.0
    )
    assert torch.equal(everything, future)
    cut = mask_future(
        future,
        seed=3,
        adjacency=adjacency,
        none_prob=0.0,
        region_prob=0.0,
        cutoff_prob=1.0,
        cutoff_steps=(5, 5),
    )
    assert torch.equal(cut[:5], future[:5]) and not cut[5:].any()
    options = dict(
        adjacency=adjacency,
        none_prob=0.0,
        region_prob=1.0,
        cutoff_prob=0.0,
        region_fraction=(0.1, 0.2),
        region_hops=(1, 2),
    )
    regional = mask_future(future, seed=4, **options)
    kept = regional.any(dim=(0, 2))
    assert kept.any() and (~kept).any()
    assert torch.equal(regional[:, kept], future[:, kept])
    again = mask_future(future * 3, seed=4, **options)
    assert torch.equal(again.any(dim=(0, 2)), kept)
