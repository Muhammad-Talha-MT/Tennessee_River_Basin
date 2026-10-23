"""Whole-basin routing: operator rows, closed sampling, and exact patches for the model."""

import numpy as np
import pytest
import torch

from tva_flood.data import river as river_module
from tva_flood.data.river import River, build_operator, graph_key
from tva_flood.data.subgraph import SubgraphSampler
from tva_flood.models.sparse import sparse_supports
from tva_flood.models.ssl import TemporalGraphEncoder, WeatherForecastModel


def random_river(seed=0, n=60):
    """A random network of several outlets, with one divergence that rejoins."""
    rng = np.random.default_rng(seed)
    edges = [(i, int(rng.integers(i + 1, min(n, i + 6)))) for i in range(n - 1) if i % 17 != 16]
    edges.append((3, 9))  # divergence from 3, rejoining downstream
    edges = np.unique(np.array(edges), axis=0)
    area = rng.uniform(0.0, 3.0, n)
    area[::7] = 0.0
    length = rng.uniform(0.2, 4.0, n)
    return n, edges, area, length


@pytest.fixture
def network(tmp_path):
    n, edges, area, length = random_river()
    ids = np.arange(100, 100 + n)
    path = tmp_path / "routing.npz"
    np.savez(
        path, graph_key=np.array(graph_key(ids, edges)), **build_operator(n, edges, area, length)
    )
    return n, edges, area, length, River(path, ids, edges)


def brute_upstream(n, edges, i):
    """Share of each reach's water reaching i, by summing over every path."""
    out = np.bincount(edges[:, 0], minlength=n)
    share = np.zeros(n)
    share[i] = 1.0
    for node in range(i - 1, -1, -1):  # edges point to larger indices
        for s, t in edges:
            if s == node:
                share[node] += share[t] / out[node]
    return share


def test_rows_are_area_weighted_basin_means(network):
    n, edges, area, length, river = network
    clipped = np.maximum(area, river_module.MIN_AREA_KM2)
    for i in (0, 9, 25, n - 1):
        share = brute_upstream(n, edges, i)
        row = slice(river.indptr[i], river.indptr[i + 1])
        weight = np.zeros(n)
        weight[river.indices[row]] = river.weight[row]
        expected = clipped * share / (clipped * share).sum()
        assert np.allclose(weight, expected, atol=1e-6)
        assert np.isclose(river.basin_area[i], (clipped * share).sum())
        assert (river.distance[row] >= 0).all()
        self = river.indices[row] == i
        assert np.isclose(river.distance[row][self][0], length[i] / 2, atol=1e-5)


def test_sampled_basins_are_closed(network):
    river = network[-1]
    rng = np.random.default_rng(1)
    for _ in range(5):
        chosen = river.sample_basins(rng, 25)
        assert chosen.sum() <= 25 and (river.closure(chosen) == chosen).all()
    with pytest.raises(ValueError, match="closed"):
        river.operator(np.array([40]))  # 40 has upstream reaches


def test_closed_patch_reproduces_the_full_network(network):
    n, edges, _, _, river = network
    torch.manual_seed(0)
    supports = sparse_supports(n, edges)
    encoder = TemporalGraphEncoder(
        supports,
        hidden=8,
        temporal_layers=2,
        graph_layers=2,
        direction="upstream",
        aggregation="attention",
        basin_inputs=True,
    )
    model = WeatherForecastModel(
        encoder,
        6,
        decoder_layers=2,
        decoder_graph_layers=1,
        future_channels=2,
        routing_bins=len(river_module.DISTANCE_BINS) - 1,
        routing_lags=3,
        basin_rain=True,
    ).eval()
    for p in model.parameters():  # zero-initialized heads would hide routing errors
        torch.nn.init.normal_(p, std=0.3) if p.dim() > 1 else None
    encoder.full_routing = river.operator(np.arange(n))
    dynamic = torch.randn(2, 10, n, 18)
    dynamic[..., 17] = (torch.rand(2, 10, n) > 0.5).float()
    static = torch.randn(n, 48)
    future = torch.randn(2, 6, n, 2)
    with torch.no_grad():
        full = model(dynamic, static, future)
        sampler = SubgraphSampler(n, edges, supports, 0, river)
        outlet = np.zeros(n, dtype=bool)
        outlet[[30, 31]] = True
        nodes = np.flatnonzero(sampler.closed(outlet))
        assert len(nodes) < n
        graph = sampler.subgraph(nodes, np.ones(len(nodes), dtype=bool))
        patch = model(dynamic[:, :, nodes], static[nodes], future[:, :, nodes], graph["supports"])
    assert torch.allclose(patch, full[:, :, nodes], atol=1e-5)
    # A non-closed patch is refused rather than silently wrong.
    with pytest.raises(ValueError, match="closed"):
        sampler.subgraph(nodes[1:], np.ones(len(nodes) - 1, dtype=bool))
