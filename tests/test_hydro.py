"""Distributed model: routing conserves mass; gauge ownership and assimilation levels."""

from types import SimpleNamespace

import numpy as np
import torch

from tva_flood.data.river import build_operator
from tva_flood.hydro import paths as P
from tva_flood.hydro.data import Gauges
from tva_flood.hydro.model import RiverGraph, Router

# 0 -> 1 -> 3, 2 -> 3 -> 4: a confluence at 3, outlet 4.
EDGES = np.array([[0, 1], [1, 3], [2, 3], [3, 4]])
AREA = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def network():
    op = build_operator(5, EDGES, AREA, np.full(5, 10.0))
    rows = np.repeat(np.arange(5), np.diff(op["indptr"]))
    share = op["weight"] * op["basin_area_km2"][rows] / AREA[op["indices"]]
    group = (op["distance"] > 12).astype(np.int16)  # near and far groups
    return SimpleNamespace(
        n=5,
        indptr=op["indptr"],
        members=op["indices"],
        share=share.astype(np.float32),
        group=group,
        basin_area=op["basin_area_km2"],
    )


def router(net):
    r = Router(2, np.array([0.0, 3.0]))
    r.attach(net, np.array([3, 4]), torch.device("cpu"))
    return r


def test_steady_inflow_adds_up_downstream():
    net = network()
    q = torch.as_tensor(np.repeat(AREA[:, None], P.FLOW_STEPS, axis=1), dtype=torch.float32)
    routed = router(net)(q)[:, P.ROUTE :]  # past the routing memory: steady state
    expected = torch.as_tensor(net.basin_area, dtype=torch.float32)[:, None]
    assert torch.allclose(routed, expected.expand_as(routed), rtol=1e-5)


def test_pulse_mass_is_conserved_and_delayed():
    net = network()
    q = torch.zeros(5, P.FLOW_STEPS)
    q[0, 10] = 6.0  # one pulse at the top reach
    r = router(net)
    routed = r(q)
    assert torch.isclose(routed[4].sum(), torch.tensor(6.0), rtol=1e-5)
    assert routed[4, :10].abs().sum() == 0  # nothing arrives before it starts
    gauges = r(q, gauges_only=True)
    assert torch.allclose(gauges, routed[[3, 4]])


def test_owner_is_the_smallest_visible_basin():
    # Gauges at reaches 1 (basin {0, 1}), 3 (basin {0..3}) and 4 (outlet).
    net = network()
    g = Gauges.__new__(Gauges)
    g.count = 3
    g.reach = np.array([1, 3, 4])
    g.basins = [net.members[net.indptr[r] : net.indptr[r + 1]] for r in g.reach]
    g.size = np.array([len(b) for b in g.basins])
    assert g.owners(np.array([True, True, True]), 5).tolist() == [0, 0, 1, 1, 2]
    assert g.owners(np.array([True, False, True]), 5).tolist() == [0, 0, 2, 2, 2]


def test_river_graph_averages_parents_and_children():
    graph = RiverGraph(EDGES, 5, torch.device("cpu"))
    x = torch.arange(5, dtype=torch.float32)[:, None]
    assert graph.upstream(x)[:, 0].tolist() == [0.0, 0.0, 0.0, 1.5, 3.0]  # 3: parents 1, 2
    assert graph.downstream(x)[:, 0].tolist() == [1.0, 3.0, 3.0, 4.0, 0.0]


def test_nse_star_is_the_squared_error_over_the_scaled_std():
    from tva_flood.hydro.train import nse_star

    truth = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 10.0, 10.0, 10.0]])
    pred = truth + torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    valid = torch.ones_like(truth, dtype=torch.bool)
    std = torch.tensor([0.9, 1.9])  # (std + 0.1) = 1 and 2: errors 1 and 2 give 1 and 1
    assert abs(float(nse_star(pred, truth, valid, std)) - 1.0) < 1e-6
    assert float(nse_star(truth, truth, valid, std)) == 0.0


def test_pinball_penalizes_a_quantile_below_the_truth_by_q_and_above_by_one_minus_q():
    from tva_flood.hydro.train import pinball

    truth = torch.full((1, 4), 10.0)
    valid = torch.ones(1, 4, dtype=torch.bool)
    std = torch.tensor([0.9])  # scale 1
    below = torch.full((1, 4, 3), 8.0)  # misses by 2: loss q * 2 for q = 0.5, 0.9, 0.99
    above = torch.full((1, 4, 3), 12.0)  # exceeds by 2: loss (1 - q) * 2
    assert abs(float(pinball(below, truth, valid, std)) - 2 * (0.5 + 0.9 + 0.99) / 3) < 1e-5
    assert abs(float(pinball(above, truth, valid, std)) - 2 * (0.5 + 0.1 + 0.01) / 3) < 1e-5


def test_attention_trunk_is_causal_and_keeps_the_trunk_interface():
    from tva_flood.hydro.model import REACH_INPUTS, STEP_INPUTS, AttentionTrunk

    torch.manual_seed(0)
    trunk = AttentionTrunk(hidden=16, channels=8, graph_layers=1, heads=2, layers=2).eval()
    steps = torch.randn(3, P.WINDOW, STEP_INPUTS)
    reach = torch.randn(3, REACH_INPUTS)
    before = trunk._time(steps, reach)
    assert before.shape == (3, P.FLOW_STEPS, 8)
    later = steps.clone()
    later[:, 150:] += 5.0  # change only the steps from 150 on
    after = trunk._time(later, reach)
    changed_from = 150 - (P.WINDOW - P.FLOW_STEPS)  # outputs cover the last FLOW_STEPS steps
    assert torch.allclose(before[:, :changed_from], after[:, :changed_from], atol=1e-6)  # nothing earlier moved
    assert not torch.allclose(before[:, changed_from:], after[:, changed_from:])  # the changed steps did


def test_attention_model_config_round_trips():
    from tva_flood.hydro.model import AttentionGNN

    model = AttentionGNN([0.0] * 3, np.zeros(3), hidden=16, channels=8, graph_layers=1, heads=2, layers=1)
    assert model.config["kind"] == "attention" and model.config["attention"] == dict(heads=2, layers=1)
    assert not hasattr(model.trunk, "lstm") and hasattr(model.trunk, "layers")
