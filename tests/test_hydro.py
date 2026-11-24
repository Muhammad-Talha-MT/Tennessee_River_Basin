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
        edges=EDGES,
        static=np.zeros((5, 48), dtype=np.float32),
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


def test_instant_routing_conserves_mass_without_delay():
    from tva_flood.hydro.model import Router as R

    net = network()
    r = R(2, np.array([0.0, 3.0]), learned=False)
    r.attach(net, np.array([3, 4]), torch.device("cpu"))
    q = torch.zeros(5, 20)
    q[0, 2] = 1.0  # one pulse at the top of the river
    out = r(q)
    assert not r.logits.requires_grad
    assert torch.allclose(out[4], out[4].clamp_min(0))
    assert int(out[4].argmax()) == 2  # arrives at the outlet in the same step
    assert torch.isclose(out[4].sum(), torch.tensor(1.0), atol=1e-5)


def test_graph_block_without_mixing_ignores_neighbours():
    from tva_flood.hydro.model import GraphBlock

    torch.manual_seed(0)
    graph = RiverGraph(EDGES, 5, torch.device("cpu"))
    block = GraphBlock(4, mixing=False)
    h = torch.randn(5, 3, 4)
    changed = h.clone()
    changed[0] += 3 * torch.randn(3, 4)  # an upstream neighbour of reach 1 changes (not a constant: LayerNorm removes it)
    assert torch.allclose(block(h, graph)[1], block(changed, graph)[1])
    block.mixing = True
    assert not torch.allclose(block(h, graph)[1], block(changed, graph)[1])


def manning(net, gauges=(1, 3)):
    from tva_flood.hydro.channel import ManningRouter

    table = dict(length=np.full(5, 10.0), slope=np.full(5, 1e-3), order=np.ones(5), lake=np.array([False, False, False, True, False]))
    r = ManningRouter()
    r.attach(net, np.array(gauges), torch.device("cpu"), table=table)
    return r


def test_manning_routing_conserves_mass_and_delays_with_distance():
    net = network()
    r = manning(net)
    q = torch.zeros(5, P.FLOW_STEPS)
    q[0, 10] = 6.0
    r.prepare(r.reference_flow(q + 1.0))
    routed = r(q)
    assert torch.isclose(routed[4].sum(), torch.tensor(6.0), rtol=1e-5)  # all of it arrives
    assert routed[4, :10].abs().sum() == 0  # nothing before the pulse
    peak = routed.argmax(dim=1)
    assert peak[0] <= peak[1] <= peak[3] <= peak[4]  # later the further down
    assert torch.allclose(r(q, gauges_only=True), routed[[1, 3]], atol=1e-6)


def test_manning_travel_time_falls_with_flow():
    net = network()
    r = manning(net)
    with torch.no_grad():
        slow = r.reach_time(torch.full((5,), 1.0))
        fast = r.reach_time(torch.full((5,), 100.0))
    river = torch.tensor([True, True, True, False, True])  # reach 3 is a lake: its speed does not depend on flow
    assert (fast[river] < slow[river]).all()
    assert torch.isclose(fast[3], slow[3])


def test_manning_gradients_reach_the_channel_parameters():
    torch.manual_seed(0)
    net = network()
    r = manning(net)
    with torch.no_grad():
        r.channel[-1].weight.normal_(0, 0.1)  # off the zero start, so every layer gets a gradient
    q = torch.rand(5, P.FLOW_STEPS, requires_grad=True)
    r.prepare(r.reference_flow(q))
    target = torch.linspace(0, 1, P.FLOW_STEPS)
    loss = ((r(q)[4] - target) ** 2).mean()
    loss.backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in r.parameters())


def test_insertion_reaches_only_reaches_below_the_gauge_and_conserves_mass():
    net = network()
    r = manning(net, gauges=(1, 3))
    q = torch.ones(5, P.FLOW_STEPS)
    r.prepare(r.reference_flow(q))
    dq = torch.zeros(2, P.FLOW_STEPS)
    dq[0, 5] = 2.0  # a correction at the gauge of reach 1
    to_reaches = r(dq, op="insert")
    assert to_reaches[[0, 1, 2]].abs().sum() == 0  # not upstream, not itself, not the other branch
    assert torch.isclose(to_reaches[3].sum(), torch.tensor(2.0), rtol=1e-5)
    assert torch.isclose(to_reaches[4].sum(), torch.tensor(2.0), rtol=1e-5)
    to_gauges = r(dq, op="insert_gauges")
    assert torch.allclose(to_gauges[1], to_reaches[3], atol=1e-6) and to_gauges[0].abs().sum() == 0


# ---- insertion audit (reports/v3, Phase 1.1) -------------------------------------------------------------------------------
def _insertion_setup(gauge_reaches=(1, 3, 4)):
    """HydroModel.insertion on a tiny network: reaches 0 -> 1 -> 3, 2 -> 3 -> 4; gauges at ``gauge_reaches``."""
    from hydro_stub import stub_for_insertion

    return stub_for_insertion(manning(network(), gauges=gauge_reaches))


def _batch(observed_value):
    """A batch holding only what ``insertion`` reads: observed flow [WINDOW, G] (3 gauges), NaN outside the given values."""
    observed = torch.full((P.WINDOW, 3), float("nan"))
    observed[:] = observed_value
    return dict(observed=observed)


def test_inserted_observation_changes_downstream_forecast_only():
    from hydro_stub import run_insertion

    r_flow = torch.ones(3, P.FLOW_STEPS)
    obs = torch.ones(P.WINDOW, 3)
    obs[:, 0] += 2.0  # visible gauge 0 (reach 1) reads 2 m3/s above what the model routes
    shown = torch.tensor([True, False, True])
    to_reaches, to_gauges = run_insertion(_insertion_setup(), dict(observed=obs), r_flow, r_flow.clone().repeat(1, 1)[:, : P.SCORED], shown)
    assert to_reaches[[0, 1, 2]].abs().sum() == 0  # nothing upstream of, at, or beside the gauge
    assert to_reaches[3].abs().sum() > 0 and to_reaches[4].abs().sum() > 0  # the downstream reaches change
    assert to_gauges[0].abs().sum() == 0  # a gauge never receives its own correction
    assert to_gauges[1].abs().sum() > 0 and to_gauges[2].abs().sum() > 0  # gauges below it do


def test_withheld_gauge_observation_never_enters_insertion():
    from hydro_stub import run_insertion

    r_flow = torch.ones(3, P.FLOW_STEPS)
    shown = torch.tensor([True, False, True])  # gauge 1 (reach 3) is withheld
    base = torch.ones(P.WINDOW, 3)
    base[:, 0] += 2.0
    leaky = base.clone()
    leaky[:, 1] = 1e6  # the withheld gauge's true flow, wildly different from the model
    gone = base.clone()
    gone[:, 1] = float("nan")  # what ``Samples.gauge_view`` really passes
    st = _insertion_setup()
    fcst = r_flow[:, : P.SCORED]
    a = run_insertion(st, dict(observed=leaky), r_flow, fcst, shown)
    b = run_insertion(st, dict(observed=gone), r_flow, fcst, shown)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_insertion_uses_only_flow_before_the_issue():
    from hydro_stub import run_insertion

    r_flow = torch.ones(3, P.FLOW_STEPS)
    shown = torch.tensor([True, True, True])
    st = _insertion_setup()
    fcst = r_flow[:, : P.SCORED]
    obs = torch.ones(P.WINDOW, 3)
    obs[: P.HISTORY, 0] += 1.0
    changed = obs.clone()
    changed[P.HISTORY :] += 50.0  # the observed flow after the issue: unknown at issue time
    a = run_insertion(st, dict(observed=obs), r_flow, fcst, shown)
    b = run_insertion(st, dict(observed=changed), r_flow, fcst, shown)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_insertion_correction_is_routed_with_the_travel_time():
    from hydro_stub import run_insertion

    r_flow = torch.ones(3, P.FLOW_STEPS)
    shown = torch.tensor([True, False, False])
    obs = torch.ones(P.WINDOW, 3)
    obs[P.HISTORY - 1, 0] += 4.0  # one step, right before the issue
    st = _insertion_setup()
    to_reaches, _ = run_insertion(st, dict(observed=obs), r_flow, r_flow[:, : P.SCORED], shown)
    # the correction arrives downstream later than it was made, and none of it is lost on the way (mass conserved)
    peak3 = int(to_reaches[3].argmax())
    assert peak3 >= P.PAST - 1 - (P.HISTORY - P.PAST) + 0  # not earlier than the step it was observed at
    assert torch.isclose(to_reaches[3].sum(), torch.tensor(4.0), rtol=1e-4) and torch.isclose(to_reaches[4].sum(), torch.tensor(4.0), rtol=1e-4)


def test_gauge_view_hides_withheld_gauges_and_shows_only_eligible_ones():
    from tva_flood.hydro.samples import Samples

    net = network()
    g = Gauges.__new__(Gauges)
    g.count = 3
    g.reach = np.array([1, 3, 4])
    g.basins = [net.members[net.indptr[r] : net.indptr[r + 1]] for r in g.reach]
    g.size = np.array([len(b) for b in g.basins])
    step = P.HISTORY + 5
    rng = np.random.default_rng(0)
    g.observed = rng.uniform(1, 2, (step + P.HORIZON + 5, 3))
    g.observed[step - 30 : step, 0] = np.nan  # gauge 0 has only 2 values in the last 32 steps (< 8): not eligible
    fake = SimpleNamespace(gauges=g, network=SimpleNamespace(n=5))
    withheld = np.array([False, True, False])
    eligible = Samples.eligible(fake, step, withheld)
    assert eligible.tolist() == [False, False, True]
    view = Samples.gauge_view(fake, step, np.array([True, True, True]), withheld)
    assert np.isnan(view["observed"][:, 1]).all()  # the withheld gauge: no flow at any time, before or after the issue
    assert not view["visible"][1] and view["visible"][2]
    # a withheld gauge's true flow never reaches the model, whatever it is
    g.observed[:, 1] = 1e6
    view2 = Samples.gauge_view(fake, step, np.array([True, True, True]), withheld)
    assert np.isnan(view2["observed"][:, 1]).all() and view2["owner"].tolist() == view["owner"].tolist()
    assert view["owner"].tolist() == [0, 0, 2, 2, 2]  # reaches 2..4 belong to the outlet gauge, not to the withheld one


# ---- M4: routing replaced by instantaneous summation (reports/v3, Phase 2) ----------------------------------------------------
def _instant_router(net, gauges=(1, 3, 4)):
    from tva_flood.hydro.model import Router as R

    r = R(2, np.array([0.0, 3.0]), learned=False)
    r.attach(net, np.array(gauges), torch.device("cpu"))
    return r


def test_instant_routing_is_the_share_weighted_sum_of_upstream_inflow_at_the_same_step():
    net = network()
    r = _instant_router(net)
    q = torch.rand(5, P.FLOW_STEPS)
    out = r(q)
    expected = np.zeros((5, P.FLOW_STEPS))
    for i in range(5):
        for k in range(net.indptr[i], net.indptr[i + 1]):
            expected[i] += net.share[k] * q[net.members[k]].numpy()  # the same mass shares the Manning router splits over delays
    assert np.allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
    assert torch.allclose(r(q, gauges_only=True), out[[1, 3, 4]], atol=1e-6)


def test_instant_routing_conserves_mass_exactly_and_adds_no_delay():
    net = network()
    r = _instant_router(net)
    q = torch.rand(5, P.FLOW_STEPS)
    out = r(q)
    # outlet 4 drains the whole tree: its flow at every step is the sum of all lateral inflow at that step (mass in = mass out, no delay)
    assert torch.allclose(out[4], q.sum(0), rtol=1e-5)
    assert torch.isclose(out[4].sum(), q.sum(), rtol=1e-5)
    pulse = torch.zeros(5, P.FLOW_STEPS)
    pulse[0, 20] = 1.0
    arrival = r(pulse)
    assert int(arrival[4].argmax()) == 20 and torch.isclose(arrival[4].sum(), torch.tensor(1.0), rtol=1e-5)  # same step, nothing lost


def test_instant_and_manning_routing_share_the_steady_state():
    # M1 against M4 is a single-factor comparison: with a constant inflow both routers give the same steady flow at every reach
    net = network()
    inst, man = _instant_router(net), manning(net)
    q = torch.rand(5, 1).repeat(1, P.FLOW_STEPS)
    man.prepare(man.reference_flow(q))
    assert torch.allclose(inst(q)[:, -1], man(q)[:, -1], rtol=1e-4)


def test_model_with_the_instant_router_has_no_routing_parameters_and_round_trips_its_config():
    from tva_flood.hydro.model import HydroModel, build_model

    model = HydroModel(2, np.array([0.0, 3.0]), hidden=16, channels=8, graph_layers=1, router="instant")
    assert model.config["router"] == "instant" and "insert" not in model.config
    assert not any(p.requires_grad for p in model.router.parameters())  # nothing to learn in the router
    again = build_model(model.config)
    assert again.config == model.config and type(again.router) is type(model.router)
    try:
        HydroModel(2, np.array([0.0, 3.0]), hidden=16, channels=8, router="instant", insert=True)
    except ValueError:
        pass
    else:
        raise AssertionError("insertion needs the Manning router")
