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


# Dams at reaches 1 (holding back 0 and 1) and 3 (2, 3 and dam 0's release); 4 runs free.
DAMS = SimpleNamespace(
    count=2,
    reach=np.array([1, 3]),
    dam_below=np.array([0, 0, 1, 1, -1]),
    downstream=np.array([1, -1]),
)


def dam_router(net):
    r = Router(2, np.array([0.0, 3.0]), levels=1, dams=True)
    r.attach(net, np.array([3, 4]), torch.device("cpu"), dams=DAMS)
    return r


def test_dams_passing_their_inflow_leave_the_river_unchanged():
    net = network()
    q = torch.as_tensor(np.repeat(AREA[:, None], P.FLOW_STEPS, axis=1), dtype=torch.float32)
    r = dam_router(net)
    release = torch.zeros(2, P.FLOW_STEPS)
    release[0] = AREA[[0, 1]].sum()  # dam 0 passes what its basin yields
    inflow = r.inflow(q, release)[:, P.ROUTE :]
    assert torch.allclose(inflow[0], torch.tensor(3.0)) and torch.allclose(inflow[1], torch.tensor(10.0))
    release[1] = 10.0
    split = r(q, release=release)[:, P.ROUTE :]
    natural = r(q)[:, P.ROUTE :]
    assert torch.allclose(split, natural, rtol=1e-5)
    assert torch.allclose(r(q, gauges_only=True, release=release)[:, P.ROUTE :], split[[3, 4]])


def test_a_dam_holding_its_water_stops_it():
    net = network()
    q = torch.as_tensor(np.repeat(AREA[:, None], P.FLOW_STEPS, axis=1), dtype=torch.float32)
    r = dam_router(net)
    release = torch.zeros(2, P.FLOW_STEPS)  # both dams release nothing
    split = r(q, release=release)[:, P.ROUTE :]
    assert torch.allclose(split[0], torch.tensor(1.0))  # above dam 0: its own runoff
    assert torch.allclose(split[[1, 3]], torch.tensor(0.0))  # the dams release nothing
    assert torch.allclose(split[2], torch.tensor(3.0))  # a free tributary
    assert torch.allclose(split[4], torch.tensor(5.0))  # only reach 4's own runoff below dam 1
    assert torch.allclose(r.inflow(q, release)[1, P.ROUTE :], torch.tensor(7.0))  # 2 and 3 reach dam 1


def test_gauge_corrections_run_down_to_the_next_visible_gauge_or_dam():
    from tva_flood.hydro.model import Corrections

    net = network()
    net.n, net.edges = 5, EDGES
    # Gauges at reaches 0 and 3, no dams: 0's correction reaches 1, 3 and 4 unless 3 is visible.
    fix = Corrections(net, np.array([0, 3]), None, 1, torch.device("cpu"))
    kernels = torch.zeros(2, P.ROUTE)
    kernels[:, 0] = 1.0  # no delay, to read the result directly
    correction = torch.tensor([[2.0] * 8, [5.0] * 8])
    out = fix(correction, torch.tensor([True, False]), kernels)
    assert out[:, -1].tolist() == [0.0, 2.0, 0.0, 2.0, 2.0]
    out = fix(correction, torch.tensor([True, True]), kernels)
    assert out[:, -1].tolist() == [0.0, 2.0, 0.0, 0.0, 5.0]  # 3 takes over below itself
    # A dam at reach 1 stops gauge 0's correction there: its release replaces the river.
    fix = Corrections(net, np.array([0, 3]), DAMS, 1, torch.device("cpu"))
    assert fix(correction, torch.tensor([True, False]), kernels)[:, -1].tolist() == [0.0] * 5
