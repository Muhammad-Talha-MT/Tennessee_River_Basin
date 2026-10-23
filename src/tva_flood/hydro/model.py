"""Graph-routing forecast model: per-reach LSTM, river-graph layers, mass-conserving routing.

Shapes: n reaches, G gauges, T weather steps (``paths.WINDOW``), F inflow steps
(``paths.FLOW_STEPS``, 80 steps before the issue to the last lead), S scored steps.

1. Inputs per reach and step: weather (rain only in the future, as forecast), season, and
   gauge history: at a visible gauge's reach its observed log flow; at every reach of a
   visible gauge's basin that gauge's specific discharge (flow per km2), so a whole basin
   sees what its gauge is doing. Per reach: attributes, antecedent storage and its area
   ratio to the owning gauge.
2. ``Trunk``: an LSTM per reach over time (run in reach chunks to fit memory), then
   ``GraphBlock`` layers that mix each reach's state with the mean state of its upstream and
   of its downstream neighbours at every step (diffusion-style message passing).
3. Two heads from the same states. Simulation: lateral inflow q >= 0 (runoff depth x
   local catchment area), trained to the NWM retrospective. Observed: q x exp(delta), with
   a time-varying delta trained on gauges (kept small by a penalty, so ungauged reaches
   stay close to the simulation).
4. ``Router``: Q_i(t) = sum over upstream j of f_ij (h_g * q_j)(t), one travel-time kernel
   per group (river distance x lakes on the path), each summing to one: mass is conserved
   and flow grows with drainage area.
5. ``GaugeHead`` at visible gauges: a sequence model over the gauge's 30-day history and the
   routed forecast that forecasts the gauge directly (starting from corrected persistence).
   Ungauged reaches and hidden gauges keep the routed observed head.
6. ``spread``: log-flow standard deviation at gauges (probabilistic scores).
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import paths as P

EPS = 0.01  # m3/s in log ratios
STEP_INPUTS = 8 + 8 + 1 + 2 + 2 + 2  # weather, validity, future, season, own gauge, owner gauge
REACH_INPUTS = 48 + 7 + 2  # attributes, antecedent storage, owner area ratio and flag


def mlp(inputs, hidden, outputs, zero=False):
    net = nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))
    if zero:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


class SparseProduct(torch.autograd.Function):
    """A @ x with a fixed sparse A (float32); the transpose is stored for the backward."""

    @staticmethod
    def forward(ctx, a, at, x):
        ctx.at = at
        return a @ x

    @staticmethod
    def backward(ctx, grad):
        return None, None, ctx.at @ grad.float().contiguous()


def csr(rows, cols, values, shape, device):
    """Sparse CSR matrix and its transpose (CSR support is marked beta: warning silenced)."""
    warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
    coo = torch.sparse_coo_tensor(
        torch.stack([torch.as_tensor(rows), torch.as_tensor(cols)]),
        torch.as_tensor(values, dtype=torch.float32),
        shape,
    ).coalesce()
    return coo.to_sparse_csr().to(device), coo.t().coalesce().to_sparse_csr().to(device)


class RiverGraph:
    """Mean state of each reach's upstream (parents) and downstream (children) neighbours.

    River networks are nearly trees (about one edge per reach), so a gather and an
    index_add over the edges is cheaper than a sparse product and stays in bf16.
    """

    def __init__(self, edges, n, device):
        e = torch.as_tensor(edges, dtype=torch.long, device=device)
        self.source, self.target = e[:, 0], e[:, 1]
        self.parents = torch.bincount(self.target, minlength=n).clamp(min=1)
        self.children = torch.bincount(self.source, minlength=n).clamp(min=1)

    def upstream(self, x):
        total = torch.zeros_like(x).index_add_(0, self.target, x[self.source])
        return total / self.parents.to(x.dtype).view(-1, *([1] * (x.dim() - 1)))

    def downstream(self, x):
        total = torch.zeros_like(x).index_add_(0, self.source, x[self.target])
        return total / self.children.to(x.dtype).view(-1, *([1] * (x.dim() - 1)))


class GraphBlock(nn.Module):
    """h + MLP([h, mean of upstream h, mean of downstream h]) at every step, on [n, F, C]."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mix = mlp(3 * channels, channels, channels)

    def forward(self, h, graph):
        x = self.norm(h)
        return h + self.mix(torch.cat([x, graph.upstream(x), graph.downstream(x)], dim=-1))


class Trunk(nn.Module):
    """Reach states [n, F, channels] from step inputs [n, T, STEP_INPUTS] and reach inputs."""

    def __init__(self, hidden, channels, graph_layers, chunk=15000):
        super().__init__()
        self.inputs = nn.Linear(STEP_INPUTS, hidden)
        self.reach = mlp(REACH_INPUTS, hidden, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        with torch.no_grad():  # forget gate open at the start: long storage memory
            self.lstm.bias_hh_l0[hidden : 2 * hidden] = 3.0
        self.down = nn.Linear(hidden, channels)
        self.blocks = nn.ModuleList(GraphBlock(channels) for _ in range(graph_layers))
        self.chunk = chunk

    def _time(self, steps, reach):
        x = self.inputs(steps) + self.reach(reach)[:, None, :]
        h, _ = self.lstm(x)
        return self.down(h[:, -P.FLOW_STEPS :])

    def forward(self, steps, reach, graph):
        parts = []
        for s, r in zip(steps.split(self.chunk), reach.split(self.chunk)):
            if self.training:
                parts.append(checkpoint(self._time, s, r, use_reentrant=False))
            else:
                parts.append(self._time(s, r))
        h = torch.cat(parts)
        for block in self.blocks:
            h = checkpoint(block, h, graph, use_reentrant=False) if self.training else block(h, graph)
        return h


class Router(nn.Module):
    """Mass-conserving routing of lateral inflow with one travel-time kernel per group."""

    def __init__(self, groups, delay):
        super().__init__()
        taps = torch.arange(P.ROUTE, dtype=torch.float32)
        center = torch.as_tensor(delay, dtype=torch.float32).clamp(max=P.ROUTE - 1)[:, None]
        self.logits = nn.Parameter(-((torch.log1p(taps)[None] - torch.log1p(center)) ** 2) / (2 * 0.35**2))
        self.groups = groups

    def attach(self, network, gauge_reaches, device):
        """Sparse operators: all reaches (rows group * n + i) and gauge reaches only."""
        n = network.n
        rows = np.repeat(np.arange(n), np.diff(network.indptr))
        group = network.group.astype(np.int64)
        self.n, self.gauge_count = n, len(gauge_reaches)
        self.full = csr(group * n + rows, network.members, network.share, (self.groups * n, n), device)
        position = np.full(n, -1)
        position[gauge_reaches] = np.arange(len(gauge_reaches))
        keep = position[rows] >= 0
        self.gauges = csr(
            group[keep] * self.gauge_count + position[rows[keep]],
            network.members[keep],
            network.share[keep],
            (self.groups * self.gauge_count, n),
            device,
        )

    def kernels(self):
        return torch.softmax(self.logits, dim=-1)  # [group, tap], each sums to one

    @torch.autocast("cuda", enabled=False)
    def forward(self, q, gauges_only=False):
        """Routed flow [rows, F] (float32) from lateral inflow q [n, F] (m3/s)."""
        a, at = self.gauges if gauges_only else self.full
        rows = self.gauge_count if gauges_only else self.n
        u = SparseProduct.apply(a, at, q.float().contiguous())
        u = u.view(self.groups, rows, -1).permute(1, 0, 2)
        out = F.conv1d(F.pad(u, (P.ROUTE - 1, 0)), self.kernels().flip(-1)[:, None, :], groups=self.groups)
        return out.sum(dim=1)


class GaugeHead(nn.Module):
    """Direct forecast [g, lead] at visible gauges from their history and the routed flow.

    Encoder: an LSTM over the last 30 days at the gauge: observed flow (and whether it
    exists), and over the last 8 days the routed observed-head and simulated flow there; with
    ``lstm_inputs`` also the basin-mean weather (16 channels) and the 48 basin-mean attributes
    at every step (the LSTM baseline's inputs). Decoder: an LSTM over the 40 leads, started
    from the encoder state, reading the routed forecast of both heads, the basin-mean forecast
    rain, a lead embedding (and the attributes). Gauge descriptors enter the encoder's start state.
    Output: a log correction of the routed observed-head flow, a learned share of the latest
    misfit fading with lead (short leads begin as corrected persistence, later leads as the
    routed forecast) plus a decoder term.

    ``scale``: flows enter as log1p(flow) / 5 ("log5") or as the log flow normalized by its mean
    and std over the training data ("z", the LSTM baseline's scaling). The defaults are the first
    head of this model; ``hidden=256, scale="z", lstm_inputs=True`` is the one in use (see
    ``scripts/head_settings.py``).
    """

    def __init__(self, hidden=64, statics=48, scale="log5", lstm_inputs=False, flow_mean=0.0, flow_std=1.0):
        super().__init__()
        self.scale, self.lstm_inputs, self.flow_mean, self.flow_std = scale, lstm_inputs, flow_mean, flow_std
        self.describe = mlp(statics + 2, hidden, hidden)
        self.encode = nn.LSTM(4 + (16 + 48 if lstm_inputs else 0), hidden, batch_first=True)
        self.decode = nn.LSTM(4 + 16 + (48 if lstm_inputs else 0), hidden, batch_first=True)
        self.lead = nn.Embedding(P.HORIZON, 16)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.keep = nn.Parameter(3.0 - 0.25 * torch.arange(P.HORIZON, dtype=torch.float32))

    def flow(self, x):
        z = torch.log1p(x.clamp_min(0))
        return z / 5 if self.scale == "log5" else (z - self.flow_mean) / self.flow_std

    @torch.autocast("cuda", enabled=False)
    def forward(self, history, routed, simulated, rain, describe, weather=None, basin=None):
        """history [g, H] observed flow (NaN missing) before t; routed, simulated [g, S]
        (scored window); rain [g, lead, 2] basin-mean forecast rain and known share;
        describe [g, statics + 2]; with ``lstm_inputs`` weather [g, H, 16] and basin attributes
        [g, 48]. Returns log corrections [g, lead]."""
        g = history.shape[0]
        before = P.INNOVATION
        valid = torch.isfinite(history)
        obs = torch.where(valid, self.flow(torch.nan_to_num(history)), 0.0)
        tail = torch.zeros(g, history.shape[1], 2, device=history.device)
        tail[:, -before:, 0] = self.flow(routed[:, :before])
        tail[:, -before:, 1] = self.flow(simulated[:, :before])
        steps = [obs[..., None], valid[..., None].float(), tail]
        future = [self.flow(routed[:, before:])[..., None], self.flow(simulated[:, before:])[..., None], rain.float(),
                  self.lead.weight[None].expand(g, -1, -1)]
        if self.lstm_inputs:
            steps += [weather.float(), basin.float()[:, None].expand(-1, history.shape[1], -1)]
            future.append(basin.float()[:, None].expand(-1, P.HORIZON, -1))
        h0 = self.describe(describe.float())[None].contiguous()
        _, (h, c) = self.encode(torch.cat(steps, dim=-1), (h0, torch.zeros_like(h0)))
        decoded, _ = self.decode(torch.cat(future, dim=-1), (h, c))
        learned = 3 * torch.tanh(self.out(decoded)[..., 0])
        latest = valid[:, -1]
        misfit = torch.log(torch.nan_to_num(history[:, -1]).clamp_min(0) + EPS) - torch.log(routed[:, before - 1] + EPS)
        misfit = torch.where(latest, misfit.clamp(-5, 5), torch.zeros_like(misfit))
        return torch.sigmoid(self.keep)[None] * misfit[:, None] + learned


QUANTILES = (0.5, 0.9, 0.99)  # flow quantiles of the tail head (``quantiles=True``)


class HydroModel(nn.Module):
    """``quantiles=True`` replaces the spread head by a tail head: the 0.5, 0.9 and 0.99 quantiles of
    the flow at every gauge and lead, each the forecast (detached, so the tail loss never moves the
    mean) times exp of a non-decreasing offset."""

    def __init__(self, groups, delay, hidden=128, channels=32, graph_layers=3, quantiles=False, head=None):
        super().__init__()
        self.config = dict(
            groups=groups,
            delay=[float(d) for d in delay],
            hidden=hidden,
            channels=channels,
            graph_layers=graph_layers,
        )
        if quantiles:
            self.config["quantiles"] = True
        if head:
            self.config["head"] = head  # GaugeHead settings
        self.trunk = Trunk(hidden, channels, graph_layers)
        self.simulation = nn.Linear(channels, 1)
        self.observed = nn.Linear(channels, 1)
        nn.init.zeros_(self.observed.weight)
        nn.init.zeros_(self.observed.bias)
        self.router = Router(groups, delay)
        self.head = GaugeHead(**(head or {}))
        self.lead = nn.Embedding(P.HORIZON, 16)
        if quantiles:
            self.tail = mlp(48 + 16 + 4, 64, len(QUANTILES))
            nn.init.zeros_(self.tail[-1].weight)
            self.tail[-1].bias.data = torch.tensor([0.0, -0.5, -0.5])  # softplus(-0.5) = 0.47 per step up
        else:
            self.spread = mlp(48 + 16 + 3, 64, 1)

    def attach(self, network, edges, gauge_reaches, device):
        """Fixed operators of the river graph on the device (not parameters): routing, graph,
        and for each gauge its basin-mean weights and the share of its basin behind lakes."""
        self.router.attach(network, gauge_reaches, device)
        self.graph = RiverGraph(edges, network.n, device)
        rows = np.repeat(np.arange(network.n), np.diff(network.indptr))
        position = np.full(network.n, -1)
        position[gauge_reaches] = np.arange(len(gauge_reaches))
        keep = position[rows] >= 0
        mass = network.share[keep] * network.area[network.members[keep]] / network.basin_area[rows[keep]]
        g = len(gauge_reaches)
        self.basin_mean = csr(position[rows[keep]], network.members[keep], mass, (g, network.n), device)
        lake = (network.group[keep] % 3) > 0  # lake or reservoir reaches on the path
        share = np.bincount(position[rows[keep]], weights=mass * lake, minlength=g)
        self.lake_share = torch.as_tensor(share, dtype=torch.float32, device=device)

    def quantiles(self, static, flags, mean):
        """Flow quantiles ``QUANTILES`` [g, lead, 3] around ``mean`` [g, lead] (the forecast, detached): the mean times exp of a
        non-decreasing offset. ``static`` [g, 48] attributes at the gauge reach; ``flags`` [g, 3]: visible, owned, log flow
        at the last lead."""
        g = len(static)
        features = torch.cat(
            [
                static[:, None, :].expand(-1, P.HORIZON, -1),
                self.lead.weight[None].expand(g, -1, -1),
                flags[:, None, :].expand(-1, P.HORIZON, -1),
                (torch.log1p(mean) / 5)[..., None],
            ],
            dim=-1,
        )
        raw = self.tail(features).float()
        offset = torch.cumsum(torch.cat([raw[..., :1], F.softplus(raw[..., 1:])], dim=-1), dim=-1)
        return mean[..., None].float() * torch.exp(offset.clamp(max=6.0))

    def gauge_rain(self, b):
        """Basin-mean forecast rain and whether it is known at every gauge [G, lead, 2]."""
        future = slice(P.HISTORY, P.WINDOW)
        rain = torch.stack([b["weather"][:, future, 0], b["valid"][:, future, 0]], dim=-1).float()
        return SparseProduct.apply(*self.basin_mean, rain.reshape(rain.shape[0], -1)).view(-1, P.HORIZON, 2)

    def gauge_weather(self, b):
        """Basin-mean weather and validity over the last 30 days at every gauge [G, 120, 16] (the LSTM's inputs)."""
        past = slice(P.HISTORY - 120, P.HISTORY)
        weather = torch.cat([b["weather"][:, past], b["valid"][:, past]], dim=-1).float()
        return SparseProduct.apply(*self.basin_mean, weather.reshape(weather.shape[0], -1)).view(-1, 120, 16)

    def gauge_describe(self, b):
        """Gauge descriptors [G, 50]: attributes at the gauge reach, log basin area, share of the basin behind lakes."""
        return torch.cat([b["static"][b["gauge_reach"]], torch.log(b["gauge_area"])[:, None] / 10, self.lake_share[:, None]], dim=-1)

    def forward(self, b, full=False):
        """Outputs over the scored window [t-32, t+40) (``paths.SCORED`` steps).

        ``sim`` [n, S]: simulation head at every reach; ``observed`` [n, S]: observed head at
        every reach; ``routed_gauge`` [G, S]: observed head at every gauge; ``gauge`` [G, S]:
        the forecast at every gauge (visible ones from the gauge head); ``delta`` [n, F]: observed-head log correction; ``sigma`` [G, lead]
        (or, with a tail head, ``quantile`` [G, lead, 3]: flow quantiles ``QUANTILES``); with
        ``full`` also ``reach`` [n, S]: ``observed`` with the gauge rows of ``gauge``.
        """
        scored = slice(P.FLOW_STEPS - P.SCORED, P.FLOW_STEPS)
        h = self.trunk(b["steps"], b["reach"], self.graph)
        inflow = (0.5 * F.softplus(self.simulation(h).float()[..., 0])) * b["area"][:, None] / 21.6
        delta = 2 * torch.tanh(self.observed(h).float()[..., 0])
        q = inflow * torch.exp(delta)
        out = dict(sim=self.router(inflow)[:, scored], delta=delta)
        out["observed"] = self.router(q)[:, scored]  # observed head at every reach
        routed = self.router(q, gauges_only=True)[:, scored]
        visible, reach = b["visible"], b["gauge_reach"]
        out["routed_gauge"] = routed  # observed head at every gauge, without its own history
        gauge = routed
        if visible.any():
            shown = torch.nonzero(visible)[:, 0]
            lead = slice(P.INNOVATION, P.SCORED)
            basin = self.gauge_rain(b)
            describe = self.gauge_describe(b)
            extra = {}
            if self.head.lstm_inputs:
                extra = dict(weather=self.gauge_weather(b)[shown], basin=SparseProduct.apply(*self.basin_mean, b["static"])[shown])
            correction = self.head(
                b["observed"][P.HISTORY - 120 : P.HISTORY, shown].T,
                routed[shown],
                out["sim"][reach][shown],
                basin[shown],
                describe[shown],
                **extra,
            )
            gauge = routed.clone()
            gauge[shown, lead] = routed[shown, lead] * torch.exp(correction)
        out["gauge"] = gauge
        flags = torch.stack(
            [visible.float(), b["owned"][reach], torch.log1p(gauge[:, -1].detach()) / 5], dim=-1
        )
        if hasattr(self, "tail"):
            out["quantile"] = self.quantiles(b["static"][reach], flags, gauge[:, P.INNOVATION :].detach())
        else:
            features = torch.cat(
                [
                    b["static"][reach][:, None, :].expand(-1, P.HORIZON, -1),
                    self.lead.weight[None].expand(len(reach), -1, -1),
                    flags[:, None, :].expand(-1, P.HORIZON, -1),
                ],
                dim=-1,
            )
            out["sigma"] = F.softplus(self.spread(features)[..., 0]) + 0.02
        if full:
            everywhere = out["observed"].clone()
            everywhere[reach] = gauge
            out["reach"] = everywhere
        return out
