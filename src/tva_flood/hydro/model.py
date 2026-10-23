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


class BasinPool(nn.Module):
    """h + MLP([h, means of h over each reach's upstream basin in near, middle and far
    river-distance bands]) at every step: what the whole basin above is doing, beyond the
    few hops of a ``GraphBlock``. Means are area-weighted (``Pooling``) over a narrow
    projection of h, so the sparse products stay cheap."""

    def __init__(self, channels, width=8, bands=3):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.project = nn.Linear(channels, width)
        self.mix = mlp(channels + bands * width, channels, channels)

    def forward(self, h, pooling):
        x = self.norm(h)
        with torch.autocast("cuda", enabled=False):
            z = self.project(x.float())
            n, steps, width = z.shape
            flat = z.reshape(n, steps * width).contiguous()
            parts = [SparseProduct.apply(a, at, flat).view(n, steps, width) for a, at in pooling.bands]
        return h + self.mix(torch.cat([x] + [part.to(x.dtype) for part in parts], dim=-1))


class Pooling:
    """Area-weighted mean over each reach's upstream basin (itself included) in three
    river-distance bands (the routing distance bins in thirds); empty bands give zero."""

    def __init__(self, network, levels, device, bands=3):
        rows = np.repeat(np.arange(network.n), np.diff(network.indptr))
        bins = network.groups // levels
        band = (network.group.astype(np.int64) // levels) * bands // bins
        weight = network.share * network.area[network.members]
        total = np.bincount(band * network.n + rows, weights=weight, minlength=bands * network.n)
        weight = weight / total[band * network.n + rows]
        self.bands = []
        for b in range(bands):
            keep = band == b
            self.bands.append(csr(rows[keep], network.members[keep], weight[keep], (network.n, network.n), device))


class Trunk(nn.Module):
    """Reach states [n, F, channels] from step inputs [n, T, STEP_INPUTS] and reach inputs:
    an LSTM per reach, then ``pool_layers`` basin-pooling and ``graph_layers`` local layers."""

    def __init__(self, hidden, channels, graph_layers, chunk=15000, pool_layers=0):
        super().__init__()
        self.inputs = nn.Linear(STEP_INPUTS, hidden)
        self.reach = mlp(REACH_INPUTS, hidden, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        with torch.no_grad():  # forget gate open at the start: long storage memory
            self.lstm.bias_hh_l0[hidden : 2 * hidden] = 3.0
        self.down = nn.Linear(hidden, channels)
        self.pools = nn.ModuleList(BasinPool(channels) for _ in range(pool_layers))
        self.blocks = nn.ModuleList(GraphBlock(channels) for _ in range(graph_layers))
        self.chunk = chunk

    def _time(self, steps, reach):
        x = self.inputs(steps) + self.reach(reach)[:, None, :]
        h, _ = self.lstm(x)
        return self.down(h[:, -P.FLOW_STEPS :])

    def forward(self, steps, reach, graph, pooling=None):
        parts = []
        for s, r in zip(steps.split(self.chunk), reach.split(self.chunk)):
            if self.training:
                parts.append(checkpoint(self._time, s, r, use_reentrant=False))
            else:
                parts.append(self._time(s, r))
        h = torch.cat(parts)
        for pool in self.pools:
            h = checkpoint(pool, h, pooling, use_reentrant=False) if self.training else pool(h, pooling)
        for block in self.blocks:
            h = checkpoint(block, h, graph, use_reentrant=False) if self.training else block(h, graph)
        return h


class Router(nn.Module):
    """Mass-conserving routing of lateral inflow with one travel-time kernel per group.

    With dams (``attach(..., dams=...)``) the river is split at every dam: water from a
    reach stops at the first dam below it, which passes on its release instead, routed
    down the river to the next dam with one kernel per distance bin (``release`` kernels,
    no reservoir delay: the release is what leaves the reservoir). ``inflow`` gives each
    reservoir what reaches it: its own basin's routed inflow plus the releases of the dams
    directly above. Routing without releases (``release=None``) is the natural river.
    """

    def __init__(self, groups, delay, levels=3, dams=False):
        super().__init__()
        taps = torch.arange(P.ROUTE, dtype=torch.float32)
        center = torch.as_tensor(delay, dtype=torch.float32).clamp(max=P.ROUTE - 1)[:, None]
        self.logits = nn.Parameter(-((torch.log1p(taps)[None] - torch.log1p(center)) ** 2) / (2 * 0.35**2))
        self.groups, self.levels = groups, levels
        self.bins = groups // levels  # river-distance bins
        if dams:  # releases start as the lake-free runoff kernels of each distance bin
            self.release_logits = nn.Parameter(self.logits.detach()[::levels].clone())

    def attach(self, network, gauge_reaches, device, dams=None):
        """Sparse operators: all reaches (rows group * n + i) and gauge reaches only; with
        ``dams`` (``data.Dams``) also the split river, the releases and reservoir inflows."""
        n = network.n
        rows = np.repeat(np.arange(n), np.diff(network.indptr))
        cols, share, group = network.members, network.share, network.group.astype(np.int64)
        self.n, self.gauge_count = n, len(gauge_reaches)
        position = np.full(n, -1)
        position[gauge_reaches] = np.arange(len(gauge_reaches))
        at_gauge = position[rows] >= 0

        def operator(keep, kind, sources, rows_=rows, cols_=cols):
            """CSR of the pairs ``keep``: rows (group or bin) x reach (or gauge), columns ``sources``."""
            g = group[keep] if kind == "group" else group[keep] // self.levels
            width = self.groups if kind == "group" else self.bins
            full = csr(g * n + rows_[keep], cols_[keep], share[keep], (width * n, sources), device)
            k = keep & at_gauge
            g = group[k] if kind == "group" else group[k] // self.levels
            gauges = csr(g * self.gauge_count + position[rows_[k]], cols_[k], share[k], (width * self.gauge_count, sources), device)
            return full, gauges

        everything = np.ones(len(rows), dtype=bool)
        self.full, self.gauges = operator(everything, "group", n)
        if dams is None:
            return
        d = dams.count
        is_dam = np.full(n, -1)
        is_dam[dams.reach] = np.arange(d)
        below = np.zeros((d, n), dtype=bool)  # reach i at or below dam k
        hit = is_dam[cols] >= 0
        below[is_dam[cols[hit]], rows[hit]] = True
        first = dams.dam_below[cols]  # the first dam on reach j's way down
        passes = (first >= 0) & below[np.maximum(first, 0), rows]
        self.split, self.split_gauges = operator(~passes, "group", n)
        # A dam's release reaches i unless the next dam down lies between them.
        source = is_dam[cols]
        nxt = dams.downstream[np.maximum(source, 0)]
        release = (source >= 0) & ~((nxt >= 0) & below[np.maximum(nxt, 0), rows])
        self.release, self.release_gauges = operator(release, "bin", d, cols_=np.maximum(source, 0))
        # Reservoir inflow: rows are dams (their reach), from their own basin and the dams above.
        dam_row = is_dam[rows]
        own = (dam_row >= 0) & (first == dam_row)
        above = (dam_row >= 0) & (source >= 0) & (nxt == dam_row) & (source != dam_row)
        self.inflow_runoff = csr(group[own] * d + dam_row[own], cols[own], share[own], (self.groups * d, n), device)
        self.inflow_release = csr(
            group[above] // self.levels * d + dam_row[above], source[above], share[above], (self.bins * d, d), device
        )

    def kernels(self):
        return torch.softmax(self.logits, dim=-1)  # [group, tap], each sums to one

    @torch.autocast("cuda", enabled=False)
    def _route(self, op, x, width, logits):
        a, at = op
        u = SparseProduct.apply(a, at, x.float().contiguous())
        rows = a.shape[0] // width
        u = u.view(width, rows, -1).permute(1, 0, 2)
        kernel = torch.softmax(logits, dim=-1).flip(-1)[:, None, :]
        return F.conv1d(F.pad(u, (P.ROUTE - 1, 0)), kernel, groups=width).sum(dim=1)

    def forward(self, q, gauges_only=False, release=None):
        """Routed flow [rows, F] (float32) from lateral inflow q [n, F] (m3/s); with
        ``release`` [dam, F] the river split at the dams, which pass on their release."""
        if release is None:
            return self._route(self.gauges if gauges_only else self.full, q, self.groups, self.logits)
        runoff = self._route(self.split_gauges if gauges_only else self.split, q, self.groups, self.logits)
        passed = self._route(self.release_gauges if gauges_only else self.release, release, self.bins, self.release_logits)
        return runoff + passed

    def inflow(self, q, release):
        """Inflow [dam, F] of each reservoir: its basin's routed runoff and the releases above."""
        return self._route(self.inflow_runoff, q, self.groups, self.logits) + self._route(
            self.inflow_release, release, self.bins, self.release_logits
        )

    def release_kernels(self):
        return torch.softmax(self.release_logits, dim=-1)  # [distance bin, tap]


class Corrections:
    """A visible gauge's correction (observed, or the gauge forecast, minus the routed flow)
    carried down its main path to the next visible gauge or dam, with the release kernels.

    Each gauge's path runs from the reach below it to the reach above the next dam (whose
    release replaces the river) or the outlet. Along it, the next visible gauge takes over:
    its own correction already includes what came from above. Built once (no parameters).
    """

    def __init__(self, network, gauge_reaches, dams, levels, device):
        import scipy.sparse as sp

        n = network.n
        below = {}
        for s, d in network.edges:  # the main path: the larger basin downstream
            if s not in below or network.basin_area[d] > network.basin_area[below[s]]:
                below[s] = d
        stop = np.zeros(n, dtype=bool)
        if dams is not None:
            stop[dams.reach] = True
        on = {int(r): g for g, r in enumerate(gauge_reaches)}
        pair_index = sp.csr_matrix(
            (np.arange(1, len(network.members) + 1), network.members, network.indptr), shape=(n, n)
        )
        rows, gauges, positions, blockers = [], [], [], []
        for g, r in enumerate(gauge_reaches):
            path, i, block = [], int(r), []
            while (i := below.get(i)) is not None and not stop[i]:
                if i in on:
                    block.append((on[i], len(path)))
                path.append(i)
            rows += path
            gauges += [g] * len(path)
            positions += list(range(len(path)))
            blockers.append(block)
        rows, gauges = np.array(rows, dtype=np.int64), np.array(gauges, dtype=np.int64)
        index = np.asarray(pair_index[rows, np.asarray(gauge_reaches)[gauges]]).ravel() - 1
        if (index < 0).any():
            raise ValueError("A gauge is not upstream of a reach on its own path")
        self.n, self.levels, self.blockers = n, levels, blockers
        self.rows = torch.as_tensor(rows, device=device)
        self.gauge = torch.as_tensor(gauges, device=device)
        self.position = torch.as_tensor(np.array(positions), device=device)
        self.bin = torch.as_tensor(network.group[index].astype(np.int64) // levels, device=device)
        self.share = torch.as_tensor(network.share[index], device=device)

    @torch.autocast("cuda", enabled=False)
    def __call__(self, correction, visible, kernels):
        """Flow change [n, S] from corrections [G, S] at the ``visible`` gauges [G] (bool)."""
        shown = visible.cpu().numpy()
        cutoff = torch.full((len(self.blockers),), 1 << 30, device=correction.device)
        for g, block in enumerate(self.blockers):
            later = [pos for other, pos in block if shown[other]]
            if later:
                cutoff[g] = later[0]
        active = visible[self.gauge] & (self.position < cutoff[self.gauge])
        out = torch.zeros(self.n, correction.shape[1], device=correction.device)
        if not active.any():
            return out
        gauge, rows, bins = self.gauge[active], self.rows[active], self.bin[active]
        x = correction.float()[gauge] * self.share[active, None]
        for b in torch.unique(bins).tolist():
            sel = bins == b
            kernel = kernels[b].float().flip(-1)[None, None, :]
            y = F.conv1d(F.pad(x[sel][:, None, :], (P.ROUTE - 1, 0)), kernel)[:, 0]
            out.index_add_(0, rows[sel], y)
        return out


class GaugeHead(nn.Module):
    """Direct forecast [g, lead] at visible gauges from their history and the routed flow.

    Encoder: an LSTM over the last 30 days at the gauge: observed log flow (and whether it
    exists), and over the last 8 days the routed observed-head and simulated flow there.
    Decoder: an LSTM over the 40 leads, started from the encoder state, reading the routed
    forecast of both heads, the basin-mean forecast rain and a lead embedding. Gauge
    descriptors (attributes, log basin area, share of the basin behind lakes) enter both.
    Output: the routed observed-head flow x exp(correction), where the correction starts as
    a learned share of the latest misfit, fading with lead, so short leads begin as
    corrected persistence and later leads as the routed forecast.
    """

    def __init__(self, hidden=64, statics=48, weather=0):
        super().__init__()
        self.weather = weather  # basin-mean weather and validity channels (0: none)
        self.describe = mlp(statics + 2, hidden, hidden)
        self.encode = nn.LSTM(4 + weather, hidden, batch_first=True)
        self.decode = nn.LSTM(4 + 16, hidden, batch_first=True)
        self.lead = nn.Embedding(P.HORIZON, 16)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.keep = nn.Parameter(3.0 - 0.25 * torch.arange(P.HORIZON, dtype=torch.float32))

    @torch.autocast("cuda", enabled=False)
    def forward(self, history, routed, simulated, rain, describe, weather=None):
        """history [g, H] observed flow (NaN missing) before t; routed, simulated [g, S]
        (scored window); rain [g, lead, 2] basin-mean forecast rain and known share;
        describe [g, statics + 2]; weather [g, H, channels] basin-mean weather history (when
        the head reads it). Returns log corrections [g, lead]."""
        g = history.shape[0]
        before = P.INNOVATION
        valid = torch.isfinite(history)
        obs = torch.where(valid, torch.log1p(torch.nan_to_num(history).clamp_min(0)) / 5, 0.0)
        tail = torch.zeros(g, history.shape[1], 2, device=history.device)
        tail[:, -before:, 0] = torch.log1p(routed[:, :before]) / 5
        tail[:, -before:, 1] = torch.log1p(simulated[:, :before]) / 5
        state = self.describe(describe.float())
        h0 = state[None].contiguous()
        inputs = [obs[..., None], valid[..., None].float(), tail] + ([weather.float()] if self.weather else [])
        _, (h, c) = self.encode(torch.cat(inputs, dim=-1), (h0, torch.zeros_like(h0)))
        future = torch.cat(
            [
                torch.log1p(routed[:, before:])[..., None] / 5,
                torch.log1p(simulated[:, before:])[..., None] / 5,
                rain.float(),
                self.lead.weight[None].expand(g, -1, -1),
            ],
            dim=-1,
        )
        decoded, _ = self.decode(future, (h, c))
        learned = 3 * torch.tanh(self.out(decoded)[..., 0])
        latest = valid[:, -1]
        misfit = torch.log(torch.nan_to_num(history[:, -1]).clamp_min(0) + EPS) - torch.log(routed[:, before - 1] + EPS)
        misfit = torch.where(latest, misfit.clamp(-5, 5), torch.zeros_like(misfit))
        return torch.sigmoid(self.keep)[None] * misfit[:, None] + learned


class HydroModel(nn.Module):
    def __init__(self, groups, delay, hidden=128, channels=32, graph_layers=3):
        super().__init__()
        self.config = dict(
            groups=groups,
            delay=[float(d) for d in delay],
            hidden=hidden,
            channels=channels,
            graph_layers=graph_layers,
        )
        self.trunk = Trunk(hidden, channels, graph_layers)
        self.simulation = nn.Linear(channels, 1)
        self.observed = nn.Linear(channels, 1)
        nn.init.zeros_(self.observed.weight)
        nn.init.zeros_(self.observed.bias)
        self.router = Router(groups, delay)
        self.head = GaugeHead()
        self.lead = nn.Embedding(P.HORIZON, 16)
        self.spread = mlp(48 + 16 + 3, 64, 1)

    def attach(self, network, edges, gauge_reaches, device, dams=None):
        """Fixed operators of the river graph on the device (not parameters): routing, graph,
        and for each gauge its basin-mean weights and the share of its basin behind lakes."""
        self.router.attach(network, gauge_reaches, device, dams)
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

    def forward(self, b, full=False):
        """Outputs over the scored window [t-32, t+40) (``paths.SCORED`` steps).

        ``sim`` [n, S]: simulation head at every reach; ``observed`` [n, S]: observed head at
        every reach; ``routed_gauge`` [G, S]: observed head at every gauge; ``gauge`` [G, S]:
        the forecast at every gauge (visible ones from the gauge head); ``delta`` [n, F]: observed-head log correction; ``sigma`` [G, lead]; with
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
            future = slice(P.HISTORY, P.WINDOW)
            rain = torch.stack([b["weather"][:, future, 0], b["valid"][:, future, 0]], dim=-1).float()
            basin = SparseProduct.apply(*self.basin_mean, rain.reshape(rain.shape[0], -1)).view(-1, P.HORIZON, 2)
            describe = torch.cat(
                [b["static"][reach], torch.log(b["gauge_area"])[:, None] / 10, self.lake_share[:, None]], dim=-1
            )
            correction = self.head(
                b["observed"][P.HISTORY - 120 : P.HISTORY, shown].T,
                routed[shown],
                out["sim"][reach][shown],
                basin[shown],
                describe[shown],
            )
            gauge = routed.clone()
            gauge[shown, lead] = routed[shown, lead] * torch.exp(correction)
        out["gauge"] = gauge
        flags = torch.stack(
            [visible.float(), b["owned"][reach], torch.log1p(gauge[:, -1].detach()) / 5], dim=-1
        )
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


class Reservoir(nn.Module):
    """Release forecast [dam, lead] at each dam, as TVA's operators decide it.

    Encoder: a GRU over the last 20 days at the dam: observed release, TVA's observed inflow
    (each with whether it exists) and the model's routed inflow; started from the dam (an
    embedding, its mean release) and its storage change over 30 and 90 days. Decoder: a GRU
    over the leads reading the routed inflow forecast, season, the lead and, on the plan
    track, TVA's forecast release (and whether there is one). Output: the latest observed
    release x exp(change), the change bounded (x20) and starting at zero (persistence).
    """

    def __init__(self, dams, hidden=64, plan=False):
        super().__init__()
        self.plan = plan
        self.dam = nn.Embedding(dams, 16)
        self.describe = mlp(16 + 2 + 1, hidden, hidden)
        self.encode = nn.GRU(5, hidden, batch_first=True)
        self.lead = nn.Embedding(P.HORIZON, 16)
        self.decode = nn.GRU(1 + 2 + 16 + (2 if plan else 0), hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @torch.autocast("cuda", enabled=False)
    def forward(self, dams, release, observed_inflow, inflow, future_inflow, season, storage, mean, plan=None):
        """dams [d] indices; release, observed_inflow [d, H] (NaN missing), inflow [d, H] and
        future_inflow [d, lead] routed (m3/s); season [lead, 2]; storage [d, 2]; mean [d]
        (m3/s); plan [d, lead] TVA's forecast release (NaN missing). Returns [d, lead] m3/s."""
        eps = 0.01 * mean[:, None]
        scale = lambda x: torch.log((torch.nan_to_num(x).clamp_min(0) + eps) / mean[:, None]) / 2  # noqa: E731
        seen, known = torch.isfinite(release), torch.isfinite(observed_inflow)
        steps = torch.stack(
            [scale(release) * seen, seen.float(), scale(observed_inflow) * known, known.float(), scale(inflow)], dim=-1
        )
        state = self.describe(torch.cat([self.dam(dams), storage.clamp(-5, 5), torch.log(mean)[:, None] / 5], dim=-1))
        _, h = self.encode(steps.float(), state[None].contiguous())
        lead = self.lead.weight[None].expand(len(dams), -1, -1)
        future = [scale(future_inflow)[..., None], season[None].expand(len(dams), -1, -1), lead]
        if self.plan:
            given = torch.isfinite(plan)
            future += [(scale(plan) * given)[..., None], given[..., None].float()]
        decoded, _ = self.decode(torch.cat(future, dim=-1).float(), h)
        # The latest observed release (the mean where the record is empty).
        last_seen = torch.where(seen, torch.arange(seen.shape[1], device=seen.device), -1).max(dim=1).values
        latest = torch.where(last_seen >= 0, release.gather(1, last_seen.clamp(min=0)[:, None])[:, 0], mean)
        change = 3 * torch.tanh(self.out(decoded)[..., 0])
        return ((latest.clamp_min(0)[:, None] + eps) * torch.exp(change) - eps).clamp_min(0)


class RegulatedModel(HydroModel):
    """V3: the graph-routing model with the TVA dams.

    As ``HydroModel``, plus: one basin-pooling layer in the trunk (``BasinPool``); the river
    split at every dam, which passes on its release: observed before the issue (the model's
    reservoir inflow where the record has a gap), forecast after it by the ``Reservoir``
    module, dams above first; the gauge head reading the LSTM's inputs too (basin-mean
    weather and attributes); and each visible gauge's correction carried down to the next
    visible gauge or dam (``Corrections``). The simulation head is routed along the natural
    river (the NWM prior).
    """

    def __init__(self, groups, delay, dams, hidden=128, channels=32, graph_layers=2, pool_layers=1, plan=False):
        super().__init__(groups, delay, hidden, channels, graph_layers)
        self.config.update(kind="regulated", dams=dams, pool_layers=pool_layers, plan=plan)
        self.trunk = Trunk(hidden, channels, graph_layers, pool_layers=pool_layers)
        self.router = Router(groups, delay, dams=True)
        self.head = GaugeHead(statics=48 + 48, weather=16)
        self.reservoir = Reservoir(dams, plan=plan)
        self.natural = False  # diagnostic: route the natural river, ignoring the dams

    def attach(self, network, edges, gauge_reaches, device, dams=None):
        super().attach(network, edges, gauge_reaches, device, dams)
        self.pooling = Pooling(network, self.router.levels, device)
        self.corrections = Corrections(network, gauge_reaches, dams, self.router.levels, device)
        static = torch.as_tensor(network.static, dtype=torch.float32, device=device)
        self.gauge_static = SparseProduct.apply(*self.basin_mean, static)  # [G, 48] basin means
        self.levels = [torch.as_tensor(np.flatnonzero(dams.level == k), device=device) for k in range(dams.level.max() + 1)]

    def gauge_describe(self, b):
        """Gauge descriptors [G, 98]: attributes at the gauge and over its basin, log basin
        area, share of the basin behind lakes (``b`` needs only the fixed tensors)."""
        return torch.cat(
            [b["static"][b["gauge_reach"]], self.gauge_static, torch.log(b["gauge_area"])[:, None] / 10, self.lake_share[:, None]],
            dim=-1,
        )

    def gauge_inputs(self, b):
        """What the gauge head reads besides flow and descriptors, at every gauge: basin-mean
        forecast rain and whether it is known [G, lead, 2], and basin-mean weather and validity
        over the last 30 days [G, 120, 16] (the LSTM's inputs)."""
        future = slice(P.HISTORY, P.WINDOW)
        rain = torch.stack([b["weather"][:, future, 0], b["valid"][:, future, 0]], dim=-1).float()
        rain = SparseProduct.apply(*self.basin_mean, rain.reshape(rain.shape[0], -1)).view(-1, P.HORIZON, 2)
        past = slice(P.HISTORY - 120, P.HISTORY)
        weather = torch.cat([b["weather"][:, past], b["valid"][:, past]], dim=-1).float()
        weather = SparseProduct.apply(*self.basin_mean, weather.reshape(weather.shape[0], -1)).view(-1, 120, 16)
        return rain, weather

    def forward(self, b, full=False):
        """As ``HydroModel.forward``, plus ``release`` [dam, lead] (forecast releases),
        ``inflow`` [dam, S] (each reservoir's routed inflow) and ``routed`` [G, S] (the routed
        flow at the gauges, what the gauge head corrects); ``routed_gauge`` includes the
        corrections carried down from visible gauges above."""
        scored = slice(P.FLOW_STEPS - P.SCORED, P.FLOW_STEPS)
        before = P.FLOW_STEPS - P.HORIZON  # flow steps before the issue
        h = self.trunk(b["steps"], b["reach"], self.graph, self.pooling)
        inflow = (0.5 * F.softplus(self.simulation(h).float()[..., 0])) * b["area"][:, None] / 21.6
        delta = 2 * torch.tanh(self.observed(h).float()[..., 0])
        q = inflow * torch.exp(delta)
        out = dict(sim=self.router(inflow)[:, scored], delta=delta)

        # Releases over the flow window: observed before the issue, forecast after it.
        window = slice(P.WINDOW - P.FLOW_STEPS, P.WINDOW)
        record = b["release"][window].T.float()  # [dam, F]
        history = torch.arange(P.FLOW_STEPS, device=q.device) < before
        seen = torch.isfinite(record) & history[None]
        release = torch.where(seen, record.clamp_min(0), 0.0)
        release = torch.where(history[None] & ~seen, self.router.inflow(q, release), release)  # gaps: pass-through
        observed_inflow = b["dam_inflow"][window].T.float()[:, :before]
        future_season = b["season"][:, P.HISTORY :].T.float()
        storage, mean = b["storage"].T.float(), b["dam_mean_raw"]
        plan = b["plan"].T.float() if self.config["plan"] else None
        lead_mask = ~history
        for dams in self.levels:  # dams above first: their releases reach the dams below
            arriving = self.router.inflow(q, release)[dams]
            forecast = self.reservoir(
                dams,
                torch.where(seen[dams, :before], record[dams, :before], torch.nan),
                observed_inflow[dams],
                arriving[:, :before],
                arriving[:, before:],
                future_season,
                storage[dams],
                mean[dams],
                None if plan is None else plan[dams],
            )
            update = torch.zeros_like(release)
            update[dams] = torch.cat([torch.zeros(len(dams), before, device=q.device), forecast], dim=1)
            chosen = torch.zeros(release.shape[0], dtype=torch.bool, device=q.device)
            chosen[dams] = True
            release = torch.where(chosen[:, None] & lead_mask[None], update, release)
        out["release"] = release[:, before:]
        out["inflow"] = self.router.inflow(q, release)[:, scored]
        route = None if self.natural else release
        out["observed"] = self.router(q, release=route)[:, scored]
        routed = self.router(q, gauges_only=True, release=route)[:, scored]
        out["routed"] = routed

        visible, reach = b["visible"], b["gauge_reach"]
        lead = slice(P.INNOVATION, P.SCORED)
        gauge = routed
        if visible.any():
            shown = torch.nonzero(visible)[:, 0]
            basin, weather = self.gauge_inputs(b)
            describe = self.gauge_describe(b)
            correction = self.head(
                b["observed"][P.HISTORY - 120 : P.HISTORY, shown].T,
                routed[shown],
                out["sim"][reach][shown],
                basin[shown],
                describe[shown],
                weather[shown],
            )
            gauge = routed.clone()
            gauge[shown, lead] = routed[shown, lead] * torch.exp(correction)
        # Corrections at visible gauges: observed minus routed before the issue, the gauge
        # forecast minus routed after it; carried down to the next visible gauge or dam.
        recent = b["observed"][P.HISTORY - P.INNOVATION : P.HISTORY].T.float()
        before_issue = torch.where(torch.isfinite(recent), recent - routed[:, : P.INNOVATION], 0.0)
        change = torch.cat([before_issue, gauge[:, lead] - routed[:, lead]], dim=1) * visible[:, None]
        carried = self.corrections(change, visible.bool(), self.router.release_kernels())
        out["routed_gauge"] = (routed + carried[reach]).clamp_min(0)
        out["gauge"] = torch.where(visible.bool()[:, None], gauge, out["routed_gauge"])
        flags = torch.stack([visible.float(), b["owned"][reach], torch.log1p(out["gauge"][:, -1].detach()) / 5], dim=-1)
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
            everywhere = (out["observed"] + carried).clamp_min(0)
            everywhere[reach] = out["gauge"]
            out["reach"] = everywhere
        return out
