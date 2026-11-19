"""Physics-parameterized network routing (``router="manning"``) and gauge-flow insertion.

The default ``model.Router`` convolves each upstream reach's lateral inflow with one of 27 learned
kernels shared by every pair in a (river distance x lakes) group: travel time is the same for every
river and every flow. ``ManningRouter`` keeps the same mass-conserving form,

    Q_i(t) = sum over upstream j of f_ij (h_ij * q_j)(t),   each h_ij summing to one,

but each pair's kernel follows from the channel between j and i (after Bindas et al. 2024, WRR;
Song et al. 2025, WRR):

1. A small network maps each reach's channel attributes (length, slope, stream order, drainage
   area, lake flag and the 48 reach attributes) to Manning's n, a depth coefficient p and, on lake
   or reservoir reaches, a wave speed (used there in place of Manning).
2. Reach celerity is the kinematic-wave speed c = 5/3 v with Manning's v = d^(2/3) S^(1/2) / n and
   depth d = p Q^0.4 (hydraulic geometry), at the reach's mean flow Q over the sample (the basin sum
   of the lateral inflow being routed, detached): wet periods travel faster than dry ones.
3. Reach travel time t_r = length / c; the travel time of a pair j -> i is the sum
   along the main downstream path, from the middle of j to the end of i (as the distance of the
   default groups).
4. Each pair's travel time falls between two of ``CENTERS`` (log-spaced delays); it splits its share
   between those two delay classes by linear interpolation, so the routing is differentiable in the
   travel time. Every (lake class, delay class) has a kernel with that delay and a learned spread:
   physics sets the delay, the data set the attenuation.

The pattern of the sparse operator moves with the travel times, so it is rebuilt every forward
(one sort of the pair entries, ~0.1 s on the full network) and ``PairProduct`` gives the gradient
with respect to its values (sampled dense products) and to the inflow.

Gauge-flow insertion (``insert``): at visible gauges the model knows better than its own routed
flow: the observed flow before the issue and the gauge-head forecast after it. The difference
(known - routed) is routed down the network from the gauge with the same kernels, so every reach and
every hidden gauge below a visible gauge receives its correction (direct insertion, in the spirit of
HydroNets, Moshe et al. 2020, and data integration, Feng et al. 2020). Mass is conserved: the
correction is water added or removed at the gauge.
"""

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import paths as P

CENTERS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 15.0, 20.0, 26.0, 34.0, 44.0)  # delay classes (steps)
LAKE_CLASSES = 3  # lake reaches on the path: none, 1-10, more than 10 (``prepare.LAKE_LEVELS``)
CHANNEL_INPUTS = 5 + 48  # log length, log slope, stream order, log drainage area, lake flag; reach attributes
SECONDS_PER_STEP = 6 * 3600
DEPTH_EXPONENT = 0.4


def _csr(rows, cols, values, shape, device):
    warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
    coo = torch.sparse_coo_tensor(
        torch.stack([torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long)]),
        torch.as_tensor(values, dtype=torch.float32),
        shape,
    ).coalesce()
    return coo.to_sparse_csr().to(device)


def channel_table(network):
    """Per-reach channel attributes from the NHDPlus flowlines: length (km), slope (-), stream order, lake flag."""
    nodes = pd.read_parquet(
        P.PROCESSED / "graph_candidate/nodes.parquet", columns=["comid", "lengthkm", "slope", "streamorde", "wbareatype"]
    ).set_index("comid").loc[network.ids]
    length = np.maximum(nodes.lengthkm.to_numpy(dtype=np.float64), 0.01)
    slope = nodes.slope.to_numpy(dtype=np.float64)
    bad = ~(slope > 0)  # NHDPlus marks unknown slopes with -9998
    slope[bad] = np.median(slope[~bad])
    slope = np.clip(slope, 1e-5, 0.5)
    order = nodes.streamorde.to_numpy(dtype=np.float64)
    lake = nodes.wbareatype.isin(["LakePond", "Reservoir"]).to_numpy()
    return dict(length=length, slope=slope, order=order, lake=lake)


def main_path(edges, n):
    """Downstream reach on the main path (first listed edge, as ``prepare.routing``), and the pairs (r, k) for every
    reach k strictly below r on it."""
    down = np.full(n, -1)
    down[edges[::-1, 0]] = edges[::-1, 1]
    rows, cols = [], []
    r = np.flatnonzero(down >= 0)
    k = down[r]
    while len(r):
        rows.append(r)
        cols.append(k)
        nxt = down[k]
        keep = nxt >= 0
        r, k = r[keep], nxt[keep]
    return down, np.concatenate(rows), np.concatenate(cols)


class Pattern:
    """Fixed-pattern bookkeeping of a sparse matrix given in COO entries (rows, cols): CSR order and its transpose."""

    def __init__(self, rows, cols, shape):
        n_rows, n_cols = shape
        self.shape = shape
        rows, cols = rows.long(), cols.long()
        self.perm = torch.argsort(rows * n_cols + cols)
        self.crow = torch._convert_indices_from_coo_to_csr(rows[self.perm], n_rows)
        self.col = cols[self.perm]
        self.tperm = torch.argsort(cols * n_rows + rows)
        self.tcrow = torch._convert_indices_from_coo_to_csr(cols[self.tperm], n_cols)
        self.tcol = rows[self.tperm]

    def matrix(self, values):
        return torch.sparse_csr_tensor(self.crow, self.col, values[self.perm], self.shape)

    def transpose(self, values):
        return torch.sparse_csr_tensor(self.tcrow, self.tcol, values[self.tperm], self.shape[::-1])


class PairProduct(torch.autograd.Function):
    """A(values) @ x for a sparse matrix whose values carry a gradient (``Pattern``)."""

    @staticmethod
    def forward(ctx, values, x, pattern):
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
        values, x = values.float().contiguous(), x.float().contiguous()
        ctx.save_for_backward(values, x)
        ctx.pattern = pattern
        return pattern.matrix(values) @ x

    @staticmethod
    def backward(ctx, grad):
        values, x = ctx.saved_tensors
        pattern = ctx.pattern
        grad = grad.float().contiguous()
        grad_values = grad_x = None
        if ctx.needs_input_grad[0]:
            empty = torch.sparse_csr_tensor(pattern.crow, pattern.col, torch.zeros_like(values), pattern.shape)
            sampled = torch.sparse.sampled_addmm(empty, grad, x.T.contiguous(), beta=0.0)
            grad_values = torch.empty_like(values)
            grad_values[pattern.perm] = sampled.values()
        if ctx.needs_input_grad[1]:
            grad_x = pattern.transpose(values) @ grad
        return grad_values, grad_x, None


class ManningRouter(nn.Module):
    """Mass-conserving routing with reach-specific, flow-dependent travel times (module docstring)."""

    def __init__(self, n0=0.04, depth0=0.27, lake0=1.0):
        super().__init__()
        self.classes = len(CENTERS)
        self.groups = LAKE_CLASSES * self.classes
        self.channel = nn.Sequential(nn.Linear(CHANNEL_INPUTS, 64), nn.SiLU(), nn.Linear(64, 3))
        nn.init.zeros_(self.channel[-1].weight)
        nn.init.zeros_(self.channel[-1].bias)
        self.n0, self.depth0, self.lake0 = n0, depth0, lake0
        self.spread = nn.Parameter(torch.full((self.groups,), float(np.log(np.expm1(0.25)))))  # softplus -> 0.25 (+0.1)

    # ---- fixed operators ----------------------------------------------------------------------------------------
    def attach(self, network, gauge_reaches, device, table=None):
        """``table``: channel attributes (``channel_table``), read from the NHDPlus flowlines when not given."""
        n = network.n
        self.n, self.device = n, device
        table = channel_table(network) if table is None else table
        features = np.column_stack(
            [
                np.log(table["length"]),
                np.log(table["slope"]) / 5,
                table["order"] / 8,
                np.log1p(network.basin_area) / 10,
                table["lake"].astype(np.float64),
            ]
        )
        self.features = torch.as_tensor(np.column_stack([features, network.static]), dtype=torch.float32, device=device)
        self.length_m = torch.as_tensor(table["length"] * 1000, dtype=torch.float32, device=device)
        self.sqrt_slope = torch.as_tensor(np.sqrt(table["slope"]), dtype=torch.float32, device=device)
        self.lake = torch.as_tensor(table["lake"], dtype=torch.float32, device=device)
        _, below_r, below_k = main_path(network.edges, n)
        self.below = _csr(below_r, below_k, np.ones(len(below_r)), (n, n), device)  # travel time to the outlet: below @ t
        self.below_t = _csr(below_k, below_r, np.ones(len(below_r)), (n, n), device)
        rows = np.repeat(np.arange(n), np.diff(network.indptr))
        members = network.members.astype(np.int64)
        self.pair_row = torch.as_tensor(rows, dtype=torch.int32, device=device)
        self.pair_col = torch.as_tensor(members, dtype=torch.int32, device=device)
        self.pair_share = torch.as_tensor(network.share, dtype=torch.float32, device=device)
        self.pair_lake = torch.as_tensor(network.group.astype(np.int64) % LAKE_CLASSES, dtype=torch.int32, device=device)
        self.basin_sum = _csr(rows, members, network.share, (n, n), device)  # steady-state flow from lateral inflow
        position = np.full(n, -1)
        position[gauge_reaches] = np.arange(len(gauge_reaches))
        self.gauge_count = len(gauge_reaches)
        at_gauge = position[rows] >= 0
        self.gauge_pairs = torch.as_tensor(np.flatnonzero(at_gauge), dtype=torch.long, device=device)
        self.gauge_pos = torch.as_tensor(position[rows[at_gauge]], dtype=torch.int32, device=device)
        # insertion: pairs from a gauge reach j to every reach i below it (i != j)
        from_gauge = (position[members] >= 0) & (members != rows)
        self.insert_pairs = torch.as_tensor(np.flatnonzero(from_gauge), dtype=torch.long, device=device)
        self.insert_from = torch.as_tensor(position[members[from_gauge]], dtype=torch.int32, device=device)
        self.insert_to_gauge = torch.as_tensor(position[rows[from_gauge]], dtype=torch.int32, device=device)  # -1 off gauges
        self.centers = torch.log1p(torch.as_tensor(CENTERS, dtype=torch.float32, device=device))

    # ---- physics ------------------------------------------------------------------------------------------------
    def channel_parameters(self):
        """Manning's n, depth coefficient p (d = p Q^0.4, m) and lake wave speed (m/s) per reach."""
        o = self.channel(self.features).float()
        n_mann = 0.01 + 0.24 * torch.sigmoid(o[:, 0] + float(np.log((self.n0 - 0.01) / (0.25 - self.n0))))
        depth = self.depth0 * torch.exp(2 * torch.tanh(o[:, 1]))
        lake = self.lake0 * torch.exp(2 * torch.tanh(o[:, 2]))
        return n_mann, depth, lake

    def reach_time(self, q_ref):
        """Travel time through each reach (steps) at flow ``q_ref`` [n] (m3/s). River reaches: kinematic-wave celerity
        from Manning; lake and reservoir reaches (often hundreds of NHD reaches per TVA reservoir, nearly flat): a learned
        wave speed in place of Manning, which would give a crawl at their slope."""
        n_mann, depth_coef, lake_speed = self.channel_parameters()
        depth = depth_coef * (q_ref.clamp_min(0) + 0.01) ** DEPTH_EXPONENT
        velocity = depth ** (2 / 3) * self.sqrt_slope / n_mann
        celerity = torch.where(self.lake > 0, lake_speed, 5 / 3 * velocity).clamp(0.05, 8.0)
        return self.length_m / celerity / SECONDS_PER_STEP

    def pair_time(self, t):
        """Travel time of every pair j -> i (steps), along the main path from the middle of j to the end of i."""
        to_outlet = _MatVec.apply(self.below, self.below_t, t)
        row, col = self.pair_row.long(), self.pair_col.long()
        return (to_outlet[col] + 0.5 * t[col] - to_outlet[row]).clamp_min(0)

    def kernels(self):
        """[group, tap] kernels, group = lake class x delay class, each summing to one."""
        taps = torch.log1p(torch.arange(P.ROUTE, dtype=torch.float32, device=self.spread.device))
        center = self.centers.to(self.spread.device).repeat(LAKE_CLASSES)
        width = F.softplus(self.spread) + 0.1
        logits = -((taps[None] - center[:, None]) ** 2) / (2 * width[:, None] ** 2)
        return torch.softmax(logits, dim=-1)

    def split(self, tau):
        """Delay class below each travel time and the share going to the class above it."""
        x = torch.log1p(tau).clamp(0, float(self.centers[-1]) - 1e-4)
        low = (torch.searchsorted(self.centers, x.detach(), right=True) - 1).clamp(0, self.classes - 2)
        frac = (x - self.centers[low]) / (self.centers[low + 1] - self.centers[low])
        return low, frac.clamp(0, 1)

    # ---- per forward ----------------------------------------------------------------------------------------------
    @torch.autocast("cuda", enabled=False)
    def prepare(self, q_ref):
        """Operators of one forward from the reference flow ``q_ref`` [n] (detached): every reach, gauges only, insertion."""
        t = self.reach_time(q_ref.detach())
        tau = self.pair_time(t)
        low, frac = self.split(tau)
        share = self.pair_share
        lake = self.pair_lake.long()
        ops = {}

        def build(rows_of, idx, out_rows, extra_cols=None):
            lo, fr = low[idx], frac[idx]
            g0 = lake[idx] * self.classes + lo
            rows = torch.cat([g0 * out_rows + rows_of, (g0 + 1) * out_rows + rows_of])
            cols = self.pair_col[idx].long() if extra_cols is None else extra_cols
            cols = torch.cat([cols, cols])
            values = torch.cat([share[idx] * (1 - fr), share[idx] * fr])
            n_cols = self.n if extra_cols is None else self.gauge_count
            with torch.no_grad():
                pattern = Pattern(rows, cols, (self.groups * out_rows, n_cols))
            return values, pattern

        everything = torch.arange(len(share), device=share.device)
        ops["full"] = build(self.pair_row.long(), everything, self.n)
        ops["gauges"] = build(self.gauge_pos.long(), self.gauge_pairs, self.gauge_count)
        ins = self.insert_pairs
        ops["insert"] = build(self.pair_row[ins].long(), ins, self.n, extra_cols=self.insert_from.long())
        on_gauge = self.insert_to_gauge >= 0
        ins_g = ins[on_gauge]
        ops["insert_gauges"] = build(self.insert_to_gauge[on_gauge].long(), ins_g, self.gauge_count,
                                     extra_cols=self.insert_from[on_gauge].long())
        self.ops = ops
        self.last_tau = tau.detach()
        return ops

    @torch.autocast("cuda", enabled=False)
    def reference_flow(self, q):
        """Steady-state flow of lateral inflow ``q`` [n, F]: the basin sum of its mean (m3/s), no gradient."""
        with torch.no_grad():
            return (self.basin_sum @ q.float().mean(dim=1, keepdim=True))[:, 0]

    @torch.autocast("cuda", enabled=False)
    def forward(self, q, gauges_only=False, op=None):
        """Routed flow [rows, F] (float32) from lateral inflow q [n, F] (m3/s), or with ``op`` ("insert",
        "insert_gauges") from inflows at the gauges q [G, F]."""
        if op is None:
            op = "gauges" if gauges_only else "full"
        values, pattern = self.ops[op]
        rows = pattern.shape[0] // self.groups
        u = PairProduct.apply(values, q.float(), pattern)
        u = u.view(self.groups, rows, -1).permute(1, 0, 2)
        out = F.conv1d(F.pad(u, (P.ROUTE - 1, 0)), self.kernels().flip(-1)[:, None, :], groups=self.groups)
        return out.sum(dim=1)


class _MatVec(torch.autograd.Function):
    """Fixed sparse matrix times a vector, gradient through the stored transpose."""

    @staticmethod
    def forward(ctx, a, at, v):
        ctx.at = at
        return (a @ v.float()[:, None])[:, 0]

    @staticmethod
    def backward(ctx, grad):
        return None, None, (ctx.at @ grad.float()[:, None])[:, 0]
