"""The first graph-routing model (V1), for evaluation only.

Identical to ``HydroModel`` (trunk, heads, routing, spread) except at visible gauges: its
residual head is a GRU over the last 8 days of misfit (log observed over routed flow) with a
learned share of the latest misfit carried into every lead, in place of ``GaugeHead``. The
checkpoints in ``models/hydro_v1`` hold exactly these weights. V1 was trained with forecast
rain known to 96 h, so it is scored with ``rain_steps=16`` (``Device.limit_rain``).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import paths as P
from .model import EPS, HydroModel, mlp


class Misfit(nn.Module):
    """Gauge states [g, hidden] from observed and routed flow over [t-32, t)."""

    def __init__(self, hidden=32):
        super().__init__()
        self.encode = nn.GRU(3, hidden, batch_first=True)
        self.out = mlp(hidden + 1, hidden, hidden)

    def forward(self, observed, routed, log_area):
        valid = torch.isfinite(observed)
        ratio = torch.log(torch.nan_to_num(observed) + EPS) - torch.log(routed + EPS)
        ratio = torch.where(valid, ratio.clamp(-5, 5), torch.zeros_like(ratio))
        steps = torch.stack([ratio, valid.float(), torch.log1p(routed) / 5], dim=-1)
        _, h = self.encode(steps)
        return self.out(torch.cat([h[0], log_area[:, None] / 10], dim=-1))


class Residual(nn.Module):
    """Log correction [g, lead] at visible gauges: carried latest misfit + learned part."""

    def __init__(self, hidden=32):
        super().__init__()
        self.misfit = Misfit(hidden)
        self.lead = nn.Embedding(P.HORIZON, hidden)
        self.out = nn.Linear(hidden, 1)
        self.keep = nn.Parameter(3.0 - 0.25 * torch.arange(P.HORIZON, dtype=torch.float32))

    def forward(self, observed, routed, log_area):
        state = self.misfit(observed, routed, log_area)
        learned = 3 * torch.tanh(self.out(F.silu(state[:, None, :] + self.lead.weight[None]))[..., 0])
        latest = torch.isfinite(observed[:, -1])
        misfit = torch.log(torch.nan_to_num(observed[:, -1]) + EPS) - torch.log(routed[:, -1] + EPS)
        misfit = torch.where(latest, misfit.clamp(-5, 5), torch.zeros_like(misfit))
        return torch.sigmoid(self.keep)[None] * misfit[:, None] + learned


class HydroModelV1(HydroModel):
    def __init__(self, groups, delay, hidden=128, channels=32, graph_layers=3):
        super().__init__(groups, delay, hidden, channels, graph_layers)
        del self.head
        self.residual = Residual()

    def forward(self, b, full=False):
        """``sim``, ``gauge`` (visible gauges with the residual), ``delta``, ``sigma`` and with
        ``full`` ``reach`` (the observed head at every reach, gauge rows from ``gauge``)."""
        scored = slice(P.FLOW_STEPS - P.SCORED, P.FLOW_STEPS)
        h = self.trunk(b["steps"], b["reach"], self.graph)
        inflow = (0.5 * F.softplus(self.simulation(h).float()[..., 0])) * b["area"][:, None] / 21.6
        delta = 2 * torch.tanh(self.observed(h).float()[..., 0])
        q = inflow * torch.exp(delta)
        out = dict(sim=self.router(inflow)[:, scored], delta=delta)
        routed = self.router(q, gauges_only=True)[:, scored]
        visible, reach = b["visible"], b["gauge_reach"]
        gauge = routed
        if visible.any():
            shown = torch.nonzero(visible)[:, 0]
            history = b["observed"][P.HISTORY - P.INNOVATION : P.HISTORY, shown].T
            correction = self.residual(history, routed[shown, : P.INNOVATION], torch.log(b["gauge_area"][shown]))
            gauge = routed.clone()
            lead = slice(P.INNOVATION, P.SCORED)
            gauge[shown, lead] = routed[shown, lead] * torch.exp(correction)
        out["gauge"] = gauge
        flags = torch.stack([visible.float(), b["owned"][reach], torch.log1p(gauge[:, -1].detach()) / 5], dim=-1)
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
            everywhere = self.router(q)[:, scored]
            everywhere[reach] = gauge
            out["reach"] = everywhere
        return out
