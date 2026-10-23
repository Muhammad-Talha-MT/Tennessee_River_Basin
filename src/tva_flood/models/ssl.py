"""Causal temporal encoders for masked flow reconstruction and later forecasting.

Inputs use the existing normalized dynamic values and explicit availability
masks. Spatial supports are receiver-by-sender sparse matrices. Encoder weights
are independent of graph size; supports are nonpersistent buffers, not learned
weights. Distributed callers must set ``broadcast_buffers=False`` because the
graph buffers are sparse and already identical on each rank.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class _TemporalBlock(nn.Module):
    """Depthwise causal time convolution followed by channel mixing."""

    def __init__(self, hidden, dilation, dropout):
        super().__init__()
        self.left_padding = 2 * dilation
        self.norm = nn.LayerNorm(hidden)
        self.depthwise = nn.Conv2d(hidden, hidden, (1, 3), dilation=(1, dilation), groups=hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        # The only padded axis is time; no future timestep enters the kernel.
        normalized = self.norm(value).permute(0, 3, 2, 1)
        filtered = self.depthwise(F.pad(normalized, (self.left_padding, 0)))
        mixed = self.mix(F.silu(filtered.permute(0, 3, 2, 1)))
        return value + self.dropout(mixed)


def _propagate(value, support):
    """Apply a sparse spatial support independently at every batch/time index."""
    batch, steps, nodes, channels = value.shape
    flat = value.permute(2, 0, 1, 3).reshape(nodes, batch * steps * channels)
    # Sparse kernels need matching dtypes. Keep accumulation in float32 under
    # autocast; dense temporal/channel projections can still use lower precision.
    with torch.autocast(device_type=value.device.type, enabled=False):
        sparse_input = flat.float() if flat.dtype in (torch.float16, torch.bfloat16) else flat
        result = torch.sparse.mm(support.to(dtype=sparse_input.dtype), sparse_input)
    return result.to(value.dtype).reshape(nodes, batch, steps, channels).permute(1, 2, 0, 3)


def _basin(value, routing):
    """Area-weighted basin mean over each reach's whole upstream basin: ``[..., node, C]``."""
    return sum(_propagate(value, part) for part in routing.parts)


def _attend(query, key, value, support, heads):
    """Softmax attention of each receiver over its senders in ``support`` (self loop included).

    Multi-head dot product per edge; the softmax runs over one receiver's senders,
    so the result depends only on those senders, as with the mean.
    """
    batch, steps, nodes, channels = query.shape
    size = channels // heads
    receiver, sender = support.coalesce().indices()
    q = query.view(batch, steps, nodes, heads, size)[:, :, receiver]
    k = key.view(batch, steps, nodes, heads, size)[:, :, sender]
    v = value.view(batch, steps, nodes, heads, size)[:, :, sender]
    score = (q * k).sum(-1).float() / size**0.5  # [batch, steps, edge, head]
    index = receiver.view(1, 1, -1, 1).expand_as(score)
    peak = torch.full(
        (batch, steps, nodes, heads), -torch.inf, device=score.device, dtype=score.dtype
    ).scatter_reduce(2, index, score, "amax", include_self=True)
    weight = torch.exp(score - peak.detach().gather(2, index))
    total = torch.zeros_like(peak).index_add(2, receiver, weight)
    message = torch.zeros(
        batch, steps, nodes, heads, size, device=value.device, dtype=weight.dtype
    ).index_add(2, receiver, weight[..., None] * v.to(weight.dtype))
    return (message / total.clamp_min(1e-12)[..., None]).to(value.dtype).flatten(-2)


class _GraphBlock(nn.Module):
    """One spatial hop along the river: upstream senders, and downstream ones unless
    ``direction="upstream"``. Senders are averaged (``aggregation="mean"``, weights
    from the support) or attended to (``"attention"``, learned per edge)."""

    def __init__(self, hidden, dropout, direction="both", aggregation="mean", heads=4):
        super().__init__()
        if direction not in ("both", "upstream") or aggregation not in ("mean", "attention"):
            raise ValueError("direction is both|upstream; aggregation is mean|attention")
        if aggregation == "attention" and hidden % heads:
            raise ValueError("Attention needs hidden divisible by the number of heads")
        self.direction, self.aggregation, self.heads = direction, aggregation, heads
        self.norm = nn.LayerNorm(hidden)
        self.self_mix = nn.Linear(hidden, hidden)
        self.forward_mix = nn.Linear(hidden, hidden, bias=False)
        if direction == "both":
            self.reverse_mix = nn.Linear(hidden, hidden, bias=False)
        if aggregation == "attention":
            self.query = nn.Linear(hidden, hidden, bias=False)
            self.key = nn.Linear(hidden, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _gather(self, normalized, support):
        if self.aggregation == "mean":
            return _propagate(normalized, support)
        query, key = self.query(normalized), self.key(normalized)
        return _attend(query, key, normalized, support, self.heads)

    def forward(self, value, forward_support, reverse_support):
        normalized = self.norm(value)
        mixed = self.self_mix(normalized)
        mixed = mixed + self.forward_mix(self._gather(normalized, forward_support))
        if self.direction == "both":
            mixed = mixed + self.reverse_mix(self._gather(normalized, reverse_support))
        return value + self.dropout(F.silu(mixed))


class _RoutingBlock(nn.Module):
    """Route every upstream reach's state to each reach with a learned travel time.

    ``routing`` holds one basin-mean operator per river-distance bin (rows sum to
    one over all bins). For each bin the routed states pass through a causal
    convolution along the lead axis (a travel-time kernel per channel, started at
    no delay) and a linear map that starts at zero, so a new model begins as if
    there were no routing.
    """

    def __init__(self, hidden, bins, lags):
        super().__init__()
        self.lags = lags
        self.norm = nn.LayerNorm(hidden)
        kernel = torch.zeros(bins, hidden, 1, lags + 1)
        kernel[..., -1] = 1.0
        self.kernel = nn.Parameter(kernel)
        self.projections = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(bins))
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, value, routing):
        if len(routing) != len(self.projections):
            raise ValueError("Routing operator has a different number of distance bins")
        batch, leads, nodes, channels = value.shape
        normalized = self.norm(value)
        routed = 0
        for b, part in enumerate(routing.parts):
            gathered = _propagate(normalized, part)  # [batch, lead, node, channel]
            series = gathered.permute(0, 2, 3, 1).reshape(batch * nodes, channels, leads)
            series = F.conv1d(F.pad(series, (self.lags, 0)), self.kernel[b], groups=channels)
            series = series.reshape(batch, nodes, channels, leads).permute(0, 3, 1, 2)
            routed = routed + self.projections[b](series)
        return value + routed


class TemporalGraphEncoder(nn.Module):
    """Encode ``[batch, time, node, 18]`` into a causal latent sequence.

    Static inputs have shape ``[node, 48]`` and are projected once before
    broadcasting across time. Six temporal layers cover 127 six-hour steps;
    subsequent graph blocks do not mix timesteps. Every LayerNorm operates on
    channels of one node at one timestep, including during training.

    ``checkpoint_blocks=True`` recomputes block activations during backward to
    reduce memory use. Nonreentrant checkpointing supports DDP and preserves
    dropout RNG state when recomputing.
    """

    def __init__(
        self,
        supports,
        hidden=16,
        temporal_layers=6,
        graph_layers=2,
        dropout=0.0,
        checkpoint_blocks=False,
        direction="both",
        aggregation="mean",
        basin_inputs=False,
    ):
        super().__init__()
        if basin_inputs and direction != "upstream":
            raise ValueError("Basin inputs need closed patches: direction='upstream'")
        if hidden < 2 or temporal_layers < 1 or graph_layers < 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid encoder dimensions or dropout")
        if len(supports) != 2:
            raise ValueError("Expected forward and reverse sparse graph supports")
        self.direction, self.aggregation, self.basin_inputs = direction, aggregation, basin_inputs
        # Full-network routing operator (data.river.Routing), set by callers that
        # run on the whole graph; patches pass their own as supports[2].
        self.full_routing = None
        nodes = supports[0].shape[0]
        if nodes < 1 or any(
            support.layout != torch.sparse_coo or support.shape != (nodes, nodes)
            for support in supports
        ):
            raise ValueError("Supports must be equally sized square sparse COO matrices")
        self.register_buffer("forward_support", supports[0].coalesce(), persistent=False)
        self.register_buffer("reverse_support", supports[1].coalesce(), persistent=False)
        self.hidden = hidden
        self.receptive_field = 1 + 2 * (2**temporal_layers - 1)
        self.checkpoint_blocks = checkpoint_blocks
        self.dynamic_projection = nn.Linear(18 + 16 * bool(basin_inputs), hidden)
        self.static_projection = nn.Linear(48, hidden, bias=False)
        self.temporal_blocks = nn.ModuleList(
            _TemporalBlock(hidden, 2**layer, dropout) for layer in range(temporal_layers)
        )
        self.graph_blocks = nn.ModuleList(
            _GraphBlock(hidden, dropout, direction, aggregation) for _ in range(graph_layers)
        )
        self.output_norm = nn.LayerNorm(hidden)

    def _block(self, block, *inputs):
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(block, *inputs, use_reentrant=False, preserve_rng_state=True)
        return block(*inputs)

    def forward(self, dynamic, static, supports=None, last_only=False):
        """``supports`` optionally replaces the stored graph, e.g. with an exact subgraph.

        Graph blocks and the output norm act on each timestep separately, so
        ``last_only=True`` returns exactly the final step of the full result
        while skipping graph propagation at earlier steps.
        """
        if dynamic.ndim != 4 or dynamic.shape[-1] != 18 or dynamic.shape[1] < 1:
            raise ValueError("Dynamic inputs must have shape [batch, time, node, 18]")
        forward_support, reverse_support = (
            (self.forward_support, self.reverse_support) if supports is None else supports[:2]
        )
        nodes = forward_support.shape[0]
        if reverse_support.shape != (nodes, nodes):
            raise ValueError("Forward and reverse supports must have equal square shapes")
        if dynamic.shape[2] != nodes or static.shape != (nodes, 48):
            raise ValueError(
                "Dynamic/static node counts must match the graph; static is [node, 48]"
            )
        if self.basin_inputs:
            # Basin means of the 8 weather values and their 8 flags, every time step.
            basin = _basin(dynamic[..., :16], self.routing(supports, nodes))
            dynamic = torch.cat([dynamic, basin], dim=-1)
        value = self.dynamic_projection(dynamic) + self.static_projection(static)[None, None]
        for block in self.temporal_blocks:
            value = self._block(block, value)
        if last_only:
            value = value[:, -1:]
        for block in self.graph_blocks:
            value = self._block(block, value, forward_support, reverse_support)
        return self.output_norm(value)

    def routing(self, supports, nodes):
        routing = (
            self.full_routing if supports is None else (supports[2] if len(supports) > 2 else None)
        )
        if routing is None or routing.n != nodes:
            raise ValueError("This model needs the river routing operator of its node set")
        return routing


class SSLReconstructor(nn.Module):
    """Predict normalized log flow at each timestep: ``[batch, time, node]``."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.readout = nn.Linear(encoder.hidden, 1)

    def forward(self, dynamic, static, supports=None):
        return self.readout(self.encoder(dynamic, static, supports)).squeeze(-1)


class ForecastModel(nn.Module):
    """Reusable forecast head; no observed-data adapter or training policy.

    Where the latest flow is available, zero-initialized corrections reproduce
    persistence exactly. Missing-history predictions have their own trainable
    output branch and share the trainable encoder. Output is ``[batch, lead, node]``.
    """

    def __init__(self, encoder, horizon=40):
        super().__init__()
        if horizon < 1:
            raise ValueError("Forecast horizon must be positive")
        self.encoder = encoder
        self.correction = nn.Linear(encoder.hidden, horizon)
        self.missing_prediction = nn.Linear(encoder.hidden, horizon)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, dynamic, static, supports=None):
        features = self.encoder(dynamic, static, supports, last_only=True)[:, -1]
        latest = dynamic[:, -1, :, 16]
        available = dynamic[:, -1, :, 17] > 0.5
        anchored = latest[..., None] + self.correction(features)
        fallback = self.missing_prediction(features)
        return torch.where(available[..., None], anchored, fallback).permute(0, 2, 1)


class WeatherForecastModel(nn.Module):
    """Forecast head that also reads weather for every lead step.

    ``future`` is ``[batch, lead, node, 2k]``: k weather values and their
    validity flags for the forecast steps (all eight variables, or precipitation
    only with ``future_channels=2``). With realized AORC this is a
    perfect-forcing hindcast (an upper bound), not an operational forecast.
    The issue-time encoder state, a learned lead embedding and the projected
    weather pass through causal temporal blocks along the lead axis, then graph
    blocks that route information between reaches at each lead. As in
    ``ForecastModel``, zero-initialized corrections start exactly at
    persistence where the latest flow is visible.
    """

    def __init__(
        self,
        encoder,
        horizon=40,
        decoder_layers=5,
        decoder_graph_layers=2,
        future_channels=16,
        routing_bins=0,
        routing_lags=12,
        basin_rain=False,
    ):
        super().__init__()
        if (routing_bins or basin_rain) and encoder.direction != "upstream":
            raise ValueError("Routing and basin rain need closed patches: direction='upstream'")
        if horizon < 1 or decoder_layers < 1 or decoder_graph_layers < 0:
            raise ValueError("Invalid forecast horizon or decoder depth")
        if future_channels < 2 or future_channels % 2:
            raise ValueError("Future inputs are values plus one validity flag each")
        hidden = encoder.hidden
        self.encoder, self.horizon, self.future_channels = encoder, horizon, future_channels
        self.basin_rain = basin_rain
        self.weather_projection = nn.Linear(future_channels * (2 if basin_rain else 1), hidden)
        self.lead_embedding = nn.Parameter(torch.zeros(horizon, hidden))
        self.temporal_blocks = nn.ModuleList(
            _TemporalBlock(hidden, 2**layer, 0.0) for layer in range(decoder_layers)
        )
        self.graph_blocks = nn.ModuleList(
            _GraphBlock(hidden, 0.0, encoder.direction, encoder.aggregation)
            for _ in range(decoder_graph_layers)
        )
        self.routing_block = (
            _RoutingBlock(hidden, routing_bins, routing_lags) if routing_bins else None
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.correction = nn.Linear(hidden, 1)
        self.missing_prediction = nn.Linear(hidden, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, dynamic, static, future, supports=None):
        if (
            future.ndim != 4
            or future.shape[1] != self.horizon
            or future.shape[-1] != self.future_channels
        ):
            raise ValueError(
                f"Future weather must have shape [batch, lead, node, {self.future_channels}]"
            )
        if future.shape[0] != dynamic.shape[0] or future.shape[2] != dynamic.shape[2]:
            raise ValueError("Future weather batch/node counts must match the history")
        forward_support, reverse_support = (
            (self.encoder.forward_support, self.encoder.reverse_support)
            if supports is None
            else supports[:2]
        )
        nodes = forward_support.shape[0]
        state = self.encoder(dynamic, static, supports, last_only=True)
        if self.basin_rain:
            # Basin-mean future rain (value and known share) beside each reach's own.
            future = torch.cat([future, _basin(future, self.encoder.routing(supports, nodes))], -1)
        value = state + self.weather_projection(future) + self.lead_embedding[None, :, None]
        for block in self.temporal_blocks:
            value = self.encoder._block(block, value)
        for block in self.graph_blocks:
            value = self.encoder._block(block, value, forward_support, reverse_support)
        if self.routing_block is not None:
            routing = self.encoder.routing(supports, nodes)
            value = self.encoder._block(self.routing_block, value, routing)
        value = self.output_norm(value)
        latest = dynamic[:, -1, :, 16]
        available = dynamic[:, -1, :, 17] > 0.5
        anchored = latest[:, None] + self.correction(value)[..., 0]
        fallback = self.missing_prediction(value)[..., 0]
        return torch.where(available[:, None], anchored, fallback)
