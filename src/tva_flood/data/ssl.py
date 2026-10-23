"""Historical flow reconstruction windows and deterministic structured masking."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import CachedWindows
from .schema import HISTORY, HORIZON
from .subgraph import expand

# Dynamic encoder channels: 8 weather values, 8 weather validity flags, then
# normalized log flow history and its availability.
WEATHER_CHANNELS = slice(0, 16)
FLOW, FLOW_VALID = 16, 17
MODES = ("block", "tail", "reach", "patch", "all")


def history_issues(times, split_start, split_stop, history=HISTORY):
    """Return end-exclusive indices whose complete history belongs to the split.

    The timeline consists of six-hour observations. An issue may equal
    ``len(times)``: that window ends just after the final available observation.
    Unlike forecast indexing, no future target horizon is reserved or read.
    """
    if history < 1:
        raise ValueError("History length must be positive")
    times = pd.DatetimeIndex(times)
    if not len(times):
        return np.empty(0, dtype=np.int64)
    if len(times) > 1 and not np.all(np.diff(times.asi8) == pd.Timedelta(hours=6).value):
        raise ValueError("Expected an ordered, contiguous six-hour timeline")
    first, stop = pd.Timestamp(split_start), pd.Timestamp(split_stop)
    if first >= stop:
        raise ValueError("Split start must precede split stop")
    endpoints = times.append(pd.DatetimeIndex([times[-1] + pd.Timedelta(hours=6)]))
    left = int(times.searchsorted(first, side="left")) + history
    right = min(len(times), int(endpoints.searchsorted(stop, side="right")) - 1)
    return np.arange(left, max(left, right + 1), dtype=np.int64)


class EncoderWindows(CachedWindows):
    """Eighteen-channel histories with statics kept separate; optional forecast targets.

    ``horizon=0`` indexes reconstruction windows that read history only.
    ``horizon=HORIZON`` indexes forecast issue times and adds
    ``target``/``target_valid`` for the following forty steps, and with
    ``future_rain`` the realized AORC precipitation for those steps ``future``
    ``[lead, node, 2]`` (normalized value, validity flag): a perfect forecast, to be
    degraded by the caller. ``nodes`` selects a subset of graph positions without
    decoding anything twice. An optional array cache replaces NetCDF decoding with
    identical memory maps.
    """

    def __init__(
        self,
        root,
        preparation,
        split,
        horizon=0,
        cache_months=4,
        array_cache=None,
        future_rain=False,
    ):
        if horizon not in (0, HORIZON):
            raise ValueError(f"Horizon must be 0 (reconstruction) or {HORIZON} (forecast)")
        if cache_months < 1:
            raise ValueError("Positive cache capacity required")
        super().__init__(root, preparation, split, cache_months=cache_months)
        if future_rain and not horizon:
            raise ValueError("Future precipitation needs forecast windows (horizon > 0)")
        self.horizon, self.preparation = horizon, Path(preparation)
        self.future_rain = bool(future_rain)
        self.arrays = None
        if array_cache is not None:
            from .arrays import ArrayMonths

            self.arrays = ArrayMonths(array_cache, self)
        if horizon == 0:
            first, stop = self.manifest["splits"][split]
            self.issues = history_issues(self.times, first, stop)

    def month(self, number):
        if getattr(self, "arrays", None) is None:
            return super().month(number)
        return self.arrays.month(number)

    def node_window(self, start, stop, nodes=None):
        if nodes is None:
            return self.window(start, stop)
        nodes = np.asarray(nodes, dtype=np.int64)
        parts = [[], [], [], []]
        cursor = start
        while cursor < stop:
            number = int(np.searchsorted(self.offsets, cursor, side="right") - 1)
            end = min(stop, int(self.offsets[number + 1]))
            left, right = cursor - self.offsets[number], end - self.offsets[number]
            for part, array in zip(parts, self.month(number)):
                part.append(array[left:right][:, nodes])
            cursor = end
        return tuple(np.concatenate(part, axis=0) for part in parts)

    def load(self, item, nodes=None):
        issue = int(self.issues[item])
        weather, weather_valid, flow, valid = self.node_window(
            issue - HISTORY, issue + self.horizon, nodes
        )
        history_flow, history_valid = flow[:HISTORY], valid[:HISTORY]
        dynamic = np.concatenate(
            [
                weather[:HISTORY],
                weather_valid[:HISTORY].astype(np.float32),
                np.where(history_valid, history_flow, 0)[..., None],
                history_valid.astype(np.float32)[..., None],
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        result = dict(
            dynamic=torch.from_numpy(dynamic),
            flow=torch.from_numpy(np.ascontiguousarray(history_flow, dtype=np.float32)),
            valid=torch.from_numpy(np.ascontiguousarray(history_valid)),
            issue=issue,
        )
        if self.future_rain:
            future = np.concatenate(
                [weather[HISTORY:, :, :1], weather_valid[HISTORY:, :, :1].astype(np.float32)],
                axis=-1,
            )
            result["future"] = torch.from_numpy(np.ascontiguousarray(future, dtype=np.float32))
        if self.horizon:
            result.update(
                target=torch.from_numpy(np.ascontiguousarray(flow[HISTORY:], dtype=np.float32)),
                target_valid=torch.from_numpy(np.ascontiguousarray(valid[HISTORY:])),
            )
        return result

    def __getitem__(self, item):
        return self.load(item)


def mask_history(
    dynamic,
    valid,
    *,
    seed,
    mode,
    block_min=28,
    block_max=60,
    reach_fraction=0.5,
    tail_min=4,
    tail_max=40,
    hide_weather=False,
    adjacency=None,
    patch_hops=(3, 8),
):
    """Mask batched flow history without changing the caller's inputs or RNG.

    ``dynamic`` has shape ``[B,T,N,18]`` and ``valid`` is boolean ``[B,T,N]``.
    The returned loss mask includes valid hidden targets only. Modes:

    - ``block`` hides one contiguous interval per reach;
    - ``tail`` hides the final ``tail_min..tail_max`` steps at every reach, the
      shape of a forecast; ``hide_weather`` also removes weather in that tail;
    - ``reach`` hides entire, independently selected reach histories;
    - ``patch`` hides entire histories of connected groups of reaches, each
      within ``patch_hops`` undirected hops of a random seed reach, until
      ``reach_fraction`` of reaches is hidden (needs ``adjacency``). Interior
      reaches cannot copy a visible neighbour within the patch radius;
    - ``all`` hides all flow histories.

    Other input channels are retained unless ``hide_weather`` applies. A sample
    containing any valid flow always has a loss target; repairs preserve the
    mode's geometry. Naturally missing flow is always zero and unavailable.

    Masks depend on a local CPU NumPy generator, shape, validity and graph, never
    on target flow values. Supplying different epoch seeds changes training
    masks; reusing a validation seed produces exactly the same mask everywhere.
    """
    if dynamic.ndim != 4 or dynamic.shape[-1] != 18 or min(dynamic.shape[:3]) < 1:
        raise ValueError("Expected nonempty dynamic inputs shaped [B,T,N,18]")
    if tuple(valid.shape) != tuple(dynamic.shape[:3]) or valid.dtype != torch.bool:
        raise ValueError("Expected boolean validity shaped [B,T,N]")
    if not dynamic.is_floating_point() or dynamic.device != valid.device:
        raise ValueError("Dynamic inputs must be floating point on the validity device")
    if seed < 0 or int(seed) != seed:
        raise ValueError("Seed must be a nonnegative integer")
    if mode not in MODES:
        raise ValueError("Mask mode must be one of " + ", ".join(MODES))
    if not 0 < reach_fraction <= 1:
        raise ValueError("Reach fraction must be in (0,1]")
    batch, steps, nodes = valid.shape
    if mode == "block" and not 1 <= block_min <= block_max <= steps:
        raise ValueError("Block bounds must satisfy 1 <= min <= max <= history length")
    if mode == "tail" and not 1 <= tail_min <= tail_max <= steps:
        raise ValueError("Tail bounds must satisfy 1 <= min <= max <= history length")
    if mode == "patch":
        if adjacency is None or adjacency.shape != (nodes, nodes):
            raise ValueError("Patch masking needs an adjacency matrix matching the node count")
        if not 0 <= patch_hops[0] <= patch_hops[1]:
            raise ValueError("Patch hops must satisfy 0 <= min <= max")
    rng = np.random.default_rng(int(seed))
    known = valid.detach().cpu().numpy()
    clock = np.arange(steps)[None, :, None]
    if mode == "block":
        lengths = rng.integers(block_min, block_max + 1, size=(batch, nodes))
        starts = (rng.random((batch, nodes)) * (steps - lengths + 1)).astype(np.int64)
        hidden = (clock >= starts[:, None, :]) & (clock < (starts + lengths)[:, None, :])
    elif mode == "tail":
        lengths = rng.integers(tail_min, tail_max + 1, size=batch)
        hidden = np.broadcast_to(
            clock >= (steps - lengths)[:, None, None], (batch, steps, nodes)
        ).copy()
    elif mode == "reach":
        selected = rng.random((batch, nodes)) < reach_fraction
        hidden = np.broadcast_to(selected[:, None, :], (batch, steps, nodes)).copy()
    elif mode == "patch":
        selected = np.stack(
            [_patches(adjacency, rng, reach_fraction, patch_hops) for _ in range(batch)]
        )
        hidden = np.broadcast_to(selected[:, None, :], (batch, steps, nodes)).copy()
    else:
        hidden = np.ones((batch, steps, nodes), dtype=bool)
    for sample in range(batch):
        if not known[sample].any() or (hidden[sample] & known[sample]).any():
            continue
        candidates = np.flatnonzero(known[sample])
        time, node = np.unravel_index(rng.choice(candidates), (steps, nodes))
        if mode == "block":
            length = int(lengths[sample, node])
            low, high = max(0, time - length + 1), min(time, steps - length)
            start = int(rng.integers(low, high + 1))
            hidden[sample, :, node] = False
            hidden[sample, start : start + length, node] = True
        elif mode == "tail":
            # Extend the tail back to the latest valid value.
            latest = int(np.flatnonzero(known[sample].any(axis=1))[-1])
            hidden[sample, latest:] = True
        else:
            hidden[sample, :, node] = True
    hidden = torch.as_tensor(hidden, device=dynamic.device)
    loss_mask = hidden & valid
    masked = dynamic.clone()
    masked[..., FLOW : FLOW_VALID + 1] = torch.where(
        (hidden | ~valid)[..., None], 0, masked[..., FLOW : FLOW_VALID + 1]
    )
    if mode == "tail" and hide_weather:
        # Whole tail timesteps: weather values and validity flags both become zero.
        tail = hidden.all(dim=2)[..., None, None]
        masked[..., WEATHER_CHANNELS] = torch.where(tail, 0, masked[..., WEATHER_CHANNELS])
    return masked, loss_mask


def _patches(adjacency, rng, fraction, hops):
    """Union of connected patches covering at least ``fraction`` of the nodes."""
    nodes = adjacency.shape[0]
    wanted = max(1, int(np.ceil(fraction * nodes)))
    hidden = np.zeros(nodes, dtype=bool)
    # Several seeds per round keep the number of sparse expansions small on large graphs.
    per_round = max(1, nodes // 500)
    while hidden.sum() < wanted:
        seeds = np.zeros(nodes, dtype=bool)
        seeds[rng.choice(np.flatnonzero(~hidden), min(per_round, int((~hidden).sum())))] = True
        hidden |= expand(adjacency, seeds, int(rng.integers(hops[0], hops[1] + 1)))
    return hidden


def causal_persistence(flow, visible):
    """Last visible earlier value of each reach; zero (the training mean) if none.

    ``flow`` and ``visible`` have shape ``[B,T,N]``. At a hidden position this is
    the value the model could copy from its own past; whole hidden reaches get
    the normalized training mean.
    """
    steps = flow.shape[1]
    clock = torch.arange(steps, device=flow.device).view(1, steps, 1).expand_as(visible)
    latest = torch.where(visible, clock, -1).cummax(dim=1).values
    values = torch.gather(torch.where(visible, flow, 0), 1, latest.clamp_min(0))
    return torch.where(latest >= 0, values, 0)


def mask_future(
    future,
    *,
    seed,
    adjacency=None,
    none_prob=0.2,
    region_prob=0.4,
    region_fraction=(0.05, 0.5),
    region_hops=(5, 30),
    cutoff_prob=0.5,
    cutoff_steps=(4, 40),
):
    """Hide future weather the way a real forecast would be incomplete.

    ``future`` is ``[lead, node, 2k]``: k values, then their k validity flags.
    One draw per call decides the spatial coverage: nothing known (probability
    ``none_prob``), only connected regions covering a random fraction of the
    nodes (``region_prob``; needs ``adjacency``), or every node. Independently,
    with ``cutoff_prob`` only the first ``cutoff_steps`` leads stay known.
    Hidden values and their flags become zero. Deterministic for a seed;
    never depends on the weather values themselves.
    """
    if future.ndim != 3 or future.shape[-1] % 2:
        raise ValueError("Expected future inputs shaped [lead, node, 2k]")
    if none_prob < 0 or region_prob < 0 or none_prob + region_prob > 1:
        raise ValueError("Coverage probabilities must be nonnegative and sum to <= 1")
    leads, nodes, channels = future.shape
    rng = np.random.default_rng(int(seed))
    draw = rng.random()
    known = np.ones(nodes, dtype=bool)
    if draw < none_prob:
        known[:] = False
    elif draw < none_prob + region_prob:
        if adjacency is None or adjacency.shape != (nodes, nodes):
            raise ValueError("Regional coverage needs an adjacency matching the node count")
        fraction = rng.uniform(*region_fraction)
        known = _patches(adjacency, rng, fraction, region_hops)
    horizon = leads
    if rng.random() < cutoff_prob:
        horizon = int(rng.integers(cutoff_steps[0], min(cutoff_steps[1], leads) + 1))
    keep = torch.zeros((leads, nodes, 1), dtype=torch.bool)
    keep[:horizon, torch.as_tensor(known)] = True
    keep = keep.to(future.device)
    return torch.where(keep, future, torch.zeros((), dtype=future.dtype, device=future.device))
