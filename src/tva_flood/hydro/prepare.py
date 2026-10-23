"""Products the distributed model needs, written once to ``processed/hydro``.

routing.npz
    For every reach i and every reach j upstream of it (i included): the share f_ij of
    j's water that reaches i (1 except below divergences), and a routing group from the
    river distance j -> i and the number of lake or reservoir reaches on that path. The
    model routes lateral inflow q_j to Q_i = sum_j f_ij (h_g(ij) * q_j)(t), with one
    learned travel-time kernel h per group that sums to one: water is neither made nor
    lost on the way.
reach_stats.npz
    Per-reach spread of NWM retrospective flow (raw and log1p) over the training years,
    to normalize the NWM loss reach by reach (an NSE-like loss).
antecedent.npy / antecedent.json
    Daily storage proxies per reach: exponentially weighted rain over 7 to 365 days and
    temperature over 30 and 90 days, at the start of each day. They carry memory far
    beyond the model's weather window (slow groundwater, karst baseflow).
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from tva_flood.common import atomic_json
from tva_flood.data.river import River, topological_order

from . import paths as P
from .data import PRECIP, TEMPERATURE, Arrays, timeline_index

DISTANCE_KM = (0.0, 5.0, 15.0, 35.0, 75.0, 150.0, 300.0, 500.0, 800.0, np.inf)
LAKE_LEVELS = (0, 1, 11)  # lake reaches on the path: none, 1-10, more than 10
KM_PER_STEP = 21.6  # 1 m/s over six hours
RAIN_DAYS = (7, 30, 90, 180, 365)
TEMPERATURE_DAYS = (30, 90)
MIN_AREA = 0.01


def graph():
    with np.load(P.TRAIN_PREP / "graph.npz") as g:
        return g["feature_id"].copy(), np.column_stack([g["source"], g["target"]])


def routing():
    ids, edges = graph()
    river = River.load(P.ROOT, ids, edges)
    nodes = pd.read_parquet(
        P.PROCESSED / "graph_candidate/nodes.parquet", columns=["comid", "areasqkm", "wbareatype"]
    ).set_index("comid").loc[ids]
    area = np.maximum(nodes.areasqkm.to_numpy(dtype=np.float64), MIN_AREA)
    lake = nodes.wbareatype.isin(["LakePond", "Reservoir"]).to_numpy().astype(np.int64)
    n = len(ids)
    counts = np.diff(river.indptr)
    rows = np.repeat(np.arange(n), counts)
    cols = river.indices.astype(np.int64)
    share = river.weight.astype(np.float64) * river.basin_area[rows] / area[cols]
    # Lake reaches from each reach to the outlet along the main downstream path.
    down = np.full(n, -1)
    down[edges[::-1, 0]] = edges[::-1, 1]
    lakes_below = lake.copy()
    for i in topological_order(n, edges)[::-1]:
        if down[i] >= 0:
            lakes_below[i] = lake[i] + lakes_below[down[i]]
    on_path = np.maximum(lakes_below[cols] - lakes_below[rows] + lake[rows], 0)
    distance_bin = np.searchsorted(DISTANCE_KM, river.distance, side="right") - 1
    lake_level = np.searchsorted(LAKE_LEVELS, on_path, side="right") - 1
    levels = len(LAKE_LEVELS)
    group = (distance_bin * levels + lake_level).astype(np.int16)
    groups = (len(DISTANCE_KM) - 1) * levels
    middle = np.array(
        [(a + (b if np.isfinite(b) else 1000.0)) / 2 for a, b in zip(DISTANCE_KM, DISTANCE_KM[1:])]
    )
    delay = np.repeat(middle / KM_PER_STEP, levels) + np.tile([0.0, 2.0, 8.0], len(middle))
    np.savez(
        P.HYDRO / "routing.npz",
        ids=ids,
        indptr=river.indptr.astype(np.int64),
        members=cols.astype(np.int32),
        share=share.astype(np.float32),
        group=group,
        groups=np.array(groups),
        delay=delay.astype(np.float32),
        area=area.astype(np.float32),
        basin_area=river.basin_area.astype(np.float32),
    )
    used = np.bincount(group, minlength=groups)
    print(f"routing: {len(cols):,} pairs, {groups} groups, pairs per group {used.tolist()}")


def _month(k):
    """Daily rain (mm) and temperature (normalized) of month k, and NWM flow sums."""
    arrays = Arrays()
    first, steps, _ = arrays.parts[k]
    months = arrays.month(k)
    z = np.asarray(months["weather"][:, :, PRECIP], dtype=np.float64)
    valid = np.asarray(months["weather_valid"][:, :, PRECIP])
    mm = np.where(valid, np.expm1(z * arrays.std[PRECIP] + arrays.mean[PRECIP]).clip(0), 0.0)
    temperature = np.asarray(months["weather"][:, :, TEMPERATURE], dtype=np.float64)
    temperature_valid = np.asarray(months["weather_valid"][:, :, TEMPERATURE])
    days = steps // P.STEPS_PER_DAY
    shape = (days, P.STEPS_PER_DAY, -1)
    rain = mm.reshape(shape).sum(axis=1)
    rain_valid = valid.reshape(shape).any(axis=1)
    t_sum = np.where(temperature_valid, temperature, 0).reshape(shape).sum(axis=1)
    t_count = temperature_valid.reshape(shape).sum(axis=1)
    sums = None
    start = arrays.times[first]
    if P.TRAIN_FROM <= start < P.TRAIN_UNTIL:
        flow_valid = np.asarray(months["flow_valid"])
        q = np.where(flow_valid, arrays.flow_raw(np.asarray(months["flow"])), 0.0)
        logq = np.log1p(q)
        sums = np.stack(
            [flow_valid.sum(0), q.sum(0), (q * q).sum(0), logq.sum(0), (logq * logq).sum(0)]
        )
    temperature_day = np.where(t_count > 0, t_sum / np.maximum(t_count, 1), np.nan)
    return k, rain.astype(np.float32), rain_valid, temperature_day.astype(np.float32), sums


def features(workers):
    arrays = Arrays()
    n = arrays.read("flow", 0, 1).shape[1]
    days = arrays.steps // P.STEPS_PER_DAY
    rain = np.zeros((days, n), dtype=np.float32)
    rain_valid = np.zeros((days, n), dtype=bool)
    temperature = np.zeros((days, n), dtype=np.float32)
    sums = np.zeros((5, n))
    with ProcessPoolExecutor(workers) as pool:
        for k, r, rv, t, s in pool.map(_month, range(len(arrays.parts))):
            day = arrays.parts[k][0] // P.STEPS_PER_DAY
            rain[day : day + len(r)], rain_valid[day : day + len(r)] = r, rv
            temperature[day : day + len(t)] = t
            if s is not None:
                sums += s
            if k % 50 == 0:
                print(f"month {k}/{len(arrays.parts)} read", flush=True)
    count, s1, s2, l1, l2 = sums
    std_raw = np.sqrt(np.maximum(s2 / count - (s1 / count) ** 2, 0))
    std_log = np.sqrt(np.maximum(l2 / count - (l1 / count) ** 2, 0))
    np.savez(P.HYDRO / "reach_stats.npz", std_raw=std_raw, std_log=std_log, mean_raw=s1 / count)
    # Exponentially weighted memories, state at the start of each day.
    rates = np.array([1 / d for d in RAIN_DAYS + TEMPERATURE_DAYS])
    state = np.zeros((len(rates), n))
    state[: len(RAIN_DAYS)] = np.nanmean(rain[: 365 * 2][rain_valid[: 365 * 2]])
    width = len(rates)
    out = np.lib.format.open_memmap(
        P.HYDRO / "antecedent.part.npy", mode="w+", dtype=np.float32, shape=(days, n, width)
    )
    for d in range(days):
        out[d] = state.T
        observed = np.vstack(
            [np.repeat(rain[d][None], len(RAIN_DAYS), 0), np.repeat(temperature[d][None], 2, 0)]
        )
        known = np.vstack(
            [
                np.repeat(rain_valid[d][None], len(RAIN_DAYS), 0),
                np.repeat(np.isfinite(temperature[d])[None], 2, 0),
            ]
        )
        state = np.where(known, state + rates[:, None] * (observed - state), state)
    # Rain memories in log space; normalize every feature over training days.
    rain_part = slice(0, len(RAIN_DAYS))
    train = slice(
        timeline_index(P.TRAIN_FROM) // P.STEPS_PER_DAY,
        timeline_index(P.TRAIN_UNTIL) // P.STEPS_PER_DAY,
    )
    mean, std = np.zeros(width), np.ones(width)
    sample_days = np.arange(train.start, train.stop, 7)
    for d in range(0, days, 512):
        block = out[d : d + 512]
        block[..., rain_part] = np.log1p(np.maximum(block[..., rain_part], 0))
        out[d : d + 512] = block
    picked = np.asarray(out[sample_days], dtype=np.float64)
    mean = picked.reshape(-1, width).mean(axis=0)
    std = picked.reshape(-1, width).std(axis=0) + 1e-6
    final = np.lib.format.open_memmap(
        P.HYDRO / "antecedent.npy", mode="w+", dtype=np.float16, shape=(days, n, width)
    )
    for d in range(0, days, 512):
        final[d : d + 512] = ((out[d : d + 512] - mean) / std).astype(np.float16)
    final.flush()
    del out, final
    (P.HYDRO / "antecedent.part.npy").unlink()
    atomic_json(
        P.HYDRO / "antecedent.json",
        dict(
            start=str(P.START),
            days=int(days),
            features=[f"log1p rain EWMA {d} d" for d in RAIN_DAYS]
            + [f"temperature EWMA {d} d" for d in TEMPERATURE_DAYS],
            mean=mean.tolist(),
            std=std.tolist(),
            row="state at the start of the day (days before it only)",
        ),
    )
    print(f"antecedent: {days} days x {n} reaches x {width} features", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--only", choices=["routing", "features"])
    a = p.parse_args()
    P.HYDRO.mkdir(parents=True, exist_ok=True)
    if a.only != "features":
        routing()
    if a.only != "routing":
        features(a.workers)
    print("HYDRO PREPARE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
