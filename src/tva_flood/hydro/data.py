"""Everything the model reads, on one six-hour timeline from 1979-02-01 00Z.

``Arrays`` joins the training (to 2019) and test (2020 on) monthly array caches,
which share one normalization. ``Network`` holds the reach attributes, areas, gauge
basins and routing groups. ``Samples`` cuts one forecast sample at an issue time:
weather over the whole network, future rain from a chosen source, NWM targets,
observed flow at the gauges and which gauges' history the model may see.
"""

import json

import numpy as np
import pandas as pd

from tva_flood.data.dataset import FullNetworkWindows
from tva_flood.data.gauges import Catalog
from tva_flood.data.rain_forecasts import RainForecasts

from . import paths as P

PRECIP, TEMPERATURE = 0, 1


def timeline_index(time):
    """Global step of a timestamp (six-hour steps from ``paths.START``)."""
    return int((pd.Timestamp(time) - P.START) / pd.Timedelta(hours=6))


class Arrays:
    """Memory-mapped monthly arrays (weather, weather_valid, flow, flow_valid) on one timeline."""

    def __init__(self):
        self.parts = []  # (first step, steps, folder)
        for prep, arrays, keep in (
            (P.TRAIN_PREP, P.TRAIN_ARRAYS, lambda s: s < P.TEST_FROM),
            (P.TEST_PREP, P.TEST_ARRAYS, lambda s: s >= P.TEST_FROM),
        ):
            manifest = json.loads((prep / "dataset_manifest.json").read_text())
            for k, record in enumerate(manifest["months"]):
                start = pd.Timestamp(record["start"])
                if keep(start):
                    self.parts.append((timeline_index(start), record["steps"], arrays / f"{k:04d}"))
        firsts = [p[0] for p in self.parts]
        ends = [p[0] + p[1] for p in self.parts]
        if firsts[0] != 0 or firsts[1:] != ends[:-1]:
            raise ValueError("Array months do not form one contiguous timeline")
        self.first = np.array(firsts)
        self.steps = int(ends[-1])
        self.times = pd.date_range(P.START, periods=self.steps, freq="6h")
        self.open = {}
        norm = json.loads((P.TRAIN_PREP / "dynamic_normalization.json").read_text())
        test_norm = json.loads((P.TEST_PREP / "dynamic_normalization.json").read_text())
        if norm["mean"] != test_norm["mean"] or norm["std"] != test_norm["std"]:
            raise ValueError("Training and test arrays use different normalizations")
        self.mean = np.asarray(norm["mean"], dtype=np.float64)
        self.std = np.asarray(norm["std"], dtype=np.float64)

    def month(self, k):
        if k not in self.open:
            folder = self.parts[k][2]
            self.open[k] = {
                name: np.load(folder / f"{name}.npy", mmap_mode="r")
                for name in ("weather", "weather_valid", "flow", "flow_valid")
            }
        return self.open[k]

    def read(self, name, start, stop):
        """Array ``name`` over global steps [start, stop), every reach."""
        if start < 0 or stop > self.steps:
            raise ValueError(f"Steps {start}..{stop} outside the timeline")
        pieces, cursor = [], start
        while cursor < stop:
            k = int(np.searchsorted(self.first, cursor, side="right") - 1)
            end = min(stop, self.parts[k][0] + self.parts[k][1])
            local = cursor - self.parts[k][0]
            pieces.append(self.month(k)[name][local : local + end - cursor])
            cursor = end
        return np.concatenate(pieces, axis=0) if len(pieces) > 1 else np.array(pieces[0])

    def flow_raw(self, z):
        """NWM flow (m3/s) from normalized log flow."""
        return np.expm1(z.astype(np.float64) * self.std[8] + self.mean[8]).clip(min=0)

    def rain_z(self, mm):
        """Normalized precipitation from six-hour totals (mm); NaN stays NaN."""
        return (np.log1p(np.maximum(mm, 0)) - self.mean[PRECIP]) / self.std[PRECIP]


class Network:
    """Reach attributes, river edges, areas, basins and the routing groups of the full graph."""

    def __init__(self):
        windows = FullNetworkWindows(P.ROOT, P.TRAIN_PREP, split="train")
        self.ids = windows.ids
        self.n = len(self.ids)
        self.static = windows.static.astype(np.float32)  # [reach, 48]
        with np.load(P.TRAIN_PREP / "graph.npz") as g:  # downstream edges (source -> target)
            self.edges = np.column_stack([g["source"], g["target"]]).astype(np.int64)
        with np.load(P.HYDRO / "routing.npz") as r:
            if not np.array_equal(r["ids"], self.ids):
                raise ValueError("Routing groups were built for another reach order")
            self.share, self.group = r["share"], r["group"]
            self.groups = int(r["groups"])
            self.delay = r["delay"]  # initial travel time per group (steps)
            self.area = r["area"]  # local catchment area (km2, >= 0.01)
            self.basin_area = r["basin_area"]
            self.indptr = r["indptr"]  # upstream members of each reach (CSR rows)
            self.members = r["members"]
        stats = np.load(P.HYDRO / "reach_stats.npz")  # NaN at reaches without NWM flow
        self.std_raw = np.nan_to_num(stats["std_raw"], nan=1.0).astype(np.float32)
        self.mean_raw = np.nan_to_num(stats["mean_raw"], nan=1.0).astype(np.float32)

    def basin(self, reach):
        return self.members[self.indptr[reach] : self.indptr[reach + 1]]


class Gauges:
    """Observed flow at the public catalog gauges, on the timeline, and their basins."""

    def __init__(self, network, times):
        catalog = Catalog(P.GAUGES)
        self.names = catalog.names
        self.camelsh = catalog.camelsh
        self.reach = pd.Index(network.ids).get_indexer(catalog.comid)
        if (self.reach < 0).any():
            raise ValueError("A gauge is not a reach of the graph")
        self.observed = catalog.on(times)  # [time, gauge] m3/s, NaN missing
        self.catalog = catalog
        self.count = len(self.names)
        self.basins = [network.basin(r) for r in self.reach]
        self.size = np.array([len(b) for b in self.basins])
        self.area = network.basin_area[self.reach]
        train = times < P.TRAIN_UNTIL
        flow = self.observed[train]
        with np.errstate(invalid="ignore"):
            self.std_raw = np.nan_to_num(np.nanstd(flow, axis=0), nan=1.0).astype(np.float32)
            self.mean_raw = np.nan_to_num(np.nanmean(flow, axis=0), nan=1.0).astype(np.float32)
            self.p90 = np.nan_to_num(np.nanquantile(flow, 0.9, axis=0), nan=np.inf).astype(np.float32)
        self.trained = np.isfinite(flow).sum(axis=0) > 365 * 4

    def owners(self, visible, n):
        """Visible gauge owning each reach: the smallest visible basin containing it, else -1."""
        owner = np.full(n, -1, dtype=np.int64)
        shown = np.flatnonzero(visible)
        for g in shown[np.argsort(-self.size[shown], kind="stable")]:
            owner[self.basins[g]] = g
        return owner


class Dams:
    """TVA dams on the graph (``hydro.dams``): their reaches, observed releases and inflows on
    the timeline, TVA's forecast releases, and how every reach's water passes them."""

    def __init__(self, network, times):
        with np.load(P.HYDRO / "dams.npz") as d:
            if pd.Timestamp(str(d["start"])) != times[0] or d["release"].shape[0] != len(times):
                raise ValueError("dams.npz was built for another timeline (rerun hydro-dams)")
            if not np.array_equal(d["comid"], network.ids[d["reach"]]):
                raise ValueError("dams.npz was built for another reach order")
            self.names = d["dam"].astype(str)
            self.reach = d["reach"].astype(np.int64)
            self.downstream = d["downstream"].astype(np.int64)  # next dam below (-1: none)
            self.release = d["release"]  # [time, dam] m3/s, NaN missing
            self.inflow = d["inflow"]  # computed by TVA from release and storage change
            self.plan = d["forecast_release"]  # TVA forecast release [issue, lead, dam]
            self.plan_inflow = d["forecast_inflow"]  # TVA forecast inflow [issue, lead, dam]
            self.plan_index = {int(s): k for k, s in enumerate(d["forecast_issue"])}
            self.regulated = d["regulated"]  # share of each reach's basin behind a dam above it
            self.dam_below = d["dam_below"]  # first dam on each reach's way down (-1: none)
        self.count = len(self.names)
        train = times < P.TRAIN_UNTIL
        with np.errstate(invalid="ignore"):
            # At least 1 m3/s: Tellico releases almost nothing (its water takes a canal).
            self.mean_raw = np.maximum(np.nan_to_num(np.nanmean(self.release[train], axis=0), nan=1.0), 1.0).astype(np.float32)
            self.std_raw = np.nan_to_num(np.nanstd(self.release[train], axis=0), nan=1.0).astype(np.float32)
        # Upstream dams first: a dam's inflow forecast needs the releases of the dams above it.
        self.level = np.zeros(self.count, dtype=np.int64)
        for _ in range(self.count):
            for k, below in enumerate(self.downstream):
                if below >= 0:
                    self.level[below] = max(self.level[below], self.level[k] + 1)
        # Running sums of inflow minus release: storage change over any window in O(1).
        imbalance = np.nan_to_num(self.inflow.astype(np.float64) - self.release.astype(np.float64))
        self.stored = np.vstack([np.zeros((1, self.count)), np.cumsum(imbalance, axis=0)])

    def storage(self, step, days=(30, 90)):
        """Mean inflow minus release over the last ``days`` before ``step``, in mean releases [len(days), dam]."""
        out = []
        for d in days:
            span = min(step, d * P.STEPS_PER_DAY)
            out.append((self.stored[step] - self.stored[step - span]) / (max(span, 1) * self.mean_raw))
        return np.stack(out).astype(np.float32)

    def planned(self, step, max_age=P.STEPS_PER_DAY, inflow=False):
        """TVA's forecast release (or ``inflow``) [lead, dam] in force at ``step``: the latest
        issued at or before it, at most ``max_age`` steps earlier (TVA issues four times a day,
        a 00Z forecast exists on about two days in three), its leads shifted to start at
        ``step``. NaN where none (or past the 240 h kept)."""
        source = self.plan_inflow if inflow else self.plan
        out = np.full((P.HORIZON, self.count), np.nan, np.float32)
        for age in range(max_age + 1):
            k = self.plan_index.get(int(step) - age)
            if k is not None:
                out[: P.HORIZON - age] = source[k, age:]
                break
        return out


class Rain:
    """Future rain per reach (mm, NaN unknown) by source and issue day."""

    def __init__(self):
        self.sources = {}

    def get(self, source, day):
        if source not in self.sources:
            folder = dict(gefs_reforecast=P.GEFS_REFORECAST, gefs=P.GEFS_OPERATIONAL)[source]
            self.sources[source] = RainForecasts(folder)
        rain = self.sources[source]
        return rain.get(day) if day in rain else None

    def days(self, source):
        self.get(source, P.START)
        return set(self.sources[source].days)


class Antecedent:
    """Daily antecedent-storage features [day, reach, feature] (normalized, float16)."""

    def __init__(self):
        self.values = np.load(P.HYDRO / "antecedent.npy", mmap_mode="r")
        self.info = json.loads((P.HYDRO / "antecedent.json").read_text())
        if pd.Timestamp(self.info["start"]) != P.START:
            raise ValueError("Antecedent features start on another day")

    def at(self, step):
        """Features at the start of the issue's day (state after the previous day)."""
        return np.asarray(self.values[step // P.STEPS_PER_DAY], dtype=np.float32)
