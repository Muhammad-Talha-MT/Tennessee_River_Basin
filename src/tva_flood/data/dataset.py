"""Prepared-data validation, split indexing and COMID-aligned monthly loading."""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tva_flood.common import hash_array

from . import statics
from .schema import HISTORY, HORIZON, WEATHER
from .transforms import transform_precipitation


def lineage(manifest):
    """Identity of the training data and normalization behind a manifest.

    A test preparation has its own identity (its months differ) but shares the
    parent's graph, statics and normalization, which is what checkpoints record.
    """
    return manifest.get("parent_identity", manifest["identity"])


class FullNetworkWindows:
    """Validated prepared-data metadata and temporal split indexing."""

    def __init__(self, root, out, split="train", node_indices=None):
        self.root, out = Path(root), Path(out)
        self.manifest = json.loads((out / "dataset_manifest.json").read_text())
        if not self.manifest["training_ready"]:
            raise ValueError("Run full preparation without --headers-only first")
        self.normalization = json.loads((out / "dynamic_normalization.json").read_text())
        if self.normalization["identity"] != lineage(self.manifest):
            raise ValueError("Dynamic normalization belongs to a different dataset")
        # Missing fields in the original schema mean raw precipitation, never log1p.
        self.precip_transform = self.manifest.get("precip_transform", "none")
        if self.normalization.get("precip_transform", "none") != self.precip_transform:
            raise ValueError("Precipitation transform differs between manifest and normalization")
        with np.load(out / "graph.npz") as graph:
            all_ids = graph["feature_id"].copy()
            graph_edges = np.column_stack([graph["source"], graph["target"]])
        if (
            hash_array(all_ids.astype("<i8")) != self.manifest["node_hash"]
            or hash_array(graph_edges.astype("<i8")) != self.manifest["edge_hash"]
        ):
            raise ValueError("Prepared graph was modified")
        if (
            statics.sha(self.root / "processed/static/tennessee_static_core.nc")
            != self.manifest["static_sha256"]
        ):
            raise ValueError("Static source was modified; rebuild preparation")
        self.nodes = (
            np.arange(len(all_ids))
            if node_indices is None
            else np.asarray(node_indices, dtype=np.int64)
        )
        if (
            self.nodes.ndim != 1
            or not len(self.nodes)
            or len(np.unique(self.nodes)) != len(self.nodes)
        ):
            raise ValueError("Empty or duplicate node selection")
        if (self.nodes < 0).any() or (self.nodes >= len(all_ids)).any():
            raise ValueError("Node selection outside graph")
        self.ids = all_ids[self.nodes]
        self.static, _ = statics.load_static_inputs(
            self.root / "processed/static/tennessee_static_core.nc",
            self.ids,
            normalization=json.loads((out / "static_normalization.json").read_text()),
        )
        self.records = self.manifest["months"]
        self.offsets = np.r_[0, np.cumsum([r["steps"] for r in self.records])]
        self.times = pd.date_range(
            self.records[0]["start"], periods=int(self.offsets[-1]), freq="6h"
        )
        if split not in self.manifest["splits"]:
            raise ValueError("Split must be train or validation; later test years are reserved")
        first, stop = map(pd.Timestamp, self.manifest["splits"][split])
        left = max(HISTORY, int(self.times.searchsorted(first)))
        right = int(self.times.searchsorted(stop)) - HORIZON + 1
        self.issues = np.arange(left, right, dtype=np.int64)

    def __len__(self):
        return len(self.issues)


class CachedWindows(FullNetworkWindows):
    """Normalized weather, flow and validity arrays, cached by month."""

    def __init__(self, root, out, split, cache_months=6):
        super().__init__(root, out, split=split)
        self.cache, self.cache_months = OrderedDict(), cache_months
        self.mean = np.asarray(self.normalization["mean"])
        self.std = np.asarray(self.normalization["std"])

    def month(self, number):
        if number in self.cache:
            self.cache.move_to_end(number)
            return self.cache[number]
        r = self.records[number]
        path = self.root / r["path"]
        if (path.stat().st_size, path.stat().st_mtime_ns) != (r["bytes"], r["mtime_ns"]):
            raise ValueError(f"Monthly source changed: {path}")
        with xr.open_dataset(path, engine="netcdf4", cache=False) as ds:
            index = pd.Index(ds.feature_id.values)
            if (
                not index.is_unique
                or hash_array(index.to_numpy(dtype="<i8")) != r["ordered_ids_sha256"]
            ):
                raise ValueError(f"Monthly COMID order changed: {path}")
            positions = index.get_indexer(self.ids)
            if (positions < 0).any():
                raise ValueError("Graph reach missing in monthly source")
            selected = ds.isel(feature_id=positions)
            shape = (r["steps"], len(self.ids))
            x, xm = np.empty(shape + (8,), dtype=np.float32), np.empty(shape + (8,), dtype=bool)
            for i, variable in enumerate(WEATHER):
                a = selected[variable].values.astype(np.float64)
                valid = np.isfinite(a) & (selected[variable + "_valid_hours"].values == 6)
                if i == 0:
                    a = transform_precipitation(a, valid, self.precip_transform)
                x[..., i] = np.where(valid, (a - self.mean[i]) / self.std[i], 0).astype(np.float32)
                xm[..., i] = valid
            q = selected.streamflow.values.astype(np.float64)
            qm = np.isfinite(q) & (selected.streamflow_valid_hours.values == 6)
            if (q[qm] < 0).any():
                raise ValueError("Negative valid streamflow")
            q = np.where(qm, (np.log1p(np.where(qm, q, 0)) - self.mean[8]) / self.std[8], 0).astype(
                np.float32
            )
        result = x, xm, q, qm
        self.cache[number] = result
        while len(self.cache) > self.cache_months:
            self.cache.popitem(last=False)
        return result

    def window(self, start, stop):
        parts = [[], [], [], []]
        cursor = start
        while cursor < stop:
            number = int(np.searchsorted(self.offsets, cursor, side="right") - 1)
            end = min(stop, int(self.offsets[number + 1]))
            left, right = cursor - self.offsets[number], end - self.offsets[number]
            for p, a in zip(parts, self.month(number)):
                p.append(a[left:right])
            cursor = end
        return tuple(np.concatenate(p, axis=0) for p in parts)
