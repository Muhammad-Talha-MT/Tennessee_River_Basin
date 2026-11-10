#!/usr/bin/env python3
"""Full Tennessee pretraining: manifest, resumable normalization, lazy loader and sparse GWNet adapter.

Run next to 14_prepare_static_attributes.py. Uses existing processed monthly data.
Targets: training Feb 1979--Dec 2018, validation 2019; later years untouched.
Precipitation defaults to log1p of six-hour totals in mm, then standardization.
Use --replace-preparation to replace a previous configuration in the same --out.
This prepares inputs; it does not launch a training experiment.
"""
import argparse
from collections import OrderedDict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

WEATHER = ["APCP_surface", "TMP_2maboveground", "SPFH_2maboveground", "PRES_surface",
           "DSWRF_surface", "DLWRF_surface", "UGRD_10maboveground", "VGRD_10maboveground"]
HISTORY, HORIZON = 120, 40


def transform_precipitation(values, valid=None, mode="log1p"):
    """Transform raw six-hour mm totals; invalid values become zero placeholders.

    Called by both statistics fitting and the loader. Validity masks are retained
    separately; zero placeholders never participate in normalization statistics.
    """
    if mode not in ("none", "log1p"):
        raise ValueError(f"Unknown precipitation transform: {mode}")
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != values.shape or not np.isfinite(values[valid]).all():
        raise ValueError("Invalid precipitation mask or nonfinite valid values")
    if (values[valid] < 0).any():
        raise ValueError("Negative valid precipitation; inspect source instead of clipping")
    safe = np.where(valid, values, 0.)
    return np.log1p(safe) if mode == "log1p" else safe


def atomic_json(path, data):
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def hash_array(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def static_module():
    path = Path(__file__).with_name("14_prepare_static_attributes.py")
    spec = importlib.util.spec_from_file_location("pretraining_statics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_graph(root):
    path = root / "processed/graph_candidate/nodes.parquet"
    inventory = pd.read_parquet(root / "metadata/tennessee_nwm_reaches.parquet")
    if path.exists():
        nodes = pd.read_parquet(path).sort_values("node_index")
        if not np.array_equal(nodes.node_index, np.arange(len(nodes))):
            raise ValueError("Graph node_index must be contiguous")
        ids = nodes.comid.to_numpy(dtype=np.int64)
    else:
        ids = inventory.sort_values("nwm_feature_index").comid.to_numpy(dtype=np.int64)
    if len(np.unique(ids)) != len(ids) or set(ids) != set(inventory.comid):
        raise ValueError("Full graph must include every inventory COMID exactly once")
    edge_path = root / "processed/graph_candidate/edges.parquet"
    if edge_path.exists() and path.exists():
        frame = pd.read_parquet(edge_path)
        if not {"fromcomid", "tocomid"} <= set(frame):
            raise ValueError("Candidate edges missing COMIDs")
    else:
        frame = pd.read_parquet(root / "metadata/tennessee_nhd_connectivity.parquet")
        frame = frame[frame.fromcomid.isin(ids) & frame.tocomid.isin(ids)]
    index = pd.Index(ids)
    source = index.get_indexer(frame.fromcomid)
    target = index.get_indexer(frame.tocomid)
    if (source < 0).any() or (target < 0).any():
        raise ValueError("Edge outside full graph")
    if "source" in frame and "target" in frame and edge_path.exists() and path.exists():
        if not (np.array_equal(source, frame.source) and np.array_equal(target, frame.target)):
            raise ValueError("Candidate edge indices disagree with node COMIDs")
    edges = np.column_stack([source, target]).astype(np.int64)
    unique = np.unique(edges, axis=0)
    print(f"Graph: {len(ids):,} nodes, {len(unique):,} unique internal edges; {len(edges)-len(unique)} duplicate records collapsed")
    return ids, unique


def scan_archive(root, ids, start, stop):
    records = []
    for number, first in enumerate(pd.date_range(start, stop, freq="MS", inclusive="left")):
        last = first + pd.offsets.MonthBegin(1)
        stamp = f"{first:%Y%m%dT%H}_{last:%Y%m%dT%H}"
        path = root / "processed/six_hour_nodes" / str(first.year) / f"tennessee_{stamp}_six_hour_nodes.nc"
        with xr.open_dataset(path, engine="netcdf4", cache=False) as ds:
            times = pd.DatetimeIndex(ds.time.values)
            expected = pd.date_range(first, last, freq="6h", inclusive="left")
            if not times.equals(expected):
                raise ValueError(f"Time mismatch: {path}")
            actual = pd.Index(ds.feature_id.values)
            if not actual.is_unique or set(actual) != set(ids):
                raise ValueError(f"COMID set mismatch: {path}")
            for v in WEATHER + ["streamflow"]:
                for key in (v, v + "_valid_hours"):
                    if ds[key].dims != ("time", "feature_id"):
                        raise ValueError(f"Unexpected dimensions for {key}: {path}")
            records.append(dict(path=str(path.relative_to(root)), start=str(first),
                                steps=len(times), bytes=path.stat().st_size,
                                mtime_ns=path.stat().st_mtime_ns,
                                ordered_ids_sha256=hash_array(actual.to_numpy(dtype="<i8"))))
        if (number + 1) % 48 == 0:
            print(f"Validated {number+1} monthly headers", flush=True)
    return records


def fit_normalization(root, records, out, identity, train_stop, block_steps,
                      precip_transform="log1p"):
    partial = out / "normalization_progress.json"
    state = dict(identity=identity, completed=0, count=[0]*9, mean=[0.]*9, m2=[0.]*9)
    if partial.exists():
        state = json.loads(partial.read_text())
        if state["identity"] != identity:
            raise ValueError("Normalization sources/settings changed. Use a fresh --out.")
    training = [r for r in records if pd.Timestamp(r["start"]) < train_stop]
    for month in range(state["completed"], len(training)):
        r = training[month]
        with xr.open_dataset(root / r["path"], engine="netcdf4", cache=False) as ds:
            for v, variable in enumerate(WEATHER + ["streamflow"]):
                for left in range(0, r["steps"], block_steps):
                    selection = dict(time=slice(left, left + block_steps))
                    x = ds[variable].isel(**selection).values
                    mask = np.isfinite(x) & (ds[variable + "_valid_hours"].isel(**selection).values == 6)
                    x = x[mask].astype(np.float64)
                    if variable == "APCP_surface":
                        x = transform_precipitation(x, mode=precip_transform)
                    if variable == "streamflow":
                        if (x < 0).any():
                            raise ValueError(f"Negative valid streamflow in {r['path']}")
                        x = np.log1p(x)
                    if len(x) == 0:
                        continue
                    n, avg = len(x), float(x.mean())
                    m2 = float(np.square(x - avg).sum())
                    old = state["count"][v]
                    total = old + n
                    delta = avg - state["mean"][v]
                    state["mean"][v] += delta * n / total
                    state["m2"][v] += m2 + delta * delta * old * n / total
                    state["count"][v] = total
        state["completed"] = month + 1
        atomic_json(partial, state)
        print(f"Normalization: {month+1}/{len(training)} training months ({r['start'][:7]})", flush=True)
    count = np.asarray(state["count"])
    if (count == 0).any():
        raise ValueError("A training feature has no valid observations")
    std = np.sqrt(np.maximum(np.asarray(state["m2"]) / count, 0))
    std[std < 1e-6] = 1
    result = dict(variables=WEATHER + ["log1p_streamflow"], mean=state["mean"],
                  precip_transform=precip_transform,
                  std=std.tolist(), count=state["count"], identity=identity)
    atomic_json(out / "dynamic_normalization.json", result)
    return result


class FullNetworkWindows:
    """Torch-compatible dataset with lazy monthly reads and explicit node selection.

    Returns x:[120,N,64], y/mask:[40,N,1]. Eight weather values+eight masks,
    followed by 24 static values+24 masks. No historic/future discharge inputs.
    node_indices=None means all graph nodes, never a pilot selection.
    Subgraph selection is explicit; target loss must be restricted to its core
    nodes by the training loop. Include enough halo hops for the chosen model.
    """
    def __init__(self, root, out, split="train", node_indices=None):
        self.root, out = Path(root), Path(out)
        self.manifest = json.loads((out / "dataset_manifest.json").read_text())
        if not self.manifest["training_ready"]:
            raise ValueError("Run full preparation without --headers-only first")
        self.normalization = json.loads((out / "dynamic_normalization.json").read_text())
        if self.normalization["identity"] != self.manifest["identity"]:
            raise ValueError("Dynamic normalization belongs to a different dataset")
        # Missing fields in the original schema mean raw precipitation, never log1p.
        self.precip_transform = self.manifest.get("precip_transform", "none")
        if self.normalization.get("precip_transform", "none") != self.precip_transform:
            raise ValueError("Precipitation transform differs between manifest and normalization")
        with np.load(out / "graph.npz") as graph:
            all_ids = graph["feature_id"].copy()
            graph_edges = np.column_stack([graph["source"], graph["target"]])
        if (hash_array(all_ids.astype("<i8")) != self.manifest["node_hash"]
                or hash_array(graph_edges.astype("<i8")) != self.manifest["edge_hash"]):
            raise ValueError("Prepared graph was modified")
        if static_module().sha(self.root / "processed/static/tennessee_static_core.nc") != self.manifest["static_sha256"]:
            raise ValueError("Static source was modified; rebuild preparation")
        self.nodes = np.arange(len(all_ids)) if node_indices is None else np.asarray(node_indices, dtype=np.int64)
        if self.nodes.ndim != 1 or not len(self.nodes) or len(np.unique(self.nodes)) != len(self.nodes):
            raise ValueError("Empty or duplicate node selection")
        if (self.nodes < 0).any() or (self.nodes >= len(all_ids)).any():
            raise ValueError("Node selection outside graph")
        self.ids = all_ids[self.nodes]
        self.static, _ = static_module().load_static_inputs(
            self.root / "processed/static/tennessee_static_core.nc", self.ids,
            normalization=json.loads((out / "static_normalization.json").read_text()))
        self.records = self.manifest["months"]
        self.offsets = np.r_[0, np.cumsum([r["steps"] for r in self.records])]
        self.times = pd.date_range(self.records[0]["start"], periods=int(self.offsets[-1]), freq="6h")
        if split not in self.manifest["splits"]:
            raise ValueError("Split must be train or validation; later test years are reserved")
        first, stop = map(pd.Timestamp, self.manifest["splits"][split])
        left = max(HISTORY, int(self.times.searchsorted(first)))
        right = int(self.times.searchsorted(stop)) - HORIZON + 1
        self.issues = np.arange(left, right, dtype=np.int64)
        self.positions = OrderedDict()

    def __len__(self):
        return len(self.issues)

    def read(self, start, stop, variables):
        result, masks = [], []
        cursor = start
        while cursor < stop:
            month = int(np.searchsorted(self.offsets, cursor, side="right") - 1)
            r = self.records[month]
            path = self.root / r["path"]
            if (path.stat().st_size, path.stat().st_mtime_ns) != (r["bytes"], r["mtime_ns"]):
                raise ValueError(f"Dynamic source changed: {path}")
            end = min(stop, int(self.offsets[month + 1]))
            with xr.open_dataset(path, engine="netcdf4", cache=False) as ds:
                if month not in self.positions:
                    index = pd.Index(ds.feature_id.values)
                    if not index.is_unique or hash_array(index.to_numpy(dtype="<i8")) != r["ordered_ids_sha256"]:
                        raise ValueError(f"Source COMID order changed: {path}")
                    positions = index.get_indexer(self.ids)
                    if (positions < 0).any():
                        raise ValueError("Selected node missing from dynamic file")
                    self.positions[month] = positions
                    while len(self.positions) > 32:
                        self.positions.popitem(last=False)
                part = ds.isel(time=slice(cursor-self.offsets[month], end-self.offsets[month]),
                               feature_id=self.positions[month])
                x = np.stack([part[v].values for v in variables], axis=-1).astype(np.float32)
                mask = np.isfinite(x) & np.stack([part[v+"_valid_hours"].values == 6 for v in variables], axis=-1)
                result.append(x); masks.append(mask)
            cursor = end
        return np.concatenate(result), np.concatenate(masks)

    def __getitem__(self, item):
        import torch
        issue = int(self.issues[item])
        x, valid = self.read(issue-HISTORY, issue, WEATHER)
        x = x.astype(np.float64)
        x[..., 0] = transform_precipitation(x[..., 0], valid[..., 0], self.precip_transform)
        mean, std = map(np.asarray, (self.normalization["mean"], self.normalization["std"]))
        x = np.where(valid, (x-mean[:8])/std[:8], 0).astype(np.float32)
        statics = np.broadcast_to(self.static, (HISTORY,) + self.static.shape)
        x = np.concatenate([x, valid.astype(np.float32), statics], axis=-1)
        q, mask = self.read(issue, issue+HORIZON, ["streamflow"])
        if (q[mask] < 0).any():
            raise ValueError("Negative valid flow in target window")
        y = np.where(mask, (np.log1p(np.where(mask, q, 0).astype(np.float64))-mean[8])/std[8], 0).astype(np.float32)
        return dict(x=torch.from_numpy(x), y=torch.from_numpy(y), mask=torch.from_numpy(mask),
                    issue=issue, feature_id=torch.from_numpy(self.ids.copy()))


def sparse_supports(n, edges, device="cpu"):
    """Receiver-by-sender matrices; equals transpose of legacy doubletransition.

    Add one self loop per node, binary directed edges, source row normalization;
    second support propagates in the reverse direction. No N x N dense tensor.
    """
    import torch
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if (edges < 0).any() or (edges >= n).any():
        raise ValueError("Sparse edge outside graph")
    edges = np.unique(np.concatenate([edges, np.column_stack([np.arange(n)]*2)]), axis=0)
    result = []
    for pairs in (edges, edges[:, ::-1]):
        source, target = pairs.T
        degree = np.bincount(source, minlength=n)
        indices = torch.as_tensor(np.stack([target, source]), dtype=torch.long, device=device)
        weights = torch.as_tensor(1. / degree[source], dtype=torch.float32, device=device)
        result.append(torch.sparse_coo_tensor(indices, weights, (n,n), device=device,
                                             check_invariants=True).coalesce())
    return result


def sparse_nconv():
    import torch
    from torch import nn
    class SparseNConv(nn.Module):
        def forward(self, x, support):
            b, c, n, t = x.shape
            flat = x.permute(2,0,1,3).reshape(n, b*c*t)
            return torch.sparse.mm(support, flat).reshape(n,b,c,t).permute(1,2,0,3).contiguous()
    return SparseNConv()


def make_sparse_gwnet(ids, edges, repo, device="cpu", in_dim=64):
    """Reuse GWNet temporal gates, skip paths and heads; replace graph propagation.

    This is an adapter factory, not a claim that full-network training fits GPU
    memory. Twenty order-two graph layers reach up to forty graph hops. Exact
    core-node evaluation on chunks needs that halo and full-graph edge weights;
    simply renormalizing the induced subgraph is not equivalent. BatchNorm statistics
    depend on the selected chunk. A training schedule must account for this.
    """
    import torch
    from torch import nn
    repo = Path(repo).resolve()
    if not (repo / "gwnetmodel.py").is_file():
        raise FileNotFoundError(repo / "gwnetmodel.py")
    sys.path.insert(0, str(repo))
    from gwnetmodel import GWNet
    supports = sparse_supports(len(ids), edges, device)
    model = GWNet(device=device, num_nodes=len(ids), supports=supports, addaptadj=False,
                  in_dim=in_dim, out_dim=40, blocks=4, layers=5,
                  residual_channels=8, dilation_channels=8, skip_channels=32, end_channels=64)
    for layer in model.gconv:
        layer.nconv = sparse_nconv()
    for name in ("gate_convs", "residual_convs", "skip_convs"):
        for i, layer in enumerate(getattr(model, name)):
            if isinstance(layer, nn.Conv1d):
                def pair(value):
                    return value if len(value) == 2 else (1, value[0])
                fixed = nn.Conv2d(layer.in_channels, layer.out_channels,
                                  pair(layer.kernel_size), stride=pair(layer.stride),
                                  padding=(0, layer.padding[0]) if len(layer.padding)==1 else layer.padding,
                                  dilation=pair(layer.dilation), groups=layer.groups,
                                  bias=layer.bias is not None)
                fixed.load_state_dict(layer.state_dict())
                getattr(model, name)[i] = fixed
    return model.to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--out", type=Path)
    p.add_argument("--train-start", default="1979-02-01")
    p.add_argument("--train-stop", default="2019-01-01")
    p.add_argument("--validation-stop", default="2020-01-01")
    p.add_argument("--time-block", type=int, default=128)
    p.add_argument("--precip-transform", choices=["none", "log1p"], default="log1p")
    p.add_argument("--replace-preparation", action="store_true",
                   help="Replace existing preparation metadata/statistics if configuration changed; same-config runs resume")
    p.add_argument("--headers-only", action="store_true", help="Build manifest without scanning values; not training-ready")
    args = p.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "processed/pretraining_full").resolve()
    start, stop, val_stop = map(pd.Timestamp, [args.train_start, args.train_stop, args.validation_stop])
    if any(t.tzinfo is not None or t != t.to_period("M").start_time for t in (start,stop,val_stop)):
        p.error("Dates must be timezone-naive month starts")
    if not pd.Timestamp("1979-02-01") <= start < stop <= pd.Timestamp("2019-01-01") < val_stop <= pd.Timestamp("2020-01-01"):
        p.error("Training must stop by Jan 2019; validation is in 2019; later tests remain reserved")
    if args.time_block < 1:
        p.error("time-block must be positive")
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / ".preparation.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"Another preparation process is using {out}. Wait for it to finish or stop that process before replacing.") from None
    ids, edges = read_graph(root)
    statics = static_module()
    static_path = root / "processed/static/tennessee_static_core.nc"
    with xr.open_dataset(static_path, engine="netcdf4") as ds:
        if set(ds.feature_id.values) != set(ids) or not pd.Index(ds.feature_id.values).is_unique:
            raise ValueError("Static and full graph COMID sets differ")
        names = ds.static_feature.values.tolist()
        if len(names) != 24 or not np.array_equal(ds.static_valid.astype(bool), np.isfinite(ds.static_values)):
            raise ValueError("Unexpected static schema or mask")
    # Temporal split: all these network nodes are training nodes. For geographic
    # transfer, build a separate preparation using an explicit held-out-node policy.
    _, normalization = statics.load_static_inputs(static_path, ids, fit_ids=ids)
    records = scan_archive(root, ids, start, val_stop)
    identity_config = dict(train=[str(start),str(stop)], validation=["2019-01-01 00:00:00",str(val_stop)],
                           node_hash=hash_array(ids.astype("<i8")), edge_hash=hash_array(edges.astype("<i8")),
                           static_sha256=statics.sha(static_path), months=records, schema=2,
                           precip_transform=args.precip_transform)
    identity = hashlib.sha256(json.dumps(identity_config, sort_keys=True).encode()).hexdigest()
    manifest_path = out / "dataset_manifest.json"
    stale = []
    for name in ("dataset_manifest.json", "normalization_progress.json", "dynamic_normalization.json"):
        path = out / name
        if path.exists() and json.loads(path.read_text()).get("identity") != identity:
            stale.append(path)
    if stale and not args.replace_preparation:
        raise ValueError("Existing preparation differs. Use --replace-preparation to replace it in this --out, or choose a new --out.")
    if stale:
        # Mark preparation unusable before resetting stats; retain the lock inode.
        if manifest_path.exists():
            old_manifest = json.loads(manifest_path.read_text())
            old_manifest["training_ready"] = False
            atomic_json(manifest_path, old_manifest)
        for name in ("normalization_progress.json", "dynamic_normalization.json"):
            (out / name).unlink(missing_ok=True)
        print(f"Replacing preparation in {out}; recalculating normalization from month 1", flush=True)
    print(f"Precipitation: {args.precip_transform} on six-hour mm totals, then training-only standardization", flush=True)
    total_steps = sum(r["steps"] for r in records if pd.Timestamp(r["start"]) < stop)
    validation_steps = int((val_stop-pd.Timestamp("2019-01-01"))/pd.Timedelta(hours=6))
    manifest = dict(identity=identity, nodes=len(ids), edges=len(edges), months=records,
                    schema=2, precip_transform=args.precip_transform,
                    precip_reference_mm=1.0,
                    node_hash=identity_config["node_hash"], edge_hash=identity_config["edge_hash"],
                    static_sha256=identity_config["static_sha256"],
                    history=HISTORY, horizon=HORIZON, dynamic_features=WEATHER,
                    static_features=names, input_features=16+2*len(names),
                    splits=dict(train=[str(start),str(stop)], validation=["2019-01-01",str(val_stop)]),
                    train_windows=max(0,total_steps-HISTORY-HORIZON+1),
                    validation_windows=max(0,validation_steps-HORIZON+1),
                    inputs="Historical weather+mask; static descriptors+mask. No future observations.",
                    targets="Six-hour NWM mean streamflow, log1p then shared training normalization.",
                    graph_note="NHDPlus topology; NWM routing equivalence still unverified.",
                    landcover="Dated snapshots excluded from invariant core inputs.",
                    training_ready=False)
    # npz contains only O(N+E) IDs and edge arrays.
    temp = out / "graph.part.npz"
    np.savez_compressed(temp, feature_id=ids, source=edges[:,0], target=edges[:,1])
    os.replace(temp, out / "graph.npz")
    atomic_json(out / "static_normalization.json", normalization)
    atomic_json(manifest_path, manifest)
    if args.headers_only:
        print("HEADERS CHECKED. Run without --headers-only to fit training normalization.")
        return
    fit_normalization(root, records, out, identity, stop, args.time_block, args.precip_transform)
    manifest["training_ready"] = True
    atomic_json(manifest_path, manifest)
    print(f"FULL DATA READY: nodes={len(ids):,}; input features={manifest['input_features']}; train windows={manifest['train_windows']:,}; validation windows={manifest['validation_windows']:,}")
    print(f"Saved: {out}. Sparse GWNet factory is available for the training script; no optimizer run.")


if __name__ == "__main__":
    main()
