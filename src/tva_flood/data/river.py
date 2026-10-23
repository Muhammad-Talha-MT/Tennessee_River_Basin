"""Whole upstream basins of every reach: area weights, travel distances and closed patches.

A graph block sees a few reaches; a TVA headwater gauge drains a median of 181
reaches up to 37 hops away. ``build`` writes, for every reach i and every reach j
upstream of it (i itself included):

- ``weight[i, j] = a_j f_ij / A_i``: local catchment area a_j over the basin area
  A_i = sum_j a_j f_ij, where f_ij is the share of j's water that reaches i (1
  except below divergences, where water splits equally between the branches);
- ``distance[i, j]``: river distance in km from the middle of reach j to the outlet
  of reach i, along the main downstream path.

Each row sums to one, so ``weight @ x`` is the basin mean of x. The model splits
the rows into distance bins and learns a travel-time kernel for each.

A node set that holds every upstream reach of its members is *closed*. On a closed
set the operator rows are complete, so basin means, upstream-only graph blocks
and routing all equal their full-network values (``tests/test_river.py``).
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tva_flood.common import atomic_json

# River distance bins (km) for travel time. About 20 to 40 km of river per six hours.
DISTANCE_BINS = (0.0, 5.0, 15.0, 35.0, 75.0, 150.0, 300.0, np.inf)
MIN_AREA_KM2 = 0.01


def graph_key(ids, edges):
    digest = hashlib.sha256()
    digest.update(np.asarray(ids, dtype=np.int64).tobytes())
    digest.update(np.asarray(edges, dtype=np.int64).tobytes())
    return digest.hexdigest()[:16]


def topological_order(n, edges):
    """Node order with every upstream reach before its downstream reaches."""
    indegree = np.bincount(edges[:, 1], minlength=n)
    children = sp.csr_matrix(
        (np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])), shape=(n, n)
    )
    order, frontier = [], np.flatnonzero(indegree == 0)
    while len(frontier):
        order.append(frontier)
        reached = children[frontier].indices
        np.subtract.at(indegree, reached, 1)
        frontier = np.unique(reached[indegree[reached] == 0])
    order = np.concatenate(order) if order else np.empty(0, dtype=np.int64)
    if len(order) != n:
        raise ValueError("River graph has a cycle")
    return order


def build_operator(n, edges, area, length):
    """Upstream rows as CSR: indptr, indices, weight (area share), distance (km)."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    area = np.maximum(np.asarray(area, dtype=np.float64), MIN_AREA_KM2)
    length = np.asarray(length, dtype=np.float64)
    order = topological_order(n, edges)
    outdegree = np.bincount(edges[:, 0], minlength=n)
    parents = sp.csr_matrix(
        (np.ones(len(edges), dtype=bool), (edges[:, 1], edges[:, 0])), shape=(n, n)
    )
    # Distance from each reach's downstream end to the network outlet, main path only.
    down = np.full(n, -1)
    down[edges[::-1, 0]] = edges[::-1, 1]  # first listed downstream edge wins
    to_outlet = np.zeros(n)
    for i in order[::-1]:
        if down[i] >= 0:
            to_outlet[i] = to_outlet[down[i]] + length[down[i]]
    members, shares = [None] * n, [None] * n
    for i in order:
        ups = parents.indices[parents.indptr[i] : parents.indptr[i + 1]]
        if not len(ups):
            members[i], shares[i] = np.array([i]), np.array([1.0])
            continue
        index = [np.array([i])] + [members[u] for u in ups]
        share = [np.array([1.0])] + [shares[u] / outdegree[u] for u in ups]
        index, share = np.concatenate(index), np.concatenate(share)
        if len(ups) > 1:
            index, inverse = np.unique(index, return_inverse=True)
            share = np.bincount(inverse, weights=share)
        members[i], shares[i] = index, share
    counts = np.array([len(m) for m in members], dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    indices = np.concatenate(members).astype(np.int32)
    share = np.concatenate(shares)
    rows = np.repeat(np.arange(n), counts)
    mass = area[indices] * share
    basin_area = np.bincount(rows, weights=mass, minlength=n)
    weight = (mass / basin_area[rows]).astype(np.float32)
    distance = to_outlet[indices] + length[indices] / 2 - to_outlet[rows]
    distance = np.maximum(distance, 0.0).astype(np.float32)
    return dict(
        indptr=indptr,
        indices=indices,
        weight=weight,
        distance=distance,
        basin_area_km2=basin_area,
        upstream_reaches=counts,
    )


class River:
    """The cached operator plus closure, closed-basin sampling and patch operators."""

    def __init__(self, path, ids, edges):
        with np.load(path) as data:
            if str(data["graph_key"]) != graph_key(ids, edges):
                raise ValueError(f"{path} was built for a different graph; rerun build-routing")
            self.indptr, self.indices = data["indptr"], data["indices"]
            self.weight, self.distance = data["weight"], data["distance"]
            self.upstream_reaches = data["upstream_reaches"]
            self.basin_area = data["basin_area_km2"]
        self.n = len(ids)
        self.bin = (np.searchsorted(DISTANCE_BINS, self.distance, side="right") - 1).astype(np.int8)
        self.matrix = sp.csr_matrix(
            (np.arange(len(self.indices), dtype=np.int64), self.indices, self.indptr),
            shape=(self.n, self.n),
        )
        self.path = Path(path)

    @classmethod
    def load(cls, root, ids, edges):
        path = Path(root) / "processed/river_routing" / f"{graph_key(ids, edges)}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run: tva-flood build-routing")
        return cls(path, ids, edges)

    def closure(self, selected):
        """Boolean mask of the selected reaches and every reach upstream of them."""
        rows = np.flatnonzero(np.asarray(selected, dtype=bool))
        result = np.zeros(self.n, dtype=bool)
        if len(rows):
            result[self.matrix[rows].indices] = True
        return result

    def sample_basins(self, rng, size):
        """A closed node set of about ``size`` reaches made of whole basins.

        Each basin is the closure of one outlet reach drawn with probability
        proportional to the square root of its basin size. Basins larger than
        ``size`` are never drawn: their reaches cannot be trained exactly.
        """
        eligible = np.flatnonzero(self.upstream_reaches <= size)
        probability = np.sqrt(self.upstream_reaches[eligible].astype(np.float64))
        probability /= probability.sum()
        chosen = np.zeros(self.n, dtype=bool)
        for _ in range(200):
            if chosen.sum() >= 0.9 * size:
                break
            outlet = int(rng.choice(eligible, p=probability))
            basin = self.indices[self.indptr[outlet] : self.indptr[outlet + 1]]
            if chosen.sum() + (~chosen[basin]).sum() > size:
                continue
            chosen[basin] = True
        return chosen

    def operator(self, nodes, bins=True):
        """Routing operator on a closed, sorted node set (positions in ``nodes``)."""
        nodes = np.asarray(nodes, dtype=np.int64)
        sub = self.matrix[nodes][:, nodes].tocoo()
        position = sub.data
        full_rows = self.matrix[nodes]
        if len(position) != full_rows.nnz:
            raise ValueError("Routing needs a closed node set (every upstream reach present)")
        return Routing(
            sub.row,
            sub.col,
            self.weight[position],
            self.bin[position] if bins else np.zeros(len(position), dtype=np.int8),
            len(nodes),
            len(DISTANCE_BINS) - 1 if bins else 1,
        )


class Routing:
    """Basin-mean operators of a closed node set, one sparse matrix per distance bin."""

    def __init__(self, rows, cols, weight, bins, n, count, device="cpu"):
        import torch

        self.n, self.count = n, count
        self.parts = []
        for b in range(count):
            keep = bins == b
            indices = torch.as_tensor(np.stack([rows[keep], cols[keep]]), dtype=torch.long)
            values = torch.as_tensor(weight[keep], dtype=torch.float32)
            self.parts.append(
                torch.sparse_coo_tensor(indices, values, (n, n)).coalesce().to(device)
            )

    def to(self, device):
        moved = Routing.__new__(Routing)
        moved.n, moved.count = self.n, self.count
        moved.parts = [p.to(device) for p in self.parts]
        return moved

    def __len__(self):
        return self.count


def build(a):
    root = a.root.resolve()
    prep = (a.preparation or root / "processed/pretraining_full").resolve()
    with np.load(prep / "graph.npz") as graph:
        ids = graph["feature_id"].copy()
        edges = np.column_stack([graph["source"], graph["target"]])
    nodes = pd.read_parquet(
        root / "processed/graph_candidate/nodes.parquet", columns=["comid", "areasqkm", "lengthkm"]
    ).set_index("comid")
    nodes = nodes.loc[ids]
    if nodes.isna().any().any():
        raise ValueError("Missing area or length for a graph reach")
    result = build_operator(len(ids), edges, nodes.areasqkm.to_numpy(), nodes.lengthkm.to_numpy())
    out = root / "processed/river_routing"
    out.mkdir(parents=True, exist_ok=True)
    key = graph_key(ids, edges)
    np.savez(out / f"{key}.npz", graph_key=np.array(key), **result)
    counts = result["upstream_reaches"]
    atomic_json(
        out / f"{key}.json",
        dict(
            graph=str(prep / "graph.npz"),
            reaches=len(ids),
            upstream_pairs=int(counts.sum()),
            median_upstream_reaches=float(np.median(counts)),
            max_upstream_reaches=int(counts.max()),
            distance_bins_km=[float(b) for b in DISTANCE_BINS[:-1]] + ["inf"],
            min_area_km2=MIN_AREA_KM2,
        ),
    )
    print(f"River routing: {out / f'{key}.npz'}; {int(counts.sum()):,} upstream pairs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--preparation", type=Path)
    build(p.parse_args())


if __name__ == "__main__":
    main()
