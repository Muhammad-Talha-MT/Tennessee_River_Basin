"""Connected river patches with an exact halo for graph-size-independent encoders.

The encoder's graph blocks use receiver-by-sender supports with full-graph
degree normalization. Restricting those matrices to a node set leaves every
row exact whose neighbours all belong to the set. With ``halo`` equal to the
number of graph blocks, every core node therefore receives exactly its
full-graph encoder output. LayerNorm has no batch statistics, so the result
does not depend on which other nodes are present. Losses use core nodes only.

With a ``river`` (``data.river.River``) the model passes messages upstream to
downstream only and routes whole basins, so the exact patch is instead *closed*:
whole upstream basins, no halo. ``sample`` then draws whole basins, and every
subgraph carries the basin routing operator as ``supports[2]``.
"""

import numpy as np
import scipy.sparse as sp
import torch


def undirected_adjacency(n, edges):
    """Symmetric boolean CSR adjacency without self loops."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if (edges < 0).any() or (edges >= n).any():
        raise ValueError("Edge outside graph")
    edges = edges[edges[:, 0] != edges[:, 1]]
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    matrix = sp.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))
    matrix.sum_duplicates()
    matrix.data[:] = True
    return matrix


def expand(adjacency, selected, hops):
    """Nodes within ``hops`` undirected steps of a boolean selection."""
    result = np.asarray(selected, dtype=bool).copy()
    for _ in range(hops):
        reached = adjacency @ result.astype(np.float32) > 0
        if not (reached & ~result).any():
            break
        result |= reached
    return result


def to_scipy(support):
    support = support.coalesce().cpu()
    rows, cols = support.indices().numpy()
    return sp.csr_matrix((support.values().numpy(), (rows, cols)), shape=tuple(support.shape))


def to_torch(matrix, device):
    matrix = matrix.tocoo()
    indices = torch.as_tensor(np.stack([matrix.row, matrix.col]), dtype=torch.long)
    values = torch.as_tensor(matrix.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce().to(device)


class SubgraphSampler:
    """Sample a connected core patch, add its halo and slice full-graph supports."""

    def __init__(self, n, edges, supports, halo, river=None):
        if halo < 0 or len(supports) != 2:
            raise ValueError("Need a nonnegative halo and forward/reverse supports")
        self.n, self.halo, self.river = n, halo, river
        self.adjacency = undirected_adjacency(n, edges)
        self.supports = [to_scipy(s) for s in supports]

    def patch(self, rng, size):
        """Breadth-first connected patch of exactly ``size`` nodes (if the graph allows)."""
        size = min(int(size), self.n)
        chosen = np.zeros(self.n, dtype=bool)
        while chosen.sum() < size:
            # A new seed is needed only when a disconnected component is exhausted.
            seed = rng.choice(np.flatnonzero(~chosen))
            chosen[seed] = True
            frontier = np.array([seed])
            while len(frontier) and chosen.sum() < size:
                neighbours = np.unique(self.adjacency[frontier].indices)
                neighbours = neighbours[~chosen[neighbours]]
                room = size - int(chosen.sum())
                if len(neighbours) > room:
                    # Each frontier neighbour touches the patch, so any subset stays connected.
                    neighbours = rng.choice(neighbours, room, replace=False)
                chosen[neighbours] = True
                frontier = neighbours
        return chosen

    def sample(self, rng, size, device="cpu"):
        if self.river is not None:
            core = self.river.sample_basins(rng, size)
            nodes = np.flatnonzero(core)
            return self.subgraph(nodes, core[nodes], device)
        core = self.patch(rng, size)
        selected = expand(self.adjacency, core, self.halo)
        nodes = np.flatnonzero(selected)
        return self.subgraph(nodes, core[nodes], device)

    def subgraph(self, nodes, core, device="cpu"):
        nodes = np.asarray(nodes, dtype=np.int64)
        supports = [to_torch(s[nodes][:, nodes], device) for s in self.supports]
        if self.river is not None:
            supports.append(self.river.operator(nodes).to(device))
        adjacency = self.adjacency[nodes][:, nodes].tocsr()
        return dict(
            nodes=nodes, core=np.asarray(core, dtype=bool), supports=supports, adjacency=adjacency
        )

    def full(self, device="cpu"):
        """Full-graph adjacency and supports; the full routing operator is built separately."""
        nodes = np.arange(self.n)
        supports = [to_torch(s, device) for s in self.supports]
        return dict(
            nodes=nodes,
            core=np.ones(self.n, dtype=bool),
            supports=supports,
            adjacency=self.adjacency,
        )

    def closed(self, selected):
        """Smallest exact node set around ``selected`` reaches (bool mask)."""
        if self.river is not None:
            return self.river.closure(selected)
        return expand(self.adjacency, selected, self.halo)
