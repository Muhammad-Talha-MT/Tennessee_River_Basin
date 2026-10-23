"""Sparse directed graph supports."""

import numpy as np


def sparse_supports(n, edges, device="cpu"):
    """Receiver-by-sender matrices for downstream and upstream propagation.

    Add one self loop per node, binary directed edges, source row normalization;
    second support propagates in the reverse direction. No N x N dense tensor.
    """
    import torch

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if (edges < 0).any() or (edges >= n).any():
        raise ValueError("Sparse edge outside graph")
    edges = np.unique(np.concatenate([edges, np.column_stack([np.arange(n)] * 2)]), axis=0)
    result = []
    for pairs in (edges, edges[:, ::-1]):
        source, target = pairs.T
        degree = np.bincount(source, minlength=n)
        indices = torch.as_tensor(np.stack([target, source]), dtype=torch.long, device=device)
        weights = torch.as_tensor(1.0 / degree[source], dtype=torch.float32, device=device)
        result.append(
            torch.sparse_coo_tensor(
                indices, weights, (n, n), device=device, check_invariants=True
            ).coalesce()
        )
    return result
