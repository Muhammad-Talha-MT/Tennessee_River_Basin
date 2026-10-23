"""Held-out gauge folds: the test of forecasting where no flow is observed.

Each CAMELS-H gauge with observations before 2019 is assigned to one of ``folds``
folds, stratified by drainage area (consecutive gauges in area order get a random
permutation of fold numbers) so every fold spans small to large basins. A model
trained for fold k never sees the flow of that fold's gauges: no targets, no flow
history as input, no checkpoint selection on them. At test time their history is
hidden too, and only they are scored. Gauges outside the CAMELS-H set or without
training-period data are in no fold (0).
"""

import numpy as np
import pandas as pd

TRAIN_END = pd.Timestamp("2019-01-01")


def holdout_folds(root, catalog, folds, seed=0):
    """Fold number (1..folds) of each catalog gauge, 0 for gauges never held out."""
    if folds < 2:
        raise ValueError("Need at least two folds")
    nodes = pd.read_parquet(
        root / "processed/graph_candidate/nodes.parquet", columns=["comid", "totdasqkm"]
    ).set_index("comid")
    area = nodes.totdasqkm.reindex(catalog.comid).to_numpy(dtype=float)
    trained = np.isfinite(catalog.observed[catalog.times < TRAIN_END]).any(axis=0)
    eligible = np.flatnonzero(catalog.camelsh & trained & np.isfinite(area))
    order = eligible[np.argsort(area[eligible], kind="stable")]
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(catalog.comid), dtype=np.int64)
    for start in range(0, len(order), folds):
        block = order[start : start + folds]
        fold[block] = rng.permutation(folds)[: len(block)] + 1
    return fold


def held_out(root, catalog, fold, folds, seed=0):
    """Boolean [gauge] mask of the gauges hidden in ``fold`` (all False when fold is 0)."""
    if not fold:
        return np.zeros(len(catalog.comid), dtype=bool)
    if not 1 <= fold <= folds:
        raise ValueError(f"Fold must be in 1..{folds}")
    return holdout_folds(root, catalog, folds, seed) == fold
