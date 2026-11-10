"""Which gauges a model may learn from, and which it is tested on without ever seeing them.

Two held-out schemes (``scheme``):

- "random": the five area-stratified folds of CAMELS-H gauges (``data.folds.holdout_folds``), the existing
  ungauged test. Fold k withholds and scores that fold's gauges.
- "blocked": whole connected tributary subnetworks (experiment S1). A block is the complete upstream basin of an
  outlet gauge, so no water enters it from outside. Fold k withholds every catalog gauge inside block k (from
  targets, inputs and selection) and scores the CAMELS-H gauges inside it. Gauges downstream of a block stay in
  training (they see the block's outflow mixed with the rest: reported as a caveat, not hidden).

Gauge coverage (experiment S2): ``keep`` < 1 also withholds a random share 1 - keep of the remaining training
gauges (seeded by ``keep_seed`` and the fold), so the model learns from fewer gauges; only the fold's gauges are
scored, never the dropped ones.
"""

import numpy as np

from tva_flood.data.folds import holdout_folds

from . import paths as P

SCHEMES = ("random", "blocked")
RANDOM_FOLDS = 5
BLOCKS = 3  # blocked folds
BLOCK_GAUGES = (8, 30)  # scored gauges per block (inclusive)
BLOCK_AREA = 0.25  # largest share of the whole basin area in one block


def random_folds(gauges):
    return holdout_folds(P.ROOT, gauges.catalog, RANDOM_FOLDS)


def blocks(gauges, network):
    """Outlet gauge of each block and the block's reaches: up to ``BLOCKS`` disjoint upstream basins of CAMELS-H
    gauges, each holding ``BLOCK_GAUGES`` scorable gauges and at most ``BLOCK_AREA`` of the network area, the
    largest gauge counts first (deterministic)."""
    eligible = random_folds(gauges) > 0
    at = np.full(network.n, -1)
    at[gauges.reach] = np.arange(gauges.count)
    total_area = network.basin_area.max()
    candidates = []
    for g in np.flatnonzero(eligible):
        inside = at[gauges.basins[g]]
        inside = inside[inside >= 0]
        count = int(eligible[inside].sum())
        if BLOCK_GAUGES[0] <= count <= BLOCK_GAUGES[1] and gauges.area[g] <= BLOCK_AREA * total_area:
            candidates.append((-count, -float(gauges.area[g]), int(g)))
    chosen, taken = [], np.zeros(network.n, bool)
    for _, _, g in sorted(candidates):
        basin = gauges.basins[g]
        if taken[basin].any():
            continue
        chosen.append((g, np.asarray(basin)))
        taken[basin] = True
        if len(chosen) == BLOCKS:
            break
    if not chosen:
        raise ValueError("No tributary basin fits the block rules")
    return chosen


def gauge_folds(scheme, gauges, network=None):
    """Scored fold (1..k) of each gauge under ``scheme``, 0 for gauges never scored as held out."""
    if scheme == "random":
        return random_folds(gauges)
    if scheme != "blocked":
        raise ValueError(f"Unknown scheme {scheme}")
    eligible = random_folds(gauges) > 0
    fold = np.zeros(gauges.count, dtype=np.int64)
    for k, (_, basin) in enumerate(blocks(gauges, network), start=1):
        inside = np.isin(gauges.reach, basin)
        fold[inside & eligible] = k
    return fold


def fold_count(scheme, gauges, network=None):
    return RANDOM_FOLDS if scheme == "random" else len(blocks(gauges, network))


def withheld(gauges, network=None, fold=0, scheme="random", keep=1.0, keep_seed=0):
    """Boolean [gauge]: gauges the model for ``fold`` never sees (no targets, no history, no selection)."""
    out = np.zeros(gauges.count, bool)
    if fold:
        if scheme == "random":
            out = random_folds(gauges) == fold
        else:
            _, basin = blocks(gauges, network)[fold - 1]
            out = np.isin(gauges.reach, basin)  # every catalog gauge in the block, scored or not
    if keep < 1.0:
        candidates = np.flatnonzero(~out & gauges.trained)  # gauges the model would otherwise learn from
        rng = np.random.default_rng(10_000 * keep_seed + fold)
        drop = rng.permutation(candidates)[: int(round((1.0 - keep) * len(candidates)))]
        out = out.copy()
        out[drop] = True
    return out


def describe(split):
    """Short text of a checkpoint's split (``dict(scheme, keep, keep_seed)``)."""
    s = split or {}
    text = s.get("scheme", "random")
    if s.get("keep", 1.0) < 1.0:
        text += f", {100 * s['keep']:.0f}% of training gauges (draw {s.get('keep_seed', 0)})"
    return text
