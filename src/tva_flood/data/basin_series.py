"""Basin-mean weather and attributes for every catalog gauge, for lumped baselines.

A lumped model (the LSTM baseline) sees each gauge's whole upstream basin as one
unit: here, the area-weighted mean (``data.river`` rows of the gauges' reaches)
of the 8 normalized AORC values and their 8 validity flags at every six-hour
step, and of the 48 normalized static attributes. The GNN reads the same weather
reach by reach, so both see identical information.

Output ``basin_series.npz`` next to the preparation: ``times``, ``weather``
[time, gauge, 16] float32, ``static`` [gauge, 48], ``gauge`` and ``comid``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tva_flood.data.gauges import Catalog
from tva_flood.data.river import River
from tva_flood.data.schema import HORIZON
from tva_flood.data.ssl import EncoderWindows
from tva_flood.data.transforms import transform_precipitation


def basin_rows(river, ids, comids):
    """Sparse [gauge, reach] basin-mean weights of the gauges' outlet reaches."""
    outlet = pd.Index(ids).get_indexer(comids)
    if (outlet < 0).any():
        raise ValueError("A gauge is not a reach of the graph")
    rows, cols, vals = [], [], []
    for g, o in enumerate(outlet):
        part = slice(river.indptr[o], river.indptr[o + 1])
        rows.append(np.full(part.stop - part.start, g))
        cols.append(river.indices[part])
        vals.append(river.weight[part])
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(comids), len(ids)),
    )


def covered_reaches(root, ids):
    """Reaches with AORC forcing (where rain inputs exist), in graph order."""
    mapping = np.load(root / "processed/aorc_mapping/grid_and_node_order.npz")
    return mapping["forcing_available"][pd.Index(mapping["node_comid"]).get_indexer(ids)]


def basin_forecast(field, rows, covered, rain, steps):
    """Future rain inputs [gauge, lead, 2] of a lumped model from a forecast per reach.

    ``field`` [step, reach] holds six-hour totals (mm, NaN where missing); rain is
    known within ``steps`` leads at covered reaches. Each reach value is normalized
    as the graph model's input, then averaged over the basin (``rows``), with the
    known share as the flag: the same as the basin series' AORC channels 0 and 8.
    """
    transform, mean, std = rain
    values = np.asarray(field, dtype=np.float64)[:steps]
    known = np.isfinite(values) & (values >= 0) & covered[None]
    z = (transform_precipitation(np.where(known, values, 0.0), known, transform) - mean) / std
    z = np.where(known, z, 0.0)
    out = np.zeros((rows.shape[0], HORIZON, 2), dtype=np.float32)
    out[:, : len(values), 0] = rows @ z.T
    out[:, : len(values), 1] = rows @ known.T.astype(np.float64)
    return out


def build(a):
    root = a.root.resolve()
    prep = (a.preparation or root / "processed/pretraining_full").resolve()
    split = a.split or ("test" if "test" in prep.name else "train")
    data = EncoderWindows(
        root, prep, split, HORIZON, 2, a.array_cache.resolve() if a.array_cache else None
    )
    with np.load(prep / "graph.npz") as graph:
        ids = graph["feature_id"].copy()
        edges = np.column_stack([graph["source"], graph["target"]])
    catalog = Catalog((a.gauges or root / "processed/gauges/gauges.nc").resolve())
    rows = basin_rows(River.load(root, ids, edges), ids, catalog.comid)
    weather = np.zeros((len(data.times), len(catalog), 16), dtype=np.float32)
    for number in range(len(data.records)):
        values, valid, _, _ = data.month(number)
        steps = values.shape[0]
        stacked = np.concatenate([values, valid.astype(np.float32)], axis=-1)  # [t, n, 16]
        flat = np.moveaxis(stacked, 1, 0).reshape(len(ids), -1)
        mean = (rows @ flat).reshape(len(catalog), steps, 16)
        start = int(data.offsets[number])
        weather[start : start + steps] = np.moveaxis(mean, 0, 1)
    out = prep / "basin_series.npz"
    np.savez(
        out,
        times=data.times.to_numpy(),
        weather=weather,
        static=(rows @ np.asarray(data.static, dtype=np.float64)).astype(np.float32),
        gauge=catalog.names,
        comid=catalog.comid,
    )
    print(f"{out}: {len(data.times)} steps x {len(catalog)} gauges")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--preparation", type=Path)
    p.add_argument("--array-cache", type=Path)
    p.add_argument("--split", help="Default: test for a test preparation, else train")
    p.add_argument("--gauges", type=Path, help="Default: <root>/processed/gauges/gauges.nc")
    build(p.parse_args())


if __name__ == "__main__":
    main()
