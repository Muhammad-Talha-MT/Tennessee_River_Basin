"""Basin-mean antecedent-storage features per gauge and day, for the LSTM ladder (--antecedent).

The graph models (A0-A6) read 7 antecedent-storage features per reach (EWMA rain over 7-365 days, temperature over 30 and 90
days; processed/hydro/antecedent.npy). A lumped model reads their area-weighted mean over the whole upstream basin
(hydro.model.LumpedModel.basin_average); this writes the same for the 128 catalog gauges, so the LSTM can be given the same
information: processed/hydro/basin_antecedent.npy [day, gauge, 7] (float16, from 1979-02-01, row = state at the start of the day).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tva_flood.data.gauges import Catalog  # noqa: E402
from tva_flood.hydro import paths as P  # noqa: E402
from tva_flood.hydro.data import Network  # noqa: E402

network = Network()
catalog = Catalog(P.GAUGES)
reach = pd.Index(network.ids).get_indexer(catalog.comid)
if (reach < 0).any():
    raise SystemExit("a gauge is not a reach of the graph")
rows, cols, vals = [], [], []
for g, r in enumerate(reach):
    members = network.members[network.indptr[r] : network.indptr[r + 1]]
    share = network.share[network.indptr[r] : network.indptr[r + 1]]
    w = share * network.area[members]
    w = w / w.sum()
    rows.append(np.full(len(members), g)), cols.append(members), vals.append(w)
W = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(reach), network.n), dtype=np.float32)
ant = np.load(P.HYDRO / "antecedent.npy", mmap_mode="r")
out = np.lib.format.open_memmap(P.HYDRO / "basin_antecedent.npy", mode="w+", dtype=np.float16, shape=(ant.shape[0], len(reach), ant.shape[2]))
began = time.perf_counter()
for start in range(0, ant.shape[0], 128):
    block = np.asarray(ant[start : start + 128], dtype=np.float32)  # [d, reach, 7]
    d = block.shape[0]
    flat = block.transpose(1, 0, 2).reshape(network.n, d * block.shape[2])
    out[start : start + d] = (W @ flat).reshape(len(reach), d, block.shape[2]).transpose(1, 0, 2).astype(np.float16)
    if start % 2048 == 0:
        print(f"day {start}/{ant.shape[0]}  {time.perf_counter() - began:.0f} s", flush=True)
out.flush()
info = json.loads((P.HYDRO / "antecedent.json").read_text())
info.update(gauges=catalog.names.tolist(), note="area-weighted basin means (LumpedModel.basin_average weights) at the catalog gauges")
(P.HYDRO / "basin_antecedent.json").write_text(json.dumps(info))
print("written", P.HYDRO / "basin_antecedent.npy", out.shape)
