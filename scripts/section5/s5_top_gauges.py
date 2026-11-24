"""Section 5, top-scoring gauges (figure figS7_top_gauges). Rule, fixed before looking at the series: among the 101 gauges scored in the gauged protocol
(the GNN was trained on all of them; the same gauges are scored for the LSTM and the NWM), rank by the GNN's mean NSE over leads 24-120 h (days 1-5), highest first; keep the first TOP.
For each, a window of 2 x HALF days around the largest observed test-period flow at the gauge: the observed discharge and the forecasts of the GNN (M0), the LSTM and the NWM
valid at each daily 00Z issue + lead, for the leads 24 h and 72 h (one value per daily issue, so each line joins forecasts made on different days).
Writes reports/section5/s5_10_top_gauges.csv (ranking and scores) and s5_10_top_gauge_series.csv. Run: python scripts/section5/s5_top_gauges.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
import pandas as pd

import v3_common as V
from tva_flood.hydro import paths as P
from tva_flood.hydro.data import Arrays, Gauges, Network

TOP, HALF, LEADS = 6, 30, (24, 72)
PROTOCOL = "gauged"  # all 101 gauges scored by the all-gauge GNN (trained on every gauge); "heldout" would rank the 98 random hold-out gauges
OUT = V.REPO / "reports" / "section5"
net, arr = Network(), Arrays()
times = arr.times
G = Gauges(net, times)
names = np.array(G.names).astype(str)
assert list(names) == list(V.STATIONS)
gt = pd.read_csv(V.REPO / "figures/section3/gauge_table.csv", dtype={"usgs_id": str}).set_index("gauge")
nodes = pd.read_parquet(P.PROCESSED / "graph_candidate/nodes.parquet", columns=["comid", "totdasqkm"]).set_index("comid")
area = nodes.totdasqkm.reindex(net.ids[G.reach]).to_numpy(float)
data = V.load_protocol(PROTOCOL, ("M0", "LSTM"))
nwm = V.load_nwm()
data["NWM"] = nwm
sc = {m: V.scores(*data[m]) for m in data}
step = np.load(V.A / "eval_M0_s1" / "forecasts_gauged.npz", allow_pickle=True)["step"].astype(int)
d15 = {m: V.days15(sc[m]["nse"]) for m in sc}
scored = np.isfinite(sc["M0"]["nse"][3]) & np.isfinite(sc["LSTM"]["nse"][3]) & np.isfinite(sc["NWM"]["nse"][3])
assert int(np.isfinite(sc["M0"]["nse"][3]).sum()) == V.N_GAUGES[PROTOCOL]
order = [g for g in np.argsort(-np.where(scored, d15["M0"], -np.inf)) if scored[g]][:TOP]
test = (times >= pd.Timestamp("2020-09-23")) & (times < pd.Timestamp("2026-01-01"))
rank_rows, series = [], []
for k, g in enumerate(order, start=1):
    tidx = np.flatnonzero(test)
    peak = int(tidx[np.nanargmax(G.observed[test, g])])
    lo, hi = times[peak] - pd.Timedelta(days=HALF), times[peak] + pd.Timedelta(days=HALF)
    rank_rows.append(dict(rank=k, gauge=names[g], usgs_id=gt.loc[names[g], "usgs_id"], area_km2=area[g], peak_time=str(times[peak]), peak_flow=float(G.observed[peak, g]),
                          **{f"nse_{m}_days1_5": float(d15[m][g]) for m in ("M0", "LSTM", "NWM")},
                          **{f"nse_{m}_{h}h": float(sc[m]["nse"][h // 6 - 1][g]) for m in ("M0", "LSTM", "NWM") for h in LEADS}))
    for h in LEADS:
        li = h // 6 - 1  # a lead h = 6 (index + 1) is the interval starting at issue + 6 index (the repo convention)
        for i, s in enumerate(step):
            vt = times[s + li]
            if lo <= vt <= hi:
                series.append(dict(rank=k, gauge=names[g], lead_h=h, issue=str(times[s]), valid_time=str(vt), observed=float(G.observed[s + li, g]),
                                   **{m: float(data[m][1][i, li, g]) for m in ("M0", "LSTM", "NWM")}))
pd.DataFrame(rank_rows).to_csv(OUT / "s5_10_top_gauges.csv", index=False)
pd.DataFrame(series).to_csv(OUT / "s5_10_top_gauge_series.csv", index=False)
print(pd.DataFrame(rank_rows).round(3).to_string())
print("series rows:", len(series))
