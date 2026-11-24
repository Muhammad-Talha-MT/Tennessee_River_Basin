"""Section 5, forecast rain against realized rain (figure fig11_rain_source). The same trained GNN (M0) and the same LSTM (L4) are scored twice on the same gauges and issues:
with the GEFS forecast rain after the issue (the rule of every other result) and with the realized AORC rain in its place (a perfect-rain diagnostic, "+aorc"; nothing is
retrained). GNN +aorc: models/attr/eval_M0_s1_aorc (gauged, random hold-out) and eval_S1_M0_s1_aorc (blocked), made by scripts/m0_aorc_evaluate.sh. LSTM: methods lstm+gefs and
lstm+aorc of the saved ladder_L4_s1_bench* files. Gauges: the protocol's scored gauges (as the other figures), required to be scored in all four series.
Writes reports/section5/s5_11_rain_source_curves.csv (median NSE by lead), s5_11_rain_source_gain.csv (observed minus forecast rain, mean over gauges with a 90% gauge bootstrap,
by lead) and s5_11_rain_source_summary.csv (days 1-5: median NSE, mean gain, GNN gain minus LSTM gain). Run: python scripts/section5/s5_rain_source.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
import pandas as pd

import v3_common as V

OUT = V.REPO / "reports" / "section5"
A = V.A
PROTOS = list(V.PROTOCOLS)
AORC_DIR = {"gauged": "eval_M0_s1_aorc", "heldout": "eval_M0_s1_aorc", "blocked": "eval_S1_M0_s1_aorc"}


def gnn_aorc(p):
    f = "forecasts_gauged.npz" if p == "gauged" else "forecasts_ungauged.npz"
    d = np.load(A / AORC_DIR[p] / f, allow_pickle=True)
    assert (d["stations"].astype(str) == V.STATIONS).all()
    names = [str(m) for m in d["methods"]]
    j = [i for i, m in enumerate(names) if m.endswith("+aorc") and m.split("+")[0] in ("hydro", "ungauged")]
    assert len(j) == 1, names
    return d["truth"].astype(np.float32), d["values"][:, j[0]].astype(np.float32)


def lstm(p, method):
    """LSTM (L4) benchmark forecasts of ``method`` merged by station name (as v3_common._lstm, which is fixed to lstm+gefs)."""
    truth = values = None
    for name in V.LSTM_FILES[p]:
        d = np.load(A / name / "forecasts.npz", allow_pickle=True)
        names = list(d["methods"].astype(str))
        v = d["values"][:, names.index(method)].astype(np.float32)
        t = d["truth"].astype(np.float32)
        if truth is None:
            truth = np.full((t.shape[0], t.shape[1], len(V.STATIONS)), np.nan, np.float32)
            values = truth.copy()
        for i, st in enumerate(d["stations"].astype(str)):
            j = np.flatnonzero(V.STATIONS == st)
            if len(j) == 0:
                continue
            fin = np.isfinite(v[:, :, i])
            if fin.any():
                values[:, :, j[0]] = np.where(fin, v[:, :, i], values[:, :, j[0]])
                truth[:, :, j[0]] = np.where(fin, t[:, :, i], truth[:, :, j[0]])
    return truth, values


curves, gains, summ = [], [], []
for p in PROTOS:
    series = {("M0", "gefs"): V.load_protocol(p, ("M0",))["M0"], ("M0", "aorc"): gnn_aorc(p),
              ("LSTM", "gefs"): V.load_protocol(p, ("LSTM",))["LSTM"], ("LSTM", "aorc"): lstm(p, "lstm+aorc")}
    nse = {k: V.scores(*v)["nse"] for k, v in series.items()}  # [lead, gauge]
    ok = np.all([np.isfinite(x[3]) for x in nse.values()], axis=0)
    assert int(ok.sum()) == V.N_GAUGES[p], (p, int(ok.sum()), {k: int(np.isfinite(x[3]).sum()) for k, x in nse.items()})
    for (m, r), x in nse.items():
        for li in range(40):
            curves.append(dict(protocol=p, model=m, rain=r, n=int(ok.sum()), lead_h=6 * (li + 1), median_nse=float(np.nanmedian(x[li, ok]))))
    gain = {m: nse[(m, "aorc")] - nse[(m, "gefs")] for m in ("M0", "LSTM")}
    for m in ("M0", "LSTM"):
        for li in range(40):
            mean, lo, hi = V.boot_mean(gain[m][li, ok])
            gains.append(dict(protocol=p, model=m, n=int(ok.sum()), lead_h=6 * (li + 1), mean_gain=mean, ci5=lo, ci95=hi))
    d15 = {k: V.days15(x)[ok] for k, x in nse.items()}  # per-gauge mean NSE over 24-120 h
    row = dict(protocol=p, n=int(ok.sum()))
    for (m, r), v in d15.items():
        row[f"median_nse_{m}_{r}"] = float(np.median(v))
    for m in ("M0", "LSTM"):
        row[f"gain_{m}"], row[f"gain_{m}_ci5"], row[f"gain_{m}_ci95"] = V.boot_mean(d15[(m, "aorc")] - d15[(m, "gefs")])
    row["gain_M0_minus_LSTM"], row["gain_M0_minus_LSTM_ci5"], row["gain_M0_minus_LSTM_ci95"] = V.boot_mean((d15[("M0", "aorc")] - d15[("M0", "gefs")]) - (d15[("LSTM", "aorc")] - d15[("LSTM", "gefs")]))
    for r in ("gefs", "aorc"):
        row[f"M0_minus_LSTM_{r}"], row[f"M0_minus_LSTM_{r}_ci5"], row[f"M0_minus_LSTM_{r}_ci95"] = V.boot_mean(d15[("M0", r)] - d15[("LSTM", r)])
    summ.append(row)
pd.DataFrame(curves).to_csv(OUT / "s5_11_rain_source_curves.csv", index=False)
pd.DataFrame(gains).to_csv(OUT / "s5_11_rain_source_gain.csv", index=False)
S = pd.DataFrame(summ)
S.to_csv(OUT / "s5_11_rain_source_summary.csv", index=False)
print(S.round(3).T.to_string())
