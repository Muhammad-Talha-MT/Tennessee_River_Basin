"""Gauged and held-out comparison of M0, M1, M2, A0 and LSTM-AS from the saved forecasts: NSE, KGE, paired tests."""
import itertools
import warnings

import numpy as np
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
A = "/data/tva_data/models/attr"
DAYS = [h // 6 - 1 for h in (24, 48, 72, 96, 120)]
LEADS = (6, 24, 72, 120, 240)


def scores(truth, value, min_count=100):
    """(NSE, KGE) [lead, station] over issues where both are finite; NaN where unscored (as the evaluation script)."""
    ok = np.isfinite(value) & np.isfinite(truth)
    n = ok.sum(0)
    o = np.where(ok, truth, 0.0); f = np.where(ok, value, 0.0)
    m = np.maximum(n, 1)
    mo, mf = o.sum(0) / m, f.sum(0) / m
    var_o = (o * o).sum(0) / m - mo**2
    so = np.sqrt(np.maximum(var_o, 0)); sf = np.sqrt(np.maximum((f * f).sum(0) / m - mf**2, 0))
    mse = ((f - o) ** 2).sum(0) / m
    good = (n >= min_count) & (mo > 0) & (so >= 1e-3)
    nse = 1 - mse / var_o
    r = ((o * f).sum(0) / m - mo * mf) / np.maximum(so * sf, 1e-12)
    kge = 1 - np.sqrt((r - 1) ** 2 + (sf / np.maximum(so, 1e-12) - 1) ** 2 + (mf / np.where(mo > 0, mo, np.nan) - 1) ** 2)
    return np.where(good, nse, np.nan), np.where(good, kge, np.nan)


def hydro(path):
    d = np.load(path, allow_pickle=True)  # method 0 is the GEFS forecast in every file
    nse, kge = scores(d["truth"].astype(np.float64), d["values"][:, 0].astype(np.float64))
    return nse, kge, d["stations"].astype(str)


def lstm(paths):
    """LSTM-AS benchmark forecasts: one file (gauged) or the fold files (held-out, one set of stations each), merged by name."""
    table = {}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        names = list(d["methods"].astype(str))
        nse, kge = scores(d["truth"].astype(np.float64), d["values"][:, names.index("lstm+gefs")].astype(np.float64))
        for i, st in enumerate(d["stations"].astype(str)):
            if np.isfinite(nse[3, i]):
                table[st] = (nse[:, i], kge[:, i])
    stations = np.array(sorted(table))
    return (np.stack([table[s][0] for s in stations], axis=1), np.stack([table[s][1] for s in stations], axis=1), stations)


def run(title, sources):
    data = {k: v() for k, v in sources.items()}
    common = set.intersection(*[set(s[2][np.isfinite(s[0][3])]) for s in data.values()])
    print(f"\n===== {title}: {len(common)} gauges scored by every model")
    sel = {}
    for k, (nse, kge, st) in data.items():
        keep = np.array([i for i, n in enumerate(st) if n in common]); order = np.argsort(st[keep])
        sel[k] = (nse[:, keep[order]], kge[:, keep[order]])
    for metric, idx in (("NSE", 0), ("KGE", 1)):
        print(f"\n{metric} (median over gauges)")
        print(f"{'model':8s} " + " ".join(f"{h:>6d}h" for h in LEADS) + "  days1-5")
        for k in sel:
            x = sel[k][idx]
            print(f"{k:8s} " + " ".join(f"{np.nanmedian(x[h // 6 - 1]):7.3f}" for h in LEADS) + f"  {np.mean(np.nanmedian(x, axis=1)[DAYS]):7.3f}")
    print("\npaired, days 1-5 NSE per gauge: mean diff (first - second), gauges first better, Wilcoxon p")
    per = {k: np.nanmean(v[0][DAYS], axis=0) for k, v in sel.items()}
    for a, b in [("M0", "A0"), ("M1", "A0"), ("M2", "A0"), ("M0", "M1"), ("M0", "M2"), ("M1", "M2"), ("M0", "L4"), ("M1", "L4"), ("M2", "L4"), ("A0", "L4")]:
        if a not in per or b not in per:
            continue
        d = per[a] - per[b]; d = d[np.isfinite(d)]
        print(f"  {a:3s} - {b:3s}  {d.mean():+.3f}  {int((d > 0).sum()):3d}/{len(d)}  p={wilcoxon(d).pvalue:.3f}")


run("Gauged", {
    "M0": lambda: hydro(f"{A}/eval_M0_s1/forecasts_gauged.npz"),
    "M1": lambda: hydro(f"{A}/eval_M1_s1/forecasts_gauged.npz"),
    "M2": lambda: hydro(f"{A}/eval_M2_s1/forecasts_gauged.npz"),
    "A0": lambda: hydro(f"{A}/eval_A0_s1/forecasts_gauged.npz"),
    "L4": lambda: lstm([f"{A}/ladder_L4_s1_bench/forecasts.npz"]),
})
run("Held-out (random folds)", {
    "M0": lambda: hydro(f"{A}/eval_M0_s1/forecasts_ungauged.npz"),
    "M1": lambda: hydro(f"{A}/eval_M1_s1/forecasts_ungauged.npz"),
    "M2": lambda: hydro(f"{A}/eval_M2_s1/forecasts_ungauged.npz"),
    "A0": lambda: hydro(f"{A}/eval_A0_s1/forecasts_ungauged.npz"),
    "L4": lambda: lstm([f"{A}/ladder_L4_s1_bench_f{k}/forecasts.npz" for k in range(1, 6)]),
})
run("Blocked", {
    "M0": lambda: hydro(f"{A}/eval_S1_M0_s1/forecasts_ungauged.npz"),
    "M2": lambda: hydro(f"{A}/eval_S1_M2_s1/forecasts_ungauged.npz"),
    "A0": lambda: hydro(f"{A}/eval_S1_A0_s1/forecasts_ungauged.npz"),
    "L4": lambda: lstm([f"{A}/ladder_L4_s1_bench_b{k}/forecasts.npz" for k in range(1, 4)]),
})
