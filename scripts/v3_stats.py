"""Phase 1.4-1.7: paired tests, equivalence tests, per-basin blocked results, stratification, dam categories, lead-time curves.

Models: M0, M1, M2, LSTM (area-scaled LSTM, L4) and NWM (reference only). Metric: per-gauge NSE; "days 1-5" = mean over the leads 24, 48, 72, 96, 120 h
(per gauge for paired tests; the median over gauges of each lead, then the mean over the five leads, for medians).
Writes reports/v3_tables/*.csv. Run: python scripts/v3_stats.py
"""
import itertools
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
from v3_common import *  # noqa: F401,F403

OUT.mkdir(parents=True, exist_ok=True)
ctx = Context()
nwm_t, nwm_v = load_nwm()
nwm_sc = scores(nwm_t, nwm_v)
AREA_EDGES = [0, 200, 600, 2000, np.inf]  # km2; near the 25th, 60th and 90th percentiles of the gauged-protocol gauge areas
AREA_LABELS = ["< 200", "200-600", "600-2,000", "> 2,000"]
PAIRS = [("M0", "LSTM"), ("M2", "LSTM"), ("M1", "LSTM"), ("M0", "M2"), ("M0", "M1"), ("M1", "M2")]
EQUIV = [("M0", "M2"), ("M0", "M1"), ("M1", "M2")]

sc, common, data = {}, {}, {}
for proto in PROTOCOLS:
    data[proto] = load_protocol(proto)
    sc[proto] = {m: scores(*data[proto][m]) for m in data[proto]}
    sc[proto]["NWM"] = nwm_sc
    common[proto] = np.all([np.isfinite(sc[proto][m]["nse"][3]) for m in data[proto]], axis=0)
    assert common[proto].sum() == N_GAUGES[proto], (proto, common[proto].sum())
area_bin = np.digitize(ctx.area, AREA_EDGES[1:-1])  # 0..3


# ---- 1.5 paired tests ---------------------------------------------------------------------------------------------------
rows = []
for proto in PROTOCOLS:
    per = {m: days15(sc[proto][m]["nse"])[common[proto]] for m in sc[proto] if m != "NWM"}
    table = []
    for a, b in PAIRS:
        if a not in per or b not in per:
            rows.append(dict(protocol=proto, model_a=a, model_b=b, note="not run"))
            continue
        d = per[a] - per[b]
        mean, lo, hi = boot_mean(d, q=(5, 95))
        t_lo, t_hi = stats.t.interval(0.90, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
        table.append(dict(protocol=proto, model_a=a, model_b=b, gauges=len(d), mean_diff=mean, median_diff=float(np.median(d)), a_better=int((d > 0).sum()),
                          wilcoxon_p=wilcoxon_p(d), boot90_lo=lo, boot90_hi=hi, t90_lo=float(t_lo), t90_hi=float(t_hi),
                          tost_p_0p01=tost(d, 0.01), tost_p_0p02=tost(d, 0.02)))
    t = pd.DataFrame(table)
    t["wilcoxon_p_holm"] = holm(t.wilcoxon_p.to_numpy())  # across all paired tests of this protocol's table
    eq = t.model_a.str.cat(t.model_b, sep="-").isin([f"{a}-{b}" for a, b in EQUIV])
    for col in ("tost_p_0p01", "tost_p_0p02"):
        t.loc[eq, col + "_holm"] = holm(t.loc[eq, col].to_numpy())  # across the equivalence tests at that margin
    t["equivalent_0p01"] = eq & (t.tost_p_0p01_holm < 0.05)
    t["equivalent_0p02"] = eq & (t.tost_p_0p02_holm < 0.05)
    rows += t.to_dict("records")
paired = pd.DataFrame(rows)
paired.to_csv(OUT / "paired_tests.csv", index=False)
print(paired.drop(columns=["note"], errors="ignore").round(3).to_string(index=False))

# ---- blocked protocol: per basin (fold) ----------------------------------------------------------------------------------
rows = []
for fold in (1, 2, 3):
    gs = common["blocked"] & (ctx.fold["blocked"] == fold)
    row = dict(blocked_basin=fold, gauges=int(gs.sum()), median_area_km2=float(np.median(ctx.area[gs])))
    for m in ("M0", "M1", "M2", "LSTM", "NWM"):
        row[f"median_nse_d15_{m}"] = days15_of_median(sc["blocked"][m]["nse"], gs)
    for a in ("M0", "M1", "M2"):
        d = (days15(sc["blocked"][a]["nse"]) - days15(sc["blocked"]["LSTM"]["nse"]))[gs]
        row[f"mean_diff_{a}_minus_LSTM"] = float(d.mean())
        row[f"{a}_better_gauges"] = int((d > 0).sum())
        row[f"sign_{a}_minus_LSTM_median_nse"] = "+" if row[f"median_nse_d15_{a}"] > row["median_nse_d15_LSTM"] else "-"
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "blocked_by_basin.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---- 1.4 stratification --------------------------------------------------------------------------------------------------
up_vis = {p: ctx.upstream_visible(p) for p in PROTOCOLS}
has_up_any = np.array([len(ctx.upstream_gauges(g)) > 0 for g in range(ctx.gauges.count)])
has_up_vis = {p: np.array([len(up_vis[p].get(g, [])) > 0 for g in range(ctx.gauges.count)]) for p in PROTOCOLS}
MODELS_NWM = ["M0", "M1", "M2", "LSTM", "NWM"]
LEAD_SETS = {"6 h": [0], "24 h": [3], "72 h": [11], "120 h": [19], "240 h": [39], "days 1-5": DAYS15}


def med(proto, m, mask, leads):
    if m not in sc[proto]:  # M1 has no blocked runs yet
        return np.nan
    x = sc[proto][m]["nse"]
    with np.errstate(all="ignore"):
        return float(np.mean(np.nanmedian(x[leads][:, mask], axis=1)))


rows = []
for proto in PROTOCOLS:
    strata = []
    for i, lab in enumerate(AREA_LABELS):
        strata.append(("area km2", lab, common[proto] & (area_bin == i)))
    strata += [("upstream visible gauge", "yes", common[proto] & has_up_vis[proto]), ("upstream visible gauge", "no", common[proto] & ~has_up_vis[proto]),
               ("any upstream gauge (visible or withheld)", "yes", common[proto] & has_up_any), ("any upstream gauge (visible or withheld)", "no", common[proto] & ~has_up_any)]
    for g_name, mask in ctx.groups:
        strata.append(("dam category", g_name, common[proto] & mask))
    for kind, lab, mask in strata:
        for lead_label, leads in LEAD_SETS.items():
            row = dict(protocol=proto, stratum=kind, level=lab, gauges=int(mask.sum()), lead=lead_label)
            for m in MODELS_NWM:
                row[f"median_nse_{m}"] = med(proto, m, mask, leads) if mask.any() else np.nan
            rows.append(row)
strat = pd.DataFrame(rows)
strat.to_csv(OUT / "strata_median_nse.csv", index=False)

# area x any-upstream-gauge cross-tab and per-gauge differences (is the blocked "below a gauge" gap just drainage area?)
rows = []
for proto in PROTOCOLS:
    for i, lab in enumerate(AREA_LABELS):
        for nested in (False, True):
            mask = common[proto] & (area_bin == i) & (has_up_any == nested)
            if not mask.any():
                continue
            row = dict(protocol=proto, area_km2=lab, any_upstream_gauge=nested, gauges=int(mask.sum()))
            for m in ("M0", "M2", "LSTM"):
                row[f"median_nse_d15_{m}"] = days15_of_median(sc[proto][m]["nse"], mask)
            for a in ("M0", "M2"):
                d = (days15(sc[proto][a]["nse"]) - days15(sc[proto]["LSTM"]["nse"]))[mask]
                row[f"median_diff_{a}_minus_LSTM"] = float(np.median(d))
                row[f"{a}_better_gauges"] = int((d > 0).sum())
            rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "strata_area_by_upstream_gauge.csv", index=False)

# OLS: per-gauge (model - LSTM) days 1-5 NSE ~ log10(area) + any upstream gauge
rows = []
for proto in PROTOCOLS:
    mask = common[proto]
    X = np.column_stack([np.ones(mask.sum()), np.log10(ctx.area[mask]), has_up_any[mask].astype(float)])
    for a in ("M0", "M2"):
        y = (days15(sc[proto][a]["nse"]) - days15(sc[proto]["LSTM"]["nse"]))[mask]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = len(y) - X.shape[1]
        cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        p = 2 * stats.t.sf(np.abs(beta / se), dof)
        for name, b_, s_, p_ in zip(("intercept", "log10(area km2)", "any upstream gauge"), beta, se, p):
            rows.append(dict(protocol=proto, model=f"{a} - LSTM", term=name, coef=b_, se=s_, p=p_, gauges=len(y)))
ols = pd.DataFrame(rows)
ols.to_csv(OUT / "strata_ols_difference.csv", index=False)
print(ols.round(3).to_string(index=False))

# ---- 1.6 dam categories for every model, including the LSTM and NWM --------------------------------------------------------
rows = []
for proto in PROTOCOLS:
    for g_name, mask in ctx.groups:
        m_ = common[proto] & mask
        row = dict(protocol=proto, category=g_name, gauges=int(m_.sum()))
        for m in MODELS_NWM:
            row[f"days1-5_median_nse_{m}"] = med(proto, m, m_, DAYS15) if m_.any() else np.nan
        rows.append(row)
dam = pd.DataFrame(rows)
dam.to_csv(OUT / "by_dam_category.csv", index=False)
print(dam.round(3).to_string(index=False))

# ---- 1.7 lead curves with bootstrap confidence intervals ----------------------------------------------------------------
rows = []
for proto in PROTOCOLS:
    gs = np.flatnonzero(common[proto])
    for lead in range(40):
        row = dict(protocol=proto, lead_h=(lead + 1) * 6, gauges=len(gs))
        for m in MODELS_NWM:
            if m == "M1" and "M1" not in data[proto]:
                continue
            row[f"median_nse_{m}"] = float(np.nanmedian(sc[proto][m]["nse"][lead][gs]))
        for a in ("M0", "M2", "M1"):
            if a not in data[proto]:
                continue
            est, lo, hi = boot_median_diff(sc[proto][a]["nse"][lead][gs], sc[proto]["LSTM"]["nse"][lead][gs], n=4000)
            row[f"diff_median_{a}_minus_LSTM"], row[f"diff_{a}_minus_LSTM_ci95_lo"], row[f"diff_{a}_minus_LSTM_ci95_hi"] = est, lo, hi
            d = sc[proto][a]["nse"][lead][gs] - sc[proto]["LSTM"]["nse"][lead][gs]
            mean, lo, hi = boot_mean(d, n=4000, q=(2.5, 97.5))
            row[f"paired_mean_{a}_minus_LSTM"], row[f"paired_mean_{a}_minus_LSTM_ci95_lo"], row[f"paired_mean_{a}_minus_LSTM_ci95_hi"] = mean, lo, hi
        rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "lead_curves.csv", index=False)
pd.DataFrame({"edges_km2": AREA_EDGES[:-1] + ["inf"], "label": AREA_LABELS + [""]}).to_csv(OUT / "area_bins.csv", index=False)
print("done")
