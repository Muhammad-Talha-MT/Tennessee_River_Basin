"""Phase 1.3: KGE validation, KGE decomposition (r, alpha, beta) and high-flow measures from the saved forecasts.

KGE variant: Kling-Gupta efficiency of Gupta et al. (2009): KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2), alpha = std_sim / std_obs,
beta = mean_sim / mean_obs (not the 2012 variant, whose alpha is the ratio of coefficients of variation).
High-flow measures use the 2-year flow as the flood threshold: the median of the gauge's annual maxima (water years, Oct-Sep) of the 6-hourly flow
before 2019-01-01, for gauges with at least 10 water years with >= 70% of the values observed.

Writes reports/v3_tables/{kge_validation,kge_components,highflow_metrics,highflow_gauges}.csv. Run: python scripts/v3_kge_highflow.py
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from v3_common import *  # noqa: F401,F403

OUT.mkdir(parents=True, exist_ok=True)
ctx = Context()
nwm_truth, nwm_values = load_nwm()
LEAD_SETS = {"6 h": [0], "24 h": [3], "72 h": [11], "120 h": [19], "240 h": [39], "days 1-5": DAYS15}

# ---- 1. validate the KGE implementation against an independent numpy implementation --------------------------------------
data = load_protocol("gauged", ("M0", "M1", "M2", "LSTM"))
rows = []
for m in ("M0", "LSTM"):
    truth, values = data[m]
    sc = scores(truth, values)
    done = 0
    for gi in np.flatnonzero(np.isfinite(sc["kge"][3])):
        for lead in (0, 3, 11, 19, 39):
            o, f = truth[:, lead, gi].astype(float), values[:, lead, gi].astype(float)
            ok = np.isfinite(o) & np.isfinite(f)
            o, f = o[ok], f[ok]
            r = np.corrcoef(o, f)[0, 1]
            alpha = np.std(f) / np.std(o)
            beta = np.mean(f) / np.mean(o)
            kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
            rows.append(dict(model=m, gauge=ctx.names[gi], lead_h=(lead + 1) * 6, kge_v3=sc["kge"][lead, gi], kge_independent=kge,
                             abs_diff=abs(sc["kge"][lead, gi] - kge)))
        done += 1
val = pd.DataFrame(rows)
val.to_csv(OUT / "kge_validation.csv", index=False)
print(f"KGE validation: {val.gauge.nunique()} gauges x 5 leads x 2 models, max |difference| = {val.abs_diff.max():.2e}")

# ---- 2. decomposition ---------------------------------------------------------------------------------------------------
comp_rows = []
for proto in PROTOCOLS:
    d = load_protocol(proto)
    sc = {m: scores(*d[m]) for m in d}
    sc["NWM"] = scores(nwm_truth, nwm_values)
    common = np.all([np.isfinite(sc[m]["nse"][3]) for m in d], axis=0)
    for m, s in sc.items():
        for label, leads in LEAD_SETS.items():
            row = dict(protocol=proto, model=m, lead=label, gauges=int(common.sum()))
            for k in ("kge", "r", "alpha", "beta", "nse"):
                with np.errstate(all="ignore"):
                    row[f"median_{k}"] = float(np.mean(np.nanmedian(s[k][leads][:, common], axis=1)))
            comp_rows.append(row)
comp = pd.DataFrame(comp_rows)
comp.to_csv(OUT / "kge_components.csv", index=False)
print(comp[comp.lead == "days 1-5"].round(3).to_string(index=False))

# ---- 3. high-flow measures ----------------------------------------------------------------------------------------------
g = ctx.gauges
obs, times = g.observed, ctx.arrays.times
train = times < pd.Timestamp("2019-01-01")
wy = (times[train] + pd.DateOffset(months=3)).year.to_numpy()
q2 = np.full(g.count, np.nan)
for gi in range(g.count):
    series = pd.Series(obs[train, gi])
    ok = series.notna()
    by = series.groupby(wy)
    good = (by.count() / by.size() >= 0.7)
    maxima = by.max()[good]
    if len(maxima) >= 10:
        q2[gi] = float(np.median(maxima))
print(f"2-year flow available for {int(np.isfinite(q2).sum())} of {g.count} gauges")


p95 = np.array([np.nanpercentile(obs[train, gi], 95) if np.isfinite(obs[train, gi]).sum() > 1000 else np.nan for gi in range(g.count)])
PEAK_BINS = {"peak <= 24 h": (0, 4), "peak 24-72 h": (4, 12), "peak 72-120 h": (12, 20), "peak 120-240 h": (20, 40)}


def csi_of(o, f, thr, min_events):
    hit = (o >= thr) & (f >= thr)
    miss = (o >= thr) & (f < thr)
    fa = (o < thr) & (f >= thr)
    return hit.sum() / max(hit.sum() + miss.sum() + fa.sum(), 1) if (o >= thr).sum() >= min_events else np.nan


def highflow(truth, values, gauges):
    """Per gauge: peak-flow bias (top 5% of observed flows), CSI at the 2-year flow and at the 95th percentile, and peak timing and
    magnitude error in 10-day windows whose observed peak is a flood (>= 2-year flow), by the lead of the observed peak."""
    out = []
    for gi in gauges:
        o, f = truth[:, :, gi].astype(float), values[:, :, gi].astype(float)  # [issue, lead]
        row = dict(gauge=ctx.names[gi], q2=q2[gi], p95=p95[gi])
        for label, leads in LEAD_SETS.items():
            bias, c2, c95, ev2 = [], [], [], []
            for lead in leads:
                ok = np.isfinite(o[:, lead]) & np.isfinite(f[:, lead])
                oo, ff = o[ok, lead], f[ok, lead]
                if len(oo) < 100:
                    continue
                top = oo >= np.percentile(oo, 95)
                bias.append(ff[top].mean() / oo[top].mean() - 1)
                if np.isfinite(q2[gi]):
                    c2.append(csi_of(oo, ff, q2[gi], 5)); ev2.append((oo >= q2[gi]).sum())
                if np.isfinite(p95[gi]):
                    c95.append(csi_of(oo, ff, p95[gi], 20))
            row[f"peak_bias_{label}"] = np.mean(bias) if bias else np.nan
            row[f"csi_2yr_{label}"] = np.nan if not c2 or np.isnan(c2).any() else np.mean(c2)
            row[f"csi_p95_{label}"] = np.nan if not c95 or np.isnan(c95).any() else np.mean(c95)
            row[f"flood_pairs_2yr_{label}"] = float(np.mean(ev2)) if ev2 else 0.0
        full = np.isfinite(o).all(1) & np.isfinite(f).all(1)
        if np.isfinite(q2[gi]) and full.any():
            ow, fw = o[full], f[full]
            peak_o, peak_f = ow.argmax(1), fw.argmax(1)
            flood = ow.max(1) >= q2[gi]
            for label, (lo, hi) in PEAK_BINS.items():
                sel = flood & (peak_o >= max(lo, 1)) & (peak_o < min(hi, 39))
                row[f"events_{label}"] = int(sel.sum())
                if sel.sum() >= 3:
                    err = (peak_f[sel] - peak_o[sel]) * 6.0
                    row[f"timing_abs_h_{label}"] = float(np.median(np.abs(err)))
                    row[f"timing_signed_h_{label}"] = float(np.median(err))
                    row[f"magnitude_rel_error_{label}"] = float(np.median(fw[sel].max(1) / ow[sel].max(1) - 1))
        out.append(row)
    return pd.DataFrame(out)


per_gauge, summary = [], []
for proto in PROTOCOLS:
    d = load_protocol(proto)
    d["NWM"] = (nwm_truth, nwm_values)
    sc = {m: scores(*d[m]) for m in d if m != "NWM"}
    common = np.flatnonzero(np.all([np.isfinite(sc[m]["nse"][3]) for m in sc], axis=0))
    for m, (t, v) in d.items():
        hf = highflow(t, v, common)
        hf.insert(0, "model", m); hf.insert(0, "protocol", proto)
        per_gauge.append(hf)
        for label in LEAD_SETS:
            summary.append(dict(protocol=proto, model=m, measure=label, gauges=len(common),
                                median_peak_bias_top5pct=float(np.nanmedian(hf[f"peak_bias_{label}"])),
                                csi_2yr=float(np.nanmedian(hf[f"csi_2yr_{label}"])) if hf[f"csi_2yr_{label}"].notna().any() else np.nan,
                                csi_2yr_gauges=int(hf[f"csi_2yr_{label}"].notna().sum()),
                                csi_p95=float(np.nanmedian(hf[f"csi_p95_{label}"])) if hf[f"csi_p95_{label}"].notna().any() else np.nan,
                                csi_p95_gauges=int(hf[f"csi_p95_{label}"].notna().sum())))
        for label in PEAK_BINS:
            c = f"timing_abs_h_{label}"
            summary.append(dict(protocol=proto, model=m, measure=label, gauges=len(common),
                                timing_gauges=int(hf[c].notna().sum()) if c in hf else 0,
                                timing_median_abs_h=float(np.nanmedian(hf[c])) if c in hf and hf[c].notna().any() else np.nan,
                                timing_median_signed_h=float(np.nanmedian(hf[f"timing_signed_h_{label}"])) if c in hf and hf[c].notna().any() else np.nan,
                                magnitude_median_rel_error=float(np.nanmedian(hf[f"magnitude_rel_error_{label}"])) if c in hf and hf[c].notna().any() else np.nan))
pd.concat(per_gauge).to_csv(OUT / "highflow_gauges.csv", index=False)
hs = pd.DataFrame(summary)
hs.to_csv(OUT / "highflow_metrics.csv", index=False)
print(hs[hs.measure == "days 1-5"].round(3).to_string(index=False))
print(hs[hs.measure.str.startswith("peak")].round(2).to_string(index=False))
