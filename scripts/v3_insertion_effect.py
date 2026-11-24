"""Phase 1.1: the effect of insertion at test time, from M0 with and without insertion (same weights) and from M1.

Needs models/attr/eval_M0_s1_insertoff (M0 evaluated with ``--insertion-off``). Writes reports/v3_tables/insertion_effect_*.csv.
Run: python scripts/v3_insertion_effect.py
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
from v3_common import *  # noqa: F401,F403
from v3_common import _hydro

OUT.mkdir(parents=True, exist_ok=True)
ctx = Context()
off_dir = A / "eval_M0_s1_insertoff"
assert (off_dir / "forecasts_ungauged.npz").exists(), "run the insertion-off evaluation first"
cov = pd.read_csv(OUT / "insertion_upstream_visible_by_gauge.csv")
rows, per_gauge = [], []

# ---- gauged protocol: a gauge is visible at an issue only if it has >= 8 values in the last 32 steps (Samples.eligible) ------------------------
steps = np.load(A / "eval_M0_s1" / "forecasts_gauged.npz", allow_pickle=True)["step"]
recent_ok = np.stack([np.isfinite(ctx.gauges.observed[t - 32 : t]).sum(0) >= 8 for t in steps])  # [issue, gauge]: visible in the gauged protocol
t_on, v_on = _hydro(A / "eval_M0_s1" / "forecasts_gauged.npz")
t_off, v_off = _hydro(off_dir / "forecasts_gauged.npz")
d = np.abs(v_on - v_off)  # [issue, lead, gauge]
vis = np.broadcast_to(recent_ok[:, None, :], d.shape)
ok = np.isfinite(d)
for label, mask in (("visible at the issue", vis & ok), ("not visible at the issue (no recent flow)", ~vis & ok)):
    x = d[mask]
    rows.append(dict(protocol="gauged", measure=f"gauge-issue-lead values {label}: share of all values", value=float(mask.sum() / ok.sum())))
    rows.append(dict(protocol="gauged", measure=f"gauge-issue-lead values {label}: median |change from insertion| (m3/s)", value=float(np.median(x))))
    rows.append(dict(protocol="gauged", measure=f"gauge-issue-lead values {label}: 99.9th percentile of |change| (m3/s)", value=float(np.percentile(x, 99.9))))
    rows.append(dict(protocol="gauged", measure=f"gauge-issue-lead values {label}: max |change| (m3/s)", value=float(x.max())))
    rows.append(dict(protocol="gauged", measure=f"gauge-issue-lead values {label}: share changed by more than 1e-3 m3/s", value=float((x > 1e-3).mean())))

# ---- held-out protocol ---------------------------------------------------------------------------------------------------
t_on, v_on = _hydro(A / "eval_M0_s1" / "forecasts_ungauged.npz")
_, v_off = _hydro(off_dir / "forecasts_ungauged.npz")
_, v_m1 = _hydro(A / "eval_M1_s1" / "forecasts_ungauged.npz")
sc = {"M0": scores(t_on, v_on), "M0 insertion off": scores(t_on, v_off), "M1": scores(t_on, v_m1)}
scored = np.all([np.isfinite(s["nse"][3]) for s in sc.values()], axis=0)
c = cov[cov.protocol == "heldout"].set_index("gauge")
has_up = np.array([n in c.index and c.loc[n, "n_upstream_visible"] > 0 for n in ctx.names]) & scored
coverage = np.array([c.loc[n, "covered_area_share"] if n in c.index else np.nan for n in ctx.names])
share = np.array([c.loc[n, "share_issues_with_visible_upstream"] if n in c.index else np.nan for n in ctx.names])
mean_flow = np.nanmean(np.where(np.isfinite(t_on), t_on, np.nan), axis=(0, 1))

for label, mask in (("all held-out gauges", scored), ("with a visible gauge upstream", has_up), ("no visible gauge upstream", scored & ~has_up)):
    for h in (6, 24, "days 1-5"):
        i = None if h == "days 1-5" else h // 6 - 1
        get = (lambda s: days15(s["nse"])) if i is None else (lambda s, i=i: s["nse"][i])
        a, b, m1 = get(sc["M0"])[mask], get(sc["M0 insertion off"])[mask], get(sc["M1"])[mask]
        rows.append(dict(protocol="heldout", subset=label, gauges=int(mask.sum()), lead=h,
                         median_nse_M0=float(np.median(a)), median_nse_M0_insertion_off=float(np.median(b)), median_nse_M1=float(np.median(m1)),
                         mean_diff_M0_minus_insertion_off=float((a - b).mean()), wilcoxon_p_M0_vs_off=wilcoxon_p(a - b),
                         mean_diff_M0_minus_M1=float((a - m1).mean()), wilcoxon_p_M0_vs_M1=wilcoxon_p(a - m1)))

# how large is the correction? mean absolute change of the forecast caused by insertion, relative to the gauge's mean flow
for h in (6, 24, 120):
    i = h // 6 - 1
    change = np.nanmean(np.abs(v_on[:, i, :] - v_off[:, i, :]), axis=0)
    rel = change / mean_flow
    rows.append(dict(protocol="heldout", subset="with a visible gauge upstream", lead=h, gauges=int(has_up.sum()),
                     median_abs_forecast_change_share_of_mean_flow=float(np.nanmedian(rel[has_up])),
                     p90_abs_forecast_change_share_of_mean_flow=float(np.nanpercentile(rel[has_up], 90))))
    rows.append(dict(protocol="heldout", subset="no visible gauge upstream", lead=h, gauges=int((scored & ~has_up).sum()),
                     median_abs_forecast_change_share_of_mean_flow=float(np.nanmedian(rel[scored & ~has_up])),
                     p90_abs_forecast_change_share_of_mean_flow=float(np.nanpercentile(rel[scored & ~has_up], 90))))

# does the effect grow with the share of the basin covered by upstream gauges?
for h in (6, 24, "days 1-5"):
    i = None if h == "days 1-5" else h // 6 - 1
    d = (days15(sc["M0"]["nse"]) - days15(sc["M0 insertion off"]["nse"])) if i is None else (sc["M0"]["nse"][i] - sc["M0 insertion off"]["nse"][i])
    ok = has_up & np.isfinite(d) & np.isfinite(coverage)
    rho, p = stats.spearmanr(coverage[ok], d[ok])
    rows.append(dict(protocol="heldout", subset="with a visible gauge upstream", lead=h, gauges=int(ok.sum()),
                     spearman_covered_area_share_vs_gain_from_insertion=float(rho), spearman_p=float(p)))
    for g in np.flatnonzero(ok):
        per_gauge.append(dict(gauge=ctx.names[g], lead=h, covered_area_share=coverage[g], share_issues_with_visible_upstream=share[g],
                              area_km2=ctx.area[g], nse_gain_from_insertion=float(d[g])))
df = pd.DataFrame(rows)
df.to_csv(OUT / "insertion_effect_summary.csv", index=False)
pd.DataFrame(per_gauge).to_csv(OUT / "insertion_effect_by_gauge.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(df.round(4).to_string(index=False))
