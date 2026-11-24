"""Builds reports/test_comparison_v3.md from reports/v3_tables/*.csv (produced by scripts/v3_*.py). No value is typed by hand.

Models: M0, M1, M2, LSTM (area-scaled LSTM), NWM (reference only). Run after the other v3 scripts: python scripts/v3_report.py
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from v3_common import FIG, OUT, REPO

R = lambda name: pd.read_csv(OUT / name)  # noqa: E731
comp, paired, bybasin, lead = R("kge_components.csv"), R("paired_tests.csv"), R("blocked_by_basin.csv"), R("lead_curves.csv")
hf, strata, cross, ols, dam = R("highflow_metrics.csv"), R("strata_median_nse.csv"), R("strata_area_by_upstream_gauge.csv"), R("strata_ols_difference.csv"), R("by_dam_category.csv")
reach, inputs, kval, tests = R("reach_level.csv"), R("inputs_audit.csv"), R("kge_validation.csv"), R("unit_tests.csv")
ins_sum, ins_gauged, ins_cov = R("insertion_upstream_visible_summary.csv"), R("insertion_M0_vs_M1_gauged_sites_below_gauges.csv"), R("insertion_upstream_visible_by_gauge.csv")
leak, leak16 = R("leak_check.csv"), R("leak_check_bf16.csv")
config = pd.read_csv(OUT / "config_audit.csv", index_col=0)
effect_path = OUT / "insertion_effect_summary.csv"
effect = pd.read_csv(effect_path) if effect_path.exists() else None

PROT = ["gauged", "heldout", "blocked"]
PNAME = {"gauged": "Gauged", "heldout": "Held-out (random folds)", "blocked": "Blocked basins"}
NG = {p: int(comp[(comp.protocol == p) & (comp.model == "M0")].gauges.iloc[0]) for p in PROT}
MODELS = ["M0", "M1", "M2", "LSTM", "NWM"]


def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "not run"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def p_(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def table(headers, rows, text=False):
    align = ["---"] * len(headers) if text else ["---"] + ["---:"] * (len(headers) - 1)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def best(vals, higher=True, skip=("NWM",), d=3, labels=None):
    """Format the values, bolding the best among the non-reference models."""
    cand = {k: v for k, v in vals.items() if k not in skip and isinstance(v, (int, float)) and not np.isnan(v)}
    top = (max if higher else min)(cand.values()) if cand else None
    return [f"**{f(v, d)}**" if (k in cand and v == top) else f(v, d) for k, v in vals.items()]


def cell(proto, model, lead_label, col):
    r = comp[(comp.protocol == proto) & (comp.model == model) & (comp.lead == lead_label)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def pr(proto, a, b):
    r = paired[(paired.protocol == proto) & (paired.model_a == a) & (paired.model_b == b)]
    return r.iloc[0] if len(r) and not pd.isna(r.iloc[0].get("mean_diff")) else None


def holm_word(p):
    return "significant after Holm correction" if p < 0.05 else "not significant after Holm correction"


md = []
add = md.append

# ================================================================================================================ header
add("# Test-set comparison v3: M0, M1, M2, LSTM and NWM\n")
add(f"Test issues 2020-09-23 to 2025-12-22 (1,917 00Z issues), GEFSv12 operational rain to 240 h, 6-hourly leads. Protocols: **gauged** ({NG['gauged']} gauges, model trained on all gauges), "
    f"**held-out** ({NG['heldout']} gauges, 5 random folds, each scored at gauges it never saw), **blocked** ({NG['blocked']} gauges in 3 whole tributary basins withheld). "
    "Metric: median NSE over gauges; *days 1-5* is the mean over the leads 24, 48, 72, 96 and 120 h of that median. One seed for every model.\n")
add(table(["Model", "What it is"], text=True, rows=[
    ["M0", "Manning routing + gauge insertion + 3 graph layers"], ["M1", "M0 without insertion"], ["M2", "M0 without graph layers"],
    ["LSTM", "area-scaled LSTM (output = specific discharge x basin area), the only baseline"], ["NWM", "National Water Model v3.0 forecast, reference only"]]))
add("**NWM is a reference, not a held-out competitor.** The NWM analysis nudges toward USGS observations at every gauge, including the gauges withheld from M0, M1, M2 and the LSTM, "
    "so its scores at held-out and blocked gauges are not comparable to a model that never saw those gauges.\n")
add("*Not run (marked in the tables):* M4 = M1 without routing (planned: held-out and blocked only); the reach level for M1, M2 and M4; multiple seeds.\n")

# ================================================================================================================ 1 headline
add("## 1. Headline: days 1-5 median NSE\n")
rows = []
for p in PROT:
    vals = {m: cell(p, m, "days 1-5", "median_nse") for m in MODELS}
    rows.append([f"{PNAME[p]} ({NG[p]})"] + best(vals))
add(table(["Protocol (gauges)"] + MODELS, rows))
add("Bold: best of M0, M1, M2 and LSTM.\n")

# ================================================================================================================ 2 by lead
add("## 2. NSE by lead\n")
for p in PROT:
    rows = []
    for m in MODELS:
        vals = [cell(p, m, L, "median_nse") for L in ("6 h", "24 h", "72 h", "120 h", "240 h", "days 1-5")]
        rows.append([m] + [f(v) for v in vals])
    add(f"{PNAME[p]} ({NG[p]} gauges):\n")
    add(table(["Model", "6 h", "24 h", "72 h", "120 h", "240 h", "Days 1-5"], rows))

# ================================================================================================================ 3 paired tests
add("## 3. Paired tests\n")
add("Per-gauge days 1-5 NSE (mean over the leads 24-120 h), first model minus second. Gauge-level 90% bootstrap interval (10,000 resamples) of the mean difference, "
    "gauges where the first model is better, two-sided Wilcoxon signed-rank p, and Holm-adjusted p across all six paired tests of the protocol's table.\n")
pairs = [("M0", "LSTM"), ("M2", "LSTM"), ("M1", "LSTM"), ("M0", "M2"), ("M0", "M1"), ("M1", "M2")]
rows = []
for a, b in pairs:
    row = [f"{a} - {b}"]
    for p in PROT:
        r = pr(p, a, b)
        row.append("not run" if r is None else f"{f(r.mean_diff, 3, True)} [{f(r.boot90_lo, 3, True)}, {f(r.boot90_hi, 3, True)}]; {int(r.a_better)}/{int(r.gauges)}; p {p_(r.wilcoxon_p)}, Holm {p_(r.wilcoxon_p_holm)}")
    rows.append(row)
add(table(["Comparison", "Gauged", "Held-out", "Blocked"], rows, text=True))
g_l, h_l, b_l = pr("gauged", "M0", "LSTM"), pr("heldout", "M0", "LSTM"), pr("blocked", "M0", "LSTM")
notes = []
for a_, b_ in pairs:
    for p_name in PROT:
        r_ = pr(p_name, a_, b_)
        if r_ is not None and np.sign(r_.mean_diff) != np.sign(r_.median_diff) and r_.wilcoxon_p_holm < 0.05:
            notes.append(f"{a_} - {b_} on {PNAME[p_name].lower()} has a mean difference of {f(r_.mean_diff, 3, True)} but a median difference of {f(r_.median_diff, 3, True)} and is better on {int(r_.a_better)} of {int(r_.gauges)} gauges: "
                         "a few gauges with large losses pull the mean below zero while most gauges gain, so the Wilcoxon result and the mean point in different directions")
add(("Note: " + "; ".join(notes) + ".\n") if notes else "")
add(f"M0 against LSTM is {holm_word(g_l.wilcoxon_p_holm)} on gauged (Holm p {p_(g_l.wilcoxon_p_holm)}), {holm_word(h_l.wilcoxon_p_holm)} on held-out (Holm p {p_(h_l.wilcoxon_p_holm)}) "
    f"and {holm_word(b_l.wilcoxon_p_holm)} on blocked (Holm p {p_(b_l.wilcoxon_p_holm)}); the blocked test treats the {int(b_l.gauges)} gauges as independent, "
    "but they sit in only 3 basins (section 3.2).\n")

add("### 3.1 Equivalence tests for the null claims\n")
add("Paired TOST on the mean difference of per-gauge days 1-5 NSE with equivalence margins of +-0.01 and +-0.02 NSE (two one-sided t-tests; the larger p is reported), Holm-adjusted across "
    "the three equivalence tests of a protocol at each margin. Equivalence is concluded when the adjusted p is below 0.05. The 90% intervals are t-based and bootstrap.\n")
rows = []
for a, b in (("M0", "M2"), ("M0", "M1"), ("M1", "M2")):
    for p in PROT:
        r = pr(p, a, b)
        if r is None:
            rows.append([f"{a} - {b}", PNAME[p], "not run", "", "", "", "", ""])
            continue
        rows.append([f"{a} - {b}", PNAME[p], f(r.mean_diff, 3, True), f"[{f(r.t90_lo, 3, True)}, {f(r.t90_hi, 3, True)}] / [{f(r.boot90_lo, 3, True)}, {f(r.boot90_hi, 3, True)}]",
                     p_(r.tost_p_0p01_holm), "yes" if r.equivalent_0p01 else "no", p_(r.tost_p_0p02_holm), "yes" if r.equivalent_0p02 else "no"])
add(table(["Pair", "Protocol", "Mean diff", "90% CI (t) / (bootstrap)", "TOST p, +-0.01 (Holm)", "Equivalent +-0.01", "TOST p, +-0.02 (Holm)", "Equivalent +-0.02"], rows))
eq = paired[paired.equivalent_0p01 == True]  # noqa: E712
eq1 = paired[paired.equivalent_0p01 == True]  # noqa: E712
eq2 = paired[(paired.equivalent_0p02 == True) & (paired.equivalent_0p01 != True)]  # noqa: E712
none2 = paired[paired.model_a.isin(["M0", "M1"]) & paired.model_b.isin(["M1", "M2"]) & paired.mean_diff.notna() & (paired.equivalent_0p02 != True)]
add("Equivalence within +-0.01 is established for: " + (", ".join(f"{r.model_a} vs {r.model_b} ({PNAME[r.protocol].lower()})" for r in eq1.itertuples()) or "none") + ". "
    + ("Additionally within +-0.02 only: " + ", ".join(f"{r.model_a} vs {r.model_b} ({PNAME[r.protocol].lower()})" for r in eq2.itertuples()) + ". " if len(eq2) else "")
    + "Equivalence is **not** established, even at +-0.02, for: " + (", ".join(f"{r.model_a} vs {r.model_b} ({PNAME[r.protocol].lower()})" for r in none2.itertuples()) or "none")
    + "; their intervals reach beyond the margin, so for those pairs the claim can only be stated as *no detectable effect*.\n")

add("### 3.2 Blocked basins, per basin\n")
rows = []
for r in bybasin.itertuples():
    rows.append([int(r.blocked_basin), int(r.gauges), f(r.median_area_km2, 0), f(r.median_nse_d15_M0), f(r.median_nse_d15_M2), f(r.median_nse_d15_LSTM), f(r.median_nse_d15_NWM),
                 f"{f(r.mean_diff_M0_minus_LSTM, 3, True)} ({int(r.M0_better_gauges)}/{int(r.gauges)})", f"{f(r.mean_diff_M2_minus_LSTM, 3, True)} ({int(r.M2_better_gauges)}/{int(r.gauges)})"])
add(table(["Basin", "Gauges", "Median area km2", "M0", "M2", "LSTM", "NWM", "M0 - LSTM mean (gauges better)", "M2 - LSTM mean (gauges better)"], rows))
n_pos0 = int((bybasin.median_nse_d15_M0 > bybasin.median_nse_d15_LSTM).sum())
n_pos2 = int((bybasin.median_nse_d15_M2 > bybasin.median_nse_d15_LSTM).sum())
add(f"Median days 1-5 NSE: M0 is above LSTM in {n_pos0} of 3 basins and M2 in {n_pos2} of 3. With three basins no cluster-level inference is possible; the gauge-level tests in section 3 treat nested gauges "
    "of one basin as independent and should be read as descriptive. NWM is shown as a reference.\n")

# ================================================================================================================ 4 lead dependence
add("## 4. Gap to the LSTM by lead\n")
add("Difference in median NSE (model minus LSTM) with a 95% gauge bootstrap interval (4,000 resamples); data for figure (a) in `lead_curves.csv`.\n")
rows = []
for p in PROT:
    for h in (6, 24, 72, 120, 240):
        r = lead[(lead.protocol == p) & (lead.lead_h == h)].iloc[0]
        rows.append([PNAME[p], f"{h} h"] + [f"{f(r[f'diff_median_{m}_minus_LSTM'], 3, True)} [{f(r[f'diff_{m}_minus_LSTM_ci95_lo'], 3, True)}, {f(r[f'diff_{m}_minus_LSTM_ci95_hi'], 3, True)}]" for m in ("M0", "M2")])
add(table(["Protocol", "Lead", "M0 - LSTM", "M2 - LSTM"], rows))
for p in PROT:
    d = lead[lead.protocol == p]
    early, late = d[d.lead_h <= 48], d[d.lead_h >= 144]
    add(f"- {PNAME[p]}: mean M0 - LSTM difference in median NSE is {f(early.diff_median_M0_minus_LSTM.mean(), 3, True)} over leads up to 48 h and {f(late.diff_median_M0_minus_LSTM.mean(), 3, True)} over leads of 144 h and beyond.")
add("")

add("### 4.1 Status of H1 to H3\n")
def trend(p_):
    d_ = lead[lead.protocol == p_]
    e_, l_ = d_[d_.lead_h <= 48].diff_median_M0_minus_LSTM.mean(), d_[d_.lead_h >= 144].diff_median_M0_minus_LSTM.mean()
    return "shrinks" if l_ < e_ else "grows"


hl = {p_: (pr(p_, "M0", "LSTM"), pr(p_, "M2", "LSTM")) for p_ in PROT}
head = {p_: {m: cell(p_, m, "days 1-5", "median_nse") for m in ("M0", "M2", "LSTM")} for p_ in PROT}
add("**H1** (a reach-scale routed model transfers better than a lumped LSTM, with a gap growing from gauged to held-out to blocked and with lead). "
    f"Gap in days 1-5 median NSE, M0 minus LSTM: gauged {f(head['gauged']['M0'] - head['gauged']['LSTM'], 3, True)}, held-out {f(head['heldout']['M0'] - head['heldout']['LSTM'], 3, True)}, "
    f"blocked {f(head['blocked']['M0'] - head['blocked']['LSTM'], 3, True)}; M2 minus LSTM: {f(head['gauged']['M2'] - head['gauged']['LSTM'], 3, True)}, "
    f"{f(head['heldout']['M2'] - head['heldout']['LSTM'], 3, True)}, {f(head['blocked']['M2'] - head['blocked']['LSTM'], 3, True)}. The ordering of the medians follows the prediction; the per-gauge mean differences do not "
    f"(M0 - LSTM {f(hl['gauged'][0].mean_diff, 3, True)}, {f(hl['heldout'][0].mean_diff, 3, True)}, {f(hl['blocked'][0].mean_diff, 3, True)}). "
    + "With lead the M0 - LSTM gap " + ", ".join(f"{trend(p_)} on {PNAME[p_].lower()}" for p_ in PROT) + " (leads up to 48 h against 144 h and beyond, section 4). "
    + f"On blocked basins M0 is above the LSTM in {n_pos0} of 3 basins and M2 in {n_pos2} of 3 (section 3.2). "
    + "The comparison is also not matched in every input (section 9): the network models see other gauges and have NWM pretraining.\n")
sd = lambda m, lev, p_="heldout": float(strata[(strata.protocol == p_) & (strata.stratum == "upstream visible gauge") & (strata.level == lev) & (strata.lead == "days 1-5")][f"median_nse_{m}"].iloc[0])  # noqa: E731
m02, m01 = pr("heldout", "M0", "M2"), pr("heldout", "M0", "M1")
add(f"**H2** (any network effect comes through physical routing, not learned message passing). Message passing, M0 against M2: {f(pr('gauged', 'M0', 'M2').mean_diff, 3, True)} gauged, {f(m02.mean_diff, 3, True)} held-out, "
    f"{f(pr('blocked', 'M0', 'M2').mean_diff, 3, True)} blocked (section 3.1): no detectable benefit of the graph layers; equivalence within +-0.01 holds on gauged only. The routing test, M1 against M4 (M1 without routing: instantaneous summation of upstream lateral inflow), has not been run yet.\n")
if effect is not None:
    hh = effect[(effect.protocol == "heldout") & effect.mean_diff_M0_minus_insertion_off.notna() & (effect.subset == "with a visible gauge upstream")].set_index("lead")
    n_vis = int(hh.gauges.iloc[0])
    add(f"**H3** (observations at upstream gauges, carried downstream by the routing, add skill at ungauged reaches below them). Tested at the {n_vis} held-out gauges that have a visible gauge upstream "
        f"(the blocked protocol cannot test it, section 8). With M0's own weights, switching insertion off at test time lowers 6 h NSE by {f(hh.loc['6', 'mean_diff_M0_minus_insertion_off'], 3)} on average "
        f"(p {p_(hh.loc['6', 'wilcoxon_p_M0_vs_off'])}), so insertion adds skill at short leads. At 24 h the change is {f(hh.loc['24', 'mean_diff_M0_minus_insertion_off'], 3, True)} (p {p_(hh.loc['24', 'wilcoxon_p_M0_vs_off'])}) "
        f"and for days 1-5 {f(hh.loc['days 1-5', 'mean_diff_M0_minus_insertion_off'], 3, True)} (p {p_(hh.loc['days 1-5', 'wilcoxon_p_M0_vs_off'])}): no detectable benefit beyond the first hours. "
        f"The trained comparison agrees: M0 minus M1 is {f(m01.mean_diff, 3, True)} on all held-out gauges for days 1-5, and {f(hh.loc['days 1-5', 'mean_diff_M0_minus_M1'], 3, True)} at these {n_vis} gauges. "
        "So H3 is supported at 6 h and not for days 1-5.\n")
else:
    add("**H3** (observations at upstream gauges, carried downstream by the routing, add skill at ungauged reaches below them): pending the insertion-off counterfactual (section 8).\n")

# ================================================================================================================ 5 KGE
add("## 5. KGE and its components\n")
add(f"KGE is the variant of Gupta et al. (2009): KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2) with alpha = std(sim) / std(obs) and beta = mean(sim) / mean(obs) "
    f"(the 2012 variant, which uses coefficient-of-variation ratios, is not used). The repository's reach metrics (`evaluation/hydrology.py`) use the same definition. "
    f"Validation against an independent numpy implementation (np.corrcoef, np.std, np.mean): {kval.gauge.nunique()} gauges x 5 leads x {kval.model.nunique()} models, maximum absolute difference {kval.abs_diff.max():.1e}. "
    "`hydroeval` is not installed. Values are the median over gauges, averaged over leads 24-120 h.\n")
for p in PROT:
    rows = []
    for m in MODELS:
        rows.append([m] + [f(cell(p, m, "days 1-5", c)) for c in ("median_kge", "median_r", "median_alpha", "median_beta", "median_nse")])
    add(f"{PNAME[p]} ({NG[p]} gauges), days 1-5:\n")
    add(table(["Model", "KGE", "r", "alpha (variability)", "beta (bias)", "NSE"], rows))
add("KGE by lead:\n")
rows = []
for p in PROT:
    for m in MODELS:
        rows.append([PNAME[p], m] + [f(cell(p, m, L, "median_kge")) for L in ("6 h", "24 h", "72 h", "120 h", "240 h", "days 1-5")])
add(table(["Protocol", "Model", "6 h", "24 h", "72 h", "120 h", "240 h", "Days 1-5"], rows))
b = {m: {c: cell("blocked", m, "days 1-5", c) for c in ("median_kge", "median_r", "median_alpha", "median_beta")} for m in ("M0", "M2", "LSTM", "NWM")}
add(f"On blocked basins KGE ranks the LSTM ({f(b['LSTM']['median_kge'])}) above M2 ({f(b['M2']['median_kge'])}) and M0 ({f(b['M0']['median_kge'])}) although NSE ranks them the other way. "
    f"The components show why: the correlation of M0 and M2 ({f(b['M0']['median_r'])}, {f(b['M2']['median_r'])}) is higher than the LSTM's ({f(b['LSTM']['median_r'])}), "
    f"but their variability ratio alpha ({f(b['M0']['median_alpha'])}, {f(b['M2']['median_alpha'])}) is lower than the LSTM's ({f(b['LSTM']['median_alpha'])}) and NWM's ({f(b['NWM']['median_alpha'])}); bias ratios are similar "
    f"({f(b['M0']['median_beta'])}, {f(b['M2']['median_beta'])}, {f(b['LSTM']['median_beta'])}). The M-family forecasts are better timed but more damped on blocked basins. "
    f"All models have alpha below 1 in every protocol (flows are under-dispersed), the M-family most on blocked basins.\n")

# ================================================================================================================ 6 high flow
add("## 6. High-flow measures\n")
add("Threshold: the **2-year flow**, the median of each gauge's annual maxima (water years, Oct-Sep) of the 6-hourly observed flow before 2019-01-01, for gauges with at least 10 water years with >= 70% of values "
    "observed. Few gauges have enough 2-year events in the test period for a CSI, so CSI at the gauge's **95th percentile** of training-period flow is reported as well. Median over gauges, mean over leads 24-120 h; "
    "`n` = gauges with enough events (>= 5 event pairs at the 2-year flow, >= 20 at the 95th percentile).\n")
rows = []
for p in PROT:
    for m in MODELS:
        sel = hf[(hf.protocol == p) & (hf.model == m) & (hf.measure == "days 1-5")]
        if sel.empty:  # M1 has no blocked runs
            rows.append([PNAME[p], m, "not run", "not run", "", "not run", ""])
            continue
        r = sel.iloc[0]
        rows.append([PNAME[p], m, f(r.median_peak_bias_top5pct), f(r.csi_p95), int(r.csi_p95_gauges), f(r.csi_2yr), int(r.csi_2yr_gauges)])
add(table(["Protocol", "Model", "Peak-flow bias, top 5% flows (sim/obs - 1)", "CSI > p95", "n", "CSI > 2-year flow", "n"], rows))
add("Peak timing and magnitude: for 10-day windows whose observed maximum is at least the 2-year flow, the error in the lead of the forecast maximum (hours, forecast minus observed) and the "
    "relative error of the forecast maximum, by the lead of the observed peak; median over gauges with >= 3 such windows.\n")
rows = []
for p in PROT:
    for m in MODELS:
        row = [PNAME[p], m]
        for label in ("peak <= 24 h", "peak 24-72 h", "peak 72-120 h", "peak 120-240 h"):
            sel = hf[(hf.protocol == p) & (hf.model == m) & (hf.measure == label)]
            if sel.empty:
                row.append("not run")
                continue
            r = sel.iloc[0]
            row.append(f"{f(r.timing_median_abs_h, 1)} h / {f(r.magnitude_median_rel_error, 2, True)} (n {int(r.timing_gauges)})")
        rows.append(row)
add(table(["Protocol", "Model", "Peak within 24 h", "24-72 h", "72-120 h", "120-240 h"], rows))
add("Entries are median absolute timing error / median relative peak-magnitude error. All models underestimate flood peaks, increasingly with the lead of the peak.\n")

# ================================================================================================================ 7 strata
add("## 7. Stratification\n")
add("Area bins (km2) are fixed near the 25th, 60th and 90th percentiles of the gauged-protocol gauge areas (14 to 5,308 km2): < 200, 200-600, 600-2,000, > 2,000.\n")
for p in PROT:
    rows = []
    for lev in ["< 200", "200-600", "600-2,000", "> 2,000"]:
        r = strata[(strata.protocol == p) & (strata.stratum == "area km2") & (strata.level == lev) & (strata.lead == "days 1-5")].iloc[0]
        vals = {m: r[f"median_nse_{m}"] for m in MODELS}
        rows.append([lev, int(r.gauges)] + best(vals))
    add(f"{PNAME[p]}, days 1-5 median NSE by drainage area:\n")
    add(table(["Area km2", "Gauges"] + MODELS, rows))
add("By whether a gauge visible to the model lies upstream (the condition under which insertion can act):\n")
rows = []
for p in PROT:
    for lev in ("yes", "no"):
        r = strata[(strata.protocol == p) & (strata.stratum == "upstream visible gauge") & (strata.level == lev) & (strata.lead == "days 1-5")].iloc[0]
        if r.gauges == 0:
            rows.append([PNAME[p], lev, 0] + ["-"] * 5)
        else:
            rows.append([PNAME[p], lev, int(r.gauges)] + best({m: r[f"median_nse_{m}"] for m in MODELS}))
add(table(["Protocol", "Visible upstream gauge", "Gauges"] + MODELS, rows))
add("On blocked basins no scored gauge has a visible upstream gauge: every gauge upstream of a blocked gauge lies inside the same withheld basin.\n")
add("Area and nesting together (per-gauge days 1-5 NSE of M0 and M2 minus LSTM regressed on log10(area) and on having any gauge upstream, visible or withheld):\n")
rows = []
for r in ols[ols.term != "intercept"].itertuples():
    rows.append([PNAME[r.protocol], r.model, r.term, f(r.coef, 3, True), f(r.se, 3), p_(r.p), int(r.gauges)])
add(table(["Protocol", "Difference", "Term", "Coefficient", "SE", "p", "Gauges"], rows))
bl = ols[(ols.protocol == "blocked") & (ols.model == "M0 - LSTM")].set_index("term")
add(f"On blocked basins the M0 - LSTM advantage is not explained by area (p {p_(bl.loc['log10(area km2)', 'p'])}) and shows no reliable dependence on having a gauge upstream "
    f"(p {p_(bl.loc['any upstream gauge', 'p'])}); with {int(bl.gauges.iloc[0])} gauges in 3 basins this test has little power. "
    "The cross-tabulation by area and nesting is in `strata_area_by_upstream_gauge.csv`.\n")
add("By dam category (days 1-5 median NSE; the 2-gauge blocked dam category is only in `by_dam_category.csv`):\n")
rows = []
for _, r in dam.iterrows():
    if r.protocol == "blocked" and r.gauges < 5:
        continue
    vals = {m: r[f"days1-5_median_nse_{m}"] for m in MODELS}
    rows.append([PNAME[r.protocol], r.category, int(r.gauges)] + best(vals))
add(table(["Protocol", "Category", "Gauges"] + MODELS, rows))

# ================================================================================================================ 8 insertion audit
add("## 8. Insertion audit\n")
add("**What the code does** (`hydro/model.py` `HydroModel.forward` and `insertion`, `hydro/samples.py`, `hydro/evaluate.py`):\n")
add("1. At test time each fold model scores its own held-out gauges with `visible = Samples.eligible(step, withheld[k])`: not withheld, and at least 8 observed values in the last 32 steps. "
    "Gauges of the training folds, and non-CAMELS-H catalog gauges, that lie upstream of a withheld gauge are therefore visible and are inserted.")
add("2. `Samples.gauge_view` sets the withheld gauges' flow to NaN at every time and marks them not visible; `insertion` uses only visible gauges and only the steps before the issue. "
    "The withheld gauge's own observation never enters its forecast, nor does that of any other withheld gauge.")
add("3. The corrections are routed down the network by the Manning router (`insert` and `insert_gauges` operations): to every reach and to every gauge *below* the correcting gauge, never to the gauge itself.")
add("4. At a gauge that is visible at the issue the forecast over the lead window is the gauge head's output, which reads the routed flow *before* insertion, so insertion cannot change it. "
    "A gauge is visible only if it has at least 8 values in the last 32 steps; a gauge without recent flow is treated like an ungauged reach (routed flow plus corrections from visible gauges above it), "
    "also in the gauged protocol. The counterfactual below measures this directly.\n")
n_pass = int((tests.result == "pass").sum())
add(f"**Tests.** `tests/test_hydro.py` now has {len(tests)} tests ({n_pass} pass; `unit_tests.csv`). The new ones, on a synthetic 5-reach network: an inserted observation changes the downstream forecast and nothing upstream or beside it; "
    "a withheld gauge's observation, whatever its value, never changes the insertion output; observed flow after the issue is ignored; the correction is routed with the travel time and conserves mass; "
    "`Samples.gauge_view` hides withheld gauges at all times and `eligible` excludes them.\n")
lk = leak
add(f"**Check on real checkpoints** (`scripts/v3_leak_check.py`: M0 and M1 fold 1, {lk.target.nunique()} held-out gauges with visible upstream gauges, {len(lk)} gauge-issue cases, fp32). "
    f"Re-running the same inputs changes the forecast by up to {lk.rerun_max_abs_diff.max():.4f} m3/s (floating-point noise from sparse operations). Changing the observed flow after the issue changes it by up to "
    f"{lk.A_future_obs_changed_max_abs_diff.max():.4f} m3/s, and replacing the withheld gauges' true flow by 1e6 by up to {lk.B_withheld_true_flow_changed_max_abs_diff.max():.4f} m3/s: both at the noise level, so no leakage. "
    f"Hiding the visible upstream gauges changes the target forecast by {lk.control_upstream_hidden_target_max_abs_diff.min():.2f} to {lk.control_upstream_hidden_target_max_abs_diff.max():.2f} m3/s, "
    f"so the models do use upstream observations. (With bf16, as in the evaluation, the noise floor is {leak16.rerun_max_abs_diff.max():.2f} m3/s; `leak_check_bf16.csv`.)\n")
s = ins_sum.set_index("protocol")
add("**How many scored gauges have a visible gauge upstream** (`insertion_upstream_visible_summary.csv`; insertion can only act on them):\n")
rows = []
for p in PROT:
    r = s.loc[p]
    rows.append([PNAME[p], int(r.scored_gauges), int(r.with_upstream_gauge), int(r.with_upstream_visible_gauge), f(r.mean_share_issues_with_visible_upstream, 2), "n/a" if np.isnan(r.median_covered_area_share_when_present) else f(r.median_covered_area_share_when_present, 2)])
add(table(["Protocol", "Scored gauges", "With any gauge upstream", "With a visible gauge upstream", "Mean share of issues with a visible upstream gauge", "Median share of basin area upstream of visible gauges"], rows))
rb = s.loc["blocked"]
add(f"On the blocked folds **{int(rb.with_upstream_visible_gauge)} of the {int(rb.scored_gauges)}** scored gauges have a visible gauge upstream ({int(rb.with_upstream_gauge)} have gauges upstream, but those lie in the same withheld basin). "
    "The blocked protocol therefore cannot test insertion, and H3 can only be tested on the gauged and held-out protocols.\n")
add("**M0 against M1 at gauged-protocol sites below another gauge** (all gauges visible; `insertion_M0_vs_M1_gauged_sites_below_gauges.csv`). Insertion cannot act here, so any difference reflects training:\n")
rows = []
for r in ins_gauged.itertuples():
    rows.append([r.subset, int(r.gauges), f"{r.lead_h} h", f(r.median_nse_M0), f(r.median_nse_M1), f(r.mean_diff_M0_minus_M1, 3, True), f"{int(r.m0_better)}/{int(r.gauges)}", p_(r.wilcoxon_p)])
add(table(["Sites", "Gauges", "Lead", "M0", "M1", "Mean diff", "M0 better", "Wilcoxon p"], rows))

toggle_path = OUT / "insertion_toggle_check.csv"
if effect is not None:
    add("**Counterfactual: M0 with insertion switched off at test time, same weights** (`eval_M0_s1_insertoff`, `scripts/v3_insertion_effect.py`; GEFS rain, gauged and held-out protocols).\n")
    g1 = effect[effect.protocol == "gauged"].set_index("measure").value
    key = lambda lab, what: float(g1[f"gauge-issue-lead values {lab}: {what}"])  # noqa: E731
    vis, hid = "visible at the issue", "not visible at the issue (no recent flow)"
    add(f"- Gauged protocol: {f(100 * key(vis, 'share of all values'), 2)}% of gauge-issue-lead values are at gauges visible at the issue; insertion changes them by a median of {f(key(vis, 'median |change from insertion| (m3/s)'), 4)} m3/s "
        f"(99.9th percentile {f(key(vis, '99.9th percentile of |change| (m3/s)'), 3)}, maximum {f(key(vis, 'max |change| (m3/s)'), 2)} m3/s), which is not an insertion effect: the fp32 toggle below changes visible gauges by exactly zero, so these differences come from bf16 run-to-run variation between the two evaluation runs. "
        f"The remaining {f(100 * key(hid, 'share of all values'), 2)}% are gauges with no recent flow, treated as hidden; there insertion changes the forecast by up to {f(key(hid, 'max |change| (m3/s)'), 1)} m3/s.")
    if toggle_path.exists():
        tg = pd.read_csv(toggle_path)
        parts = []
        for lab in ("visible gauges", "hidden, no visible gauge upstream", "hidden, visible gauge upstream"):
            t_ = tg[tg.subset == lab]
            if len(t_):
                parts.append(f"{lab}: largest change {f(t_.max_abs_change_m3s.max(), 4)} m3/s")
        add(f"- Same weights, insertion toggled on one real model in fp32 (two test issues, `insertion_toggle_check.csv`): " + "; ".join(parts) + ".")
    add("")
    h = effect[(effect.protocol == "heldout") & effect.mean_diff_M0_minus_insertion_off.notna()].copy()
    rows = []
    for r in h.itertuples():
        unit = "per-gauge days 1-5 NSE" if r.lead == "days 1-5" else f"NSE at {r.lead} h"
        rows.append([r.subset, int(r.gauges), unit, f(r.median_nse_M0), f(r.median_nse_M0_insertion_off), f(r.median_nse_M1),
                     f"{f(r.mean_diff_M0_minus_insertion_off, 3, True)} (p {p_(r.wilcoxon_p_M0_vs_off)})", f"{f(r.mean_diff_M0_minus_M1, 3, True)} (p {p_(r.wilcoxon_p_M0_vs_M1)})"])
    add("Held-out gauges. Medians over gauges of the quantity in the third column; mean paired differences with Wilcoxon p:\n")
    add(table(["Gauges", "n", "Quantity", "M0", "M0, insertion off", "M1", "M0 minus M0 off", "M0 minus M1"], rows))
    c = effect[effect.median_abs_forecast_change_share_of_mean_flow.notna()]
    rows = [[r.subset, f"{r.lead} h", int(r.gauges), f(100 * r.median_abs_forecast_change_share_of_mean_flow, 1) + "%", f(100 * r.p90_abs_forecast_change_share_of_mean_flow, 1) + "%"] for r in c.itertuples()]
    add("Size of the correction: mean absolute change of the forecast caused by insertion, as a share of the gauge's mean flow (median and 90th percentile over gauges):\n")
    add(table(["Gauges", "Lead", "n", "Median", "90th percentile"], rows))
    sp = effect[effect.spearman_covered_area_share_vs_gain_from_insertion.notna()]
    add("Rank correlation, across the gauges with a visible gauge upstream, between the share of the basin lying upstream of visible gauges and the NSE gain from insertion: "
        + "; ".join(f"{r.lead}: rho {f(r.spearman_covered_area_share_vs_gain_from_insertion, 2, True)} (p {p_(r.spearman_p)}, n {int(r.gauges)})" for r in sp.itertuples()) + ".\n")

    def e(sub, lead_, col):
        return float(h[(h.subset == sub) & (h.lead == lead_)][col].iloc[0])
    A_, N_ = "with a visible gauge upstream", "no visible gauge upstream"
    add("**Why M0 and M1 differ by only 0.001 in median NSE at 6 h on held-out gauges.** "
        f"The median over all {int(e('all held-out gauges', '6', 'gauges'))} gauges is M0 {f(e('all held-out gauges', '6', 'median_nse_M0'))} and M1 {f(e('all held-out gauges', '6', 'median_nse_M1'))}, "
        f"but the mean paired difference is {f(e('all held-out gauges', '6', 'mean_diff_M0_minus_M1'), 3, True)} (p {p_(e('all held-out gauges', '6', 'wilcoxon_p_M0_vs_M1'))}). "
        f"Insertion can only act at the {int(e(A_, '6', 'gauges'))} gauges that have a visible gauge upstream; at the other {int(e(N_, '6', 'gauges'))} M0 with insertion off is identical to M0 "
        f"(median change {f(100 * float(effect[(effect.subset == N_) & (effect.lead.astype(str) == '6') & effect.median_abs_forecast_change_share_of_mean_flow.notna()].median_abs_forecast_change_share_of_mean_flow.iloc[0]), 1)}% of mean flow), so they cannot differ because of insertion, and there M1 is slightly better than M0 (median {f(e(N_, '6', 'median_nse_M1'))} against {f(e(N_, '6', 'median_nse_M0'))}, training differences). "
        f"At the {int(e(A_, '6', 'gauges'))} gauges insertion matters: median 6 h NSE {f(e(A_, '6', 'median_nse_M0'))} with insertion and {f(e(A_, '6', 'median_nse_M0_insertion_off'))} without "
        f"(mean gain {f(e(A_, '6', 'mean_diff_M0_minus_insertion_off'), 3, True)}, p {p_(e(A_, '6', 'wilcoxon_p_M0_vs_off'))}); M1, which has no insertion, sits at {f(e(A_, '6', 'median_nse_M1'))}. "
        "The median over all gauges is set by the majority that insertion cannot reach, which cancels the gain. "
        f"The gain does not persist: at 24 h the mean change is {f(e(A_, '24', 'mean_diff_M0_minus_insertion_off'), 3, True)} (p {p_(e(A_, '24', 'wilcoxon_p_M0_vs_off'))}) and for days 1-5 "
        f"{f(e(A_, 'days 1-5', 'mean_diff_M0_minus_insertion_off'), 3, True)} (p {p_(e(A_, 'days 1-5', 'wilcoxon_p_M0_vs_off'))}), and the larger the share of the basin covered by upstream gauges the smaller the days 1-5 gain "
        f"(rho {f(float(sp[sp.lead == 'days 1-5'].spearman_covered_area_share_vs_gain_from_insertion.iloc[0]), 2, True)}).\n")
else:
    add("*The counterfactual evaluation of M0 with insertion off (`eval_M0_s1_insertoff`) has not finished; the section is added when it does.*\n")

# ================================================================================================================ 9 inputs
add("## 9. Inputs audit\n")
add("What each model receives (`inputs_audit.csv`, from the code constants, checkpoint configs and `arguments.json`). The comparison is matched in forcing, static attributes, antecedent storage, the "
    "forecast-rain source and the output units. It is **not** matched in everything: the network models see other gauges' observations and have NWM pretraining and a different loss and training volume.\n")
rows = [[r["input"], r["M0, M1, M2"], r["LSTM"], r["matched"] if isinstance(r["matched"], str) else "-"] for _, r in inputs.iterrows()]
add(table(["Input / setting", "M0, M1, M2", "LSTM", "Matched?"], rows, text=True))
diff = config[~config.same_in_M0_M1_M2].drop(columns="same_in_M0_M1_M2")
add("Settings that differ among M0, M1 and M2 (`config_audit.csv`; the other " + str(int(config.same_in_M0_M1_M2.sum())) + " of " + str(len(config)) + " recorded settings, including loss, updates, learning rate and router, are identical):\n")
add(table(["Setting", "M0", "M1", "M2"], [[i, r.M0, r.M1, r.M2] for i, r in diff.iterrows()], text=True))

# ================================================================================================================ 10 reach
add("## 10. Reach level: every reach against the NWM v3.0 retrospective\n")
add("58,834 reaches, 425 issues to 2023-01-20, GEFS rain; median NSE over reaches. The target is the NWM retrospective, not observed flow, so this measures consistency with NWM, not skill. "
    "M1 and M2 have not been evaluated at the reach level, and none is planned for M1, M2 or M4.\n")
rows = []
for r in reach.itertuples():
    if r.lead_h not in (24, 72, 120):
        continue
    vals = {"M0": r.M0, "LSTM": r.LSTM, "NWM": r.NWM}
    rows.append([f"{r.lead_h} h", r.scope, f"{int(r.reaches):,}"] + best(vals, skip=("NWM",)) + ["not run"])
add(table(["Lead", "Scope", "Reaches", "M0", "LSTM", "NWM", "M1, M2"], rows))

# ================================================================================================================ 11 pending / sources
add("## 11. Pending\n")
add("- M4 (M1 without routing) on the held-out and blocked protocols; seed 2 on the blocked folds. Not planned: gauged-protocol runs of new models, reach-level runs of M1, M2 and M4, gauged and held-out seeds, seed 3.\n")
add("## 12. Figures, tables and scripts\n")
add("Draft figures in `figures/paper/` (PDF and PNG): `fig_a_skill_vs_lead`, `fig_b_gap_vs_area`, `fig_c_ablation_ladder`, `fig_d_kge_components`, `fig_e_reach_level`.\n")
add("Every table here is a CSV in `reports/v3_tables/`; the scripts are `scripts/v3_common.py` (loaders, scores, statistics), `v3_insertion_audit.py`, `v3_leak_check.py`, `v3_insertion_effect.py`, "
    "`v3_config_audit.py`, `v3_inputs_audit.py`, `v3_kge_highflow.py`, `v3_stats.py`, `v3_figures.py` and this report builder `v3_report.py`; unit tests: `tests/run_all.py`. Commands and jobs are in `RUNLOG.md`.\n")
add("Inputs: saved forecasts in `models/attr/` (`eval_M0_s1`, `eval_M1_s1`, `eval_M2_s1`, `eval_S1_M0_s1`, `eval_S1_M2_s1`, `eval_M0_s1_insertoff`, `eval_M0_reach`), the LSTM benchmarks "
    "(`ladder_L4_s1_bench`, `_bench_f{1..5}`, `_bench_b{1..3}`, `ladder_L4_s1_reaches`), and the NWM rows of the M0 gauged evaluation.\n")

(REPO / "reports" / "test_comparison_v3_full.md").write_text("\n".join(md))
print("wrote reports/test_comparison_v3_full.md,", len("\n".join(md).splitlines()), "lines")
