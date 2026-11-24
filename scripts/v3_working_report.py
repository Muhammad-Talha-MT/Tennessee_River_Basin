"""Builds the short working report reports/test_comparison_v3.md from reports/v3_tables/*.csv: only the comparisons that answer the research questions
(RQ1 reach-scale routed model vs the LSTM; RQ2 routing M1 vs M4 and message passing M0 vs M2; RQ3 insertion M0 vs M1), on held-out and blocked basins.
A pending result is shown as "-". The full report is reports/test_comparison_v3_full.md. Run: python scripts/v3_working_report.py"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from v3_common import OUT, REPO

R = lambda n: pd.read_csv(OUT / n)  # noqa: E731
comp, paired, bybasin, lead, effect = R("kge_components.csv"), R("paired_tests.csv"), R("blocked_by_basin.csv"), R("lead_curves.csv"), R("insertion_effect_summary.csv")
PROT = ["gauged", "heldout", "blocked"]
PN = {"gauged": "Gauged", "heldout": "Held-out", "blocked": "Blocked"}
NG = {p: int(comp[(comp.protocol == p) & (comp.model == "M0")].gauges.iloc[0]) for p in PROT}


def f(x, d=3, sign=False):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def pv(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else ("< 0.001" if x < 0.001 else f"{x:.3f}")


def cell(p, m, L, c):
    r = comp[(comp.protocol == p) & (comp.model == m) & (comp.lead == L)]
    return float(r[c].iloc[0]) if len(r) else np.nan


def pr(p, a, b):
    r = paired[(paired.protocol == p) & (paired.model_a == a) & (paired.model_b == b)]
    return r.iloc[0] if len(r) and not pd.isna(r.iloc[0].get("mean_diff")) else None


def diff_cell(p, a, b, equiv=False):
    r = pr(p, a, b)
    if r is None:
        return "-"
    s = f"{f(r.mean_diff, 3, True)} [{f(r.boot90_lo, 3, True)}, {f(r.boot90_hi, 3, True)}]; {int(r.a_better)}/{int(r.gauges)}; Holm p {pv(r.wilcoxon_p_holm)}"
    if equiv and not pd.isna(r.get("equivalent_0p01")):
        s += "; equivalent within +-0.01: " + ("yes" if r.equivalent_0p01 else "no")
    return s


def table(h, rows):
    return "\n".join(["| " + " | ".join(h) + " |", "|" + "|".join(["---"] + ["---:"] * (len(h) - 1)) + "|"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]) + "\n"


md = []
add = md.append
add("# Working report: results against the research questions\n")
add("Test issues 2020-09-23 to 2025-12-22 (1,917 issues), GEFSv12 rain, leads 6-240 h. Metric: median NSE over gauges; *days 1-5* = mean over the leads 24-120 h. One seed. "
    f"Protocols: gauged ({NG['gauged']} gauges), held-out ({NG['heldout']}, 5 random folds), blocked ({NG['blocked']}, 3 whole basins). **A dash (-) marks a result that is still pending.**\n")
add("| Model | Definition | Status |\n|---|---|---|\n| M0 | Manning routing + gauge insertion + 3 graph layers | done |\n| M1 | M0 without insertion | done |\n"
    "| M2 | M0 without graph layers | done |\n| M4 | M1 without routing (instantaneous summation of upstream lateral inflow) | stage A done; held-out and blocked runs in progress (-) |\n"
    "| LSTM | area-scaled LSTM, the only baseline | done |\n| NWM | v3.0 forecast, reference only: it nudges toward every USGS gauge, including withheld ones | done |\n")

add("## 1. Headline: days 1-5 median NSE\n")
rows = []
for p in PROT:
    rows.append([f"{PN[p]} ({NG[p]})"] + [f(cell(p, m, "days 1-5", "median_nse")) for m in ("M0", "M1", "M2", "M4", "LSTM", "NWM")])
add(table(["Protocol", "M0", "M1", "M2", "M4", "LSTM", "NWM"], rows))
add("M4 has no gauged runs by design.\n")

add("## 2. RQ1: does a reach-scale routed model transfer better than the LSTM?\n")
add("Per-gauge days 1-5 NSE, model minus LSTM: mean difference [90% gauge bootstrap]; gauges where the model is better; Holm-adjusted Wilcoxon p (six tests per protocol).\n")
rows = [[m + " - LSTM"] + [diff_cell(p, m, "LSTM") for p in PROT] for m in ("M0", "M2", "M1", "M4")]
add(table(["Comparison", "Gauged", "Held-out", "Blocked"], rows))
for m in ("M0", "M2", "M1", "M4"):
    for p_ in PROT:
        r_ = pr(p_, m, "LSTM")
        if r_ is not None and np.sign(r_.mean_diff) != np.sign(r_.median_diff):
            add(f"Note: {m} - LSTM on {PN[p_].lower()} has a mean of {f(r_.mean_diff, 3, True)} but a median of {f(r_.median_diff, 3, True)} and wins on {int(r_.a_better)} of {int(r_.gauges)} gauges: a few large losses pull the mean below zero.\n")
trend = {p: ("shrinks" if lead[(lead.protocol == p) & (lead.lead_h >= 144)].diff_median_M0_minus_LSTM.mean() < lead[(lead.protocol == p) & (lead.lead_h <= 48)].diff_median_M0_minus_LSTM.mean() else "grows") for p in PROT}
head = {p: {m: cell(p, m, "days 1-5", "median_nse") for m in ("M0", "LSTM")} for p in PROT}
add("Gap in median NSE (M0 minus LSTM): " + ", ".join(f"{PN[p].lower()} {f(head[p]['M0'] - head[p]['LSTM'], 3, True)}" for p in PROT) + ". "
    "With lead the gap " + ", ".join(f"{trend[p]} on {PN[p].lower()}" for p in PROT) + " (leads up to 48 h against 144 h and beyond).\n")
rows = [[int(r.blocked_basin), int(r.gauges), f(r.median_nse_d15_M0), f(r.median_nse_d15_M1), f(r.median_nse_d15_M2), f(r.median_nse_d15_LSTM), f(r.mean_diff_M0_minus_LSTM, 3, True), f"{int(r.M0_better_gauges)}/{int(r.gauges)}"] for r in bybasin.itertuples()]
add("Blocked basins one by one (the three basins are the only independent units, so the gauge-level tests above are descriptive):\n")
add(table(["Basin", "Gauges", "M0", "M1", "M2", "LSTM", "M0 - LSTM (mean)", "M0 better"], rows))
add("Caveat for the Methods: the comparison is matched in forcing, attributes, antecedent storage, rain source and output units, but the network models also see other gauges' observations and have NWM pretraining; the LSTM has neither.\n")

add("## 3. RQ2: is any network effect carried by physical routing rather than learned message passing?\n")
add("Component effect = the full model minus the model without the component, days 1-5 NSE, mean paired difference [90% bootstrap]; gauges better; Holm p; equivalence within +-0.01 (paired TOST, Holm-adjusted).\n")
add(table(["Component (comparison)", "Held-out", "Blocked"], [
    ["Routing (M1 - M4)", diff_cell("heldout", "M1", "M4", True), diff_cell("blocked", "M1", "M4", True)],
    ["Message passing (M0 - M2)", diff_cell("heldout", "M0", "M2", True), diff_cell("blocked", "M0", "M2", True)]]))

add("## 4. RQ3: do upstream gauge observations, carried downstream by the routing, add skill?\n")
add(table(["Component (comparison)", "Held-out", "Blocked"], [["Insertion (M0 - M1)", diff_cell("heldout", "M0", "M1", True), (diff_cell("blocked", "M0", "M1", True) + " (insertion cannot act here: no scored blocked gauge has a visible gauge upstream)")]]))
rb = pr("blocked", "M0", "M1")
if rb is not None:
    add(f"On blocked basins insertion cannot act, so M0 minus M1 there ({f(rb.mean_diff, 3, True)}, 90% interval [{f(rb.boot90_lo, 3, True)}, {f(rb.boot90_hi, 3, True)}]) is the difference between two trainings, "
        "not an insertion effect; it gives a scale for run-to-run differences with a single seed.\n")
h = effect[(effect.protocol == "heldout") & effect.mean_diff_M0_minus_insertion_off.notna() & (effect.subset == "with a visible gauge upstream")].set_index("lead")
n = int(h.gauges.iloc[0])
rows = [[("per-gauge days 1-5" if L == "days 1-5" else f"{L} h"), f(h.loc[L, "median_nse_M0"]), f(h.loc[L, "median_nse_M0_insertion_off"]), f(h.loc[L, "median_nse_M1"]),
         f"{f(h.loc[L, 'mean_diff_M0_minus_insertion_off'], 3, True)} (p {pv(h.loc[L, 'wilcoxon_p_M0_vs_off'])})"] for L in ("6", "24", "days 1-5")]
add(f"Counterfactual on the {n} held-out gauges that have a visible gauge upstream (the only gauges insertion can change): M0 with its insertion switched off at test time, same weights. Median NSE; gain = M0 minus M0 without insertion.\n")
add(table(["Quantity", "M0", "M0, insertion off", "M1", "Gain from insertion (mean)"], rows))
allh = effect[(effect.protocol == "heldout") & effect.mean_diff_M0_minus_insertion_off.notna() & (effect.subset == "all held-out gauges")].set_index("lead")
add(f"Insertion helps at 6 h ({f(h.loc['6', 'mean_diff_M0_minus_insertion_off'], 3, True)}, p {pv(h.loc['6', 'wilcoxon_p_M0_vs_off'])}) and not beyond. Over all {int(allh.gauges.iloc[0])} held-out gauges the 6 h median NSE is "
    f"M0 {f(allh.loc['6', 'median_nse_M0'])} and M1 {f(allh.loc['6', 'median_nse_M1'])}, but the mean paired difference is {f(allh.loc['6', 'mean_diff_M0_minus_M1'], 3, True)}: the median is set by the gauges insertion cannot reach.\n")

add("## 5. KGE and its components (days 1-5, median over gauges)\n")
for p in ("heldout", "blocked"):
    rows = [[m] + [f(cell(p, m, "days 1-5", c)) for c in ("median_kge", "median_r", "median_alpha", "median_beta")] for m in ("M0", "M1", "M2", "M4", "LSTM", "NWM")]
    add(f"{PN[p]} ({NG[p]} gauges):\n")
    add(table(["Model", "KGE", "r", "alpha (variability)", "beta (bias)"], rows))
add("KGE is the 2009 variant (alpha = std ratio, beta = mean ratio), validated against an independent implementation. On blocked basins the LSTM leads on KGE although M0, M1 and M2 lead on NSE: the M-models have the higher correlation but are more damped (lower alpha).\n")

add("## 6. Pending\n")
add("| Item | Status |\n|---|---|\n| M1 on the 3 blocked folds | done |\n| M4: stage A, 5 held-out folds, 3 blocked folds | in progress (-) |\n| Seed 2 on the blocked folds (M0, M1, M2, M4, LSTM) | after the job table is confirmed (-) |\n"
    "| Seeds for held-out and gauged, seed 3, reach level for M1, M2, M4 | not planned |\n")
add("The full report (inputs audit, leakage checks, high-flow measures, stratification, reach level) is `test_comparison_v3_full.md`; every table is a CSV in `reports/v3_tables/`.\n")
(REPO / "reports" / "test_comparison_v3.md").write_text("\n".join(md))
print("wrote reports/test_comparison_v3.md,", len("\n".join(md).splitlines()), "lines")
