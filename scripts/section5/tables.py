"""Table 3 (median NSE and KGE, days 1-5, by protocol and model, with n) and Table S6 (every paired contrast: n, difference, 5-95% interval, Holm p),
LaTeX booktabs in the style of the paper's tables, from reports/section5/s5_2_headline.csv and s5_3_contrasts.csv.
Run: python scripts/section5/tables.py   Writes reports/section5/table3_skill.tex and reports/section5/tableS6_contrasts.tex.
"""
import numpy as np
from style import CSV, DISPLAY, PROTOCOLS, PNAME, pretty, read

head = read("s5_2_headline.csv")
con = read("s5_3_contrasts.csv")
MODELS = ["M0", "M1", "M2", "M4", "LSTM", "NWM"]
L = [r"\begin{table}", r"\caption{Median NSE and KGE, Days 1--5, by Protocol and Model}", r"\label{tab:skill}", r"\footnotesize",
     r"\setlength{\parindent}{0pt}", r"\renewcommand{\arraystretch}{1.2}", r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
     r"Protocol & Model & $n$ & NSE & KGE \\", r"\midrule"]
for k, p in enumerate(PROTOCOLS):
    h = head[head.protocol == PNAME[p]].set_index("model")
    best_nse, best_kge = h.nse_days1_5.max(), h.kge_days1_5.max()
    for i, m in enumerate(MODELS):
        label = PNAME[p] if i == 0 else ""
        if m not in h.index:
            why = "not trained" if p == "gauged" else "not evaluated"
            L.append(f"{label} & {DISPLAY[m]} & --- & \\multicolumn{{2}}{{c}}{{{why}}} \\\\")
            continue
        r = h.loc[m]
        nse = f"{r.nse_days1_5:.3f}"
        kge = f"{r.kge_days1_5:.3f}"
        if nse == f"{best_nse:.3f}":
            nse = r"\textbf{" + nse + "}"
        if kge == f"{best_kge:.3f}":
            kge = r"\textbf{" + kge + "}"
        L.append(f"{label} & {DISPLAY[m]} & {int(r.n)} & {nse} & {kge} \\\\")
    if k < len(PROTOCOLS) - 1:
        L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}", r"\begin{flushleft}",
      r"\footnotesize \emph{Note.} Median over the $n$ gauges scored by every model of the protocol, per lead, averaged over the leads 24, 48, 72, 96 and "
      r"120~h. Bold: the highest value per protocol. Differences below 0.02 NSE are not distinguishable (equivalence margin, Table~S6). GNN-Instat was not "
      r"trained for the gauged protocol. NWM: operational forecast, nudged toward observations "
      r"at gauges, a reference only.", r"\end{flushleft}", r"\end{table}"]
(CSV / "table3_skill.tex").write_text("\n".join(L) + "\n")

S = [r"\begin{table}", r"\caption{Paired Differences in Days 1--5 NSE}", r"\label{tab:contrasts}", r"\footnotesize", r"\setlength{\parindent}{0pt}",
     r"\renewcommand{\arraystretch}{1.2}", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
     r"Protocol & Contrast & $n$ & Difference & 5--95\% interval & Holm $p$ \\", r"\midrule"]
for k, p in enumerate(PROTOCOLS):
    c = con[con.protocol == PNAME[p]]
    for i, r in enumerate(c.itertuples()):
        label = PNAME[p] if i == 0 else ""
        cn = pretty(r.contrast).replace(" - ", r" $-$ ")
        if r.n == 0:
            why = "GNN-Instat not trained"
            S.append(f"{label} & {cn} & --- & \\multicolumn{{3}}{{c}}{{{why}}} \\\\")
            continue
        pv = r.wilcoxon_p_holm
        ps = f"{pv:.3f}" if pv >= 0.001 else r"$<$0.001"
        S.append(f"{label} & {cn} & {int(r.n)} & {r.mean_diff:+.3f} & [{r.ci5:+.3f}, {r.ci95:+.3f}] & {ps} \\\\")
    if k < len(PROTOCOLS) - 1:
        S.append(r"\addlinespace")
S += [r"\bottomrule", r"\end{tabular}", r"\begin{flushleft}",
      r"\footnotesize \emph{Note.} Difference = first minus second model, mean over the $n$ gauges both score of the per-gauge mean NSE over leads "
      r"24--120~h. Interval: gauge bootstrap (10,000 resamples). Holm $p$: Wilcoxon signed-rank test, Holm-corrected across the contrasts of each "
      r"protocol.", r"\end{flushleft}", r"\end{table}"]
(CSV / "tableS6_contrasts.tex").write_text("\n".join(S) + "\n")
print("wrote", CSV / "table3_skill.tex", "and", CSV / "tableS6_contrasts.tex")
