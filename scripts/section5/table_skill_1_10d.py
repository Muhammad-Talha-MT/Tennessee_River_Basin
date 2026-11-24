"""Writes reports/section5/table_skill_1_10d.tex from s5_12_table_leads.csv (scripts/section5/s5_table_leads.py): median NSE and KGE over the 40 leads (6-240 h,
1-10 days) for each protocol. Three groups of rows: the reach-scale models, the LSTM baseline (a different system, not input-matched) and the NWM (operational reference:
nudged to observations, never trained or held out by us). Bold = highest value among the trained models (reach-scale models and LSTM) in the column; the NWM is not eligible.
The row "GNN - LSTM" is the difference of the two rows above it. Run: python scripts/section5/table_skill_1_10d.py"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
d = pd.read_csv(REPO / "reports/section5/s5_12_table_leads.csv")
d = d[d.lead_h == "mean_1_10d"]
GNNS = [("M0", "GNN"), ("M1", "GNN-noIns"), ("M2", "GNN-noMP"), ("M4", "GNN-Instat")]
PROTO = [("gauged", "Gauged"), ("heldout", "Random hold-out"), ("blocked", "Blocked basins")]
fmt = lambda v: f"{v:.3f}".replace("-", "$-$")  # noqa: E731
val = {(r.protocol, r.model, k): getattr(r, k) for r in d.itertuples() for k in ("nse", "kge")}
best = {(p, k): max(val[(p, m, k)] for m in ("M0", "M1", "M2", "M4", "LSTM") if (p, m, k) in val) for p, _ in PROTO for k in ("nse", "kge")}


def cells(m, eligible=True):
    out = []
    for p, _ in PROTO:
        for k in ("nse", "kge"):
            v = val.get((p, m, k))
            out.append("---" if v is None else (f"\\textbf{{{fmt(v)}}}" if eligible and abs(v - best[(p, k)]) < 1e-12 else fmt(v)))
    return out


n = {p: int(d[d.protocol == p].n.iloc[0]) for p, _ in PROTO}
ncol = 1 + 2 * len(PROTO)
L = ["\\begin{table}", "\\centering", "\\caption{Median NSE and KGE over the full forecast horizon (6--240~h, 1--10~days)}", "\\label{tab:skill}", "\\footnotesize",
     "\\renewcommand{\\arraystretch}{1.25}", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}", "\\toprule",
     " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{nm} ($n = {n[p]}$)}}" for p, nm in PROTO) + " \\\\",
     "\\cmidrule{2-3}\\cmidrule{4-5}\\cmidrule{6-7}", "Model & NSE & KGE & NSE & KGE & NSE & KGE \\\\", "\\midrule",
     f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{Reach-scale models}}}} \\\\"]
for m, nm in GNNS:
    L.append(nm + " & " + " & ".join(cells(m)) + " \\\\")
L += ["\\midrule", f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{Baseline (different system)}}}} \\\\", "LSTM & " + " & ".join(cells("LSTM")) + " \\\\"]
diff = [fmt(val[(p, "M0", k)] - val[(p, "LSTM", k)]).replace("$-$", "$-$") if (val[(p, "M0", k)] - val[(p, "LSTM", k)]) < 0 else "+" + fmt(val[(p, "M0", k)] - val[(p, "LSTM", k)])
        for p, _ in PROTO for k in ("nse", "kge")]
L += ["GNN $-$ LSTM & " + " & ".join(diff) + " \\\\", "\\midrule", f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{Reference (not a competitor)}}}} \\\\", "NWM & " + " & ".join(cells("NWM", eligible=False)) + " \\\\",
      "\\bottomrule", "\\end{tabular*}", "\\begin{flushleft}",
      "\\footnotesize \\emph{Note.} Median over the $n$ gauges at each lead, averaged over the 40 leads of 6 to 240~h. Bold: highest value among the trained models (reach-scale models and LSTM) "
      "in each column. GNN $-$ LSTM is the difference of the two rows above it, not the paired difference of Section~\\ref{sec:res-lstm}. Differences smaller than 0.02 lie within the equivalence margin (Table~S6). GNN-Instat was not trained for the gauged protocol. The LSTM is trained in the same folds "
      "as the reach-scale models but is not matched to them in inputs (Section~\\ref{sec:baselines}). The NWM forecast is nudged toward observations, including at gauges that the other "
      "models withhold, and is shown as a reference of operational skill only.", "\\end{flushleft}", "\\end{table}"]
(REPO / "reports/section5/table_skill_1_10d.tex").write_text("\n".join(L) + "\n")
print("\n".join(l for l in L if "&" in l or "emph" in l))
