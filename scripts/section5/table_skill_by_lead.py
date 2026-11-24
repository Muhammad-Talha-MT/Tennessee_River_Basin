"""Writes reports/section5/table_skill_by_lead.tex from s5_12_table_leads.csv: median NSE over the gauges at single leads (6, 12, 18, 24, 48, 72, 120, 240 h), one block per
protocol; within a block the reach-scale models, the LSTM baseline and the NWM reference are separate groups. Bold = highest among the trained models (reach-scale models and
LSTM) in the column of its block; the NWM is not eligible. Run: python scripts/section5/table_skill_by_lead.py"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
d = pd.read_csv(REPO / "reports/section5/s5_12_table_leads.csv")
d = d[d.lead_h != "mean_1_10d"].copy()
d["lead_h"] = d.lead_h.astype(int)
LEADS = [6, 12, 18, 24, 48, 72, 120, 240]
GNNS = [("M0", "GNN"), ("M1", "GNN-noIns"), ("M2", "GNN-noMP"), ("M4", "GNN-Instat")]
PROTO = [("gauged", "Gauged"), ("heldout", "Random hold-out"), ("blocked", "Blocked basins")]
fmt = lambda v: f"{v:.3f}".replace("-", "$-$")  # noqa: E731
nc = len(LEADS) + 1
L = ["\\begin{table}", "\\centering", "\\caption{Median NSE at single lead times}", "\\label{tab:skill-leads}", "\\footnotesize", "\\renewcommand{\\arraystretch}{1.2}",
     "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "c" * len(LEADS) + "@{}}", "\\toprule", "Model & " + " & ".join(f"{h}~h" for h in LEADS) + " \\\\"]
for p, nm in PROTO:
    s = d[d.protocol == p]
    best = {h: s[(s.lead_h == h) & s.model.isin(["M0", "M1", "M2", "M4", "LSTM"])].nse.max() for h in LEADS}

    def row(m, label, eligible=True):
        r = s[s.model == m]
        if r.empty:
            return label + " & " + " & ".join(["---"] * len(LEADS)) + " \\\\"
        out = []
        for h in LEADS:
            v = float(r[r.lead_h == h].nse.iloc[0])
            out.append(f"\\textbf{{{fmt(v)}}}" if eligible and abs(v - best[h]) < 1e-12 else fmt(v))
        return label + " & " + " & ".join(out) + " \\\\"

    L += ["\\midrule", f"\\multicolumn{{{nc}}}{{l}}{{\\textbf{{{nm}}} ($n = {int(s.n.iloc[0])}$)}} \\\\"]
    L += [row(m, nm_) for m, nm_ in GNNS] + ["\\cmidrule{1-" + str(nc) + "}", row("LSTM", "LSTM (baseline)"), row("NWM", "NWM (reference)", eligible=False)]
L += ["\\bottomrule", "\\end{tabular*}", "\\begin{flushleft}",
      "\\footnotesize \\emph{Note.} Median over the $n$ gauges at each lead. Bold: highest value among the trained models (reach-scale models and LSTM) in each column of a block. "
      "Differences smaller than 0.02 lie within the equivalence margin (Table~S6). GNN-Instat was not trained for the gauged protocol. The LSTM is trained in the same folds but is not "
      "matched to the reach-scale models in inputs (Section~\\ref{sec:baselines}); the NWM is nudged toward observations, including at withheld gauges, and is a reference only.",
      "\\end{flushleft}", "\\end{table}"]
(REPO / "reports/section5/table_skill_by_lead.tex").write_text("\n".join(L) + "\n")
print("\n".join(l for l in L if "&" in l))
