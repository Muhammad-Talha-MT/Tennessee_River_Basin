"""Assemble reports/section5_brief.md: the single file for writing Section 5. PART A (brief_header.md) context and writing rules, PART B the generated numbers
(reports/section5_facts.md, written by s5_facts.py), PART C the figure and table catalog (below) with the LaTeX of Table 3 and Table S6, PART D open
items (brief_footer.md). Nothing is deleted: section5_facts.md stays the record of the numbers.
Run order: python scripts/section5/s5_facts.py; python scripts/section5/tables.py; python scripts/section5/build_brief.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
R = REPO / "reports"
body = (R / "section5_facts.md").read_text()
tex3 = (R / "section5" / "table3_skill.tex").read_text()
texS6 = (R / "section5" / "tableS6_contrasts.tex").read_text()
body = body.replace("# Section 5 facts sheet: results", "").lstrip()
body = body.replace("\n## ", "\n### ").replace("## 1. Run inventory", "### 1. Run inventory", 1)
body = body.replace("(`s5_", "(CSV `s5_")  # csv names stay visible; they sit in reports/section5/ of the repository

partC = f"""# PART C. Figures and tables

All figures are vector PDF plus 300 dpi PNG in `figures/`, drawn by `scripts/section5/fig*.py` from the CSVs in `reports/section5/` (no metric is recomputed in the plotting code). Journal style (`scripts/section5/style.py`): boxed axes, light grid, framed legends. Model names in figures and tables (keys M0, M1, M2, M4 stay in the CSVs and code): M0 = GNN, M1 = GNN-noIns (no insertion), M2 = GNN-noMP (no message passing), M4 = GNN-Instat (no insertion, instantaneous routing: all weight on the first step). One color per model in every figure: M0 red (thicker line), M1 blue, M2 green (dash-dot), M4 purple (dashed), LSTM orange, NWM black dashed; observations black points. Figure 7: nested gauges red, headwater gauges blue. Difference maps and the reach-level maps use the diverging palette PuOr: purple = first-named model higher, orange = second-named model higher, scale centered at zero. Numbering is the paper's (Figure 3 is the model schematic).

| Paper | File | What it shows | Where used |
|---|---|---|---|
| Figure 4 | `figures/fig4_skill_lead` | skill against lead over the full 10 days (6-240 h, 40 leads), GNN (M0) against the LSTM and the NWM only (ablations are shown elsewhere); columns (a) gauged n = 101, (b) random hold-out n = 98, (c) blocked n = 29; top row median NSE, bottom row paired difference GNN - LSTM (mean over gauges, 90% gauge-bootstrap band); days 1-5 shaded; shared y-axes per row; data `s5_2_lead_curves.csv`, `s5_4_contrasts_40_leads.csv` | 5.1: the two curves lie close together; the bottom row shows where they differ (counts of the 40 leads, script-checked: gauged mean difference > 0 at 36, 90% band above zero at 19; random hold-out > 0 at 13, band above zero at 5 (6, 18, 24, 42, 66 h), band never below zero; blocked > 0 at all 40, band above zero at 34, the last at 228 h) |
| Figure 5 | `figures/fig5_contrasts` | forest plot: mean paired days 1-5 NSE difference with 90% gauge-bootstrap interval for RQ1 M0 - LSTM, RQ2 M0 - M2 (message passing) and M1 - M4 (routing), RQ3 M0 - M1 (insertion); one marker per protocol; grey band = +-0.02 | 5.2-5.4 (central figure) |
| Figure 6 | `figures/fig6_map_contrasts` | maps of the per-gauge days 1-5 NSE difference in the random hold-out (n = 98) for the same four contrasts, on the basin with its major rivers; diverging scale clipped at +-0.15 (purple = first model better, orange = second); count of gauges > 0 per panel | 5.2-5.4: where the differences occur |
| Figure 7 | `figures/fig7_upstream` | M0 - M1 difference against lead (6-240 h) at nested and headwater gauges, mean with 90% band; (a) gauged, (b) random hold-out | 5.4 |
| Figure 8 | `figures/fig8_skill_area` | skill by drainage area for GNN (M0), LSTM and the NWM forecast only (ablations omitted), one drainage-area axis in each panel: (a) median NSE (days 1-5) at every reach against the NWM retrospective; (b) share of those reaches with NSE > 0; (c) median NSE against observations at the evaluation gauges of the random hold-out (gauge counts in tick labels; reach counts in thousands); data `s5_7_every_reach.csv` (now with NWM and share columns), `s5_5g_skill_by_area.csv` | 5.5; panels (a) and (b) carry the caveat that the reference is the retrospective the GNN was pretrained on; the smallest gauge stratum has 3 gauges |
| Figure 9 | `figures/fig9_gauge_contrast_leads` | per-gauge NSE difference maps, random hold-out (n = 98): rows = the four contrasts of Figure 6 (M0 - LSTM, M0 - M2 (message passing), M1 - M4 (routing), M0 - M1 (insertion)), columns = lead 6 h, 12 h, 18 h and days 1-5; purple = first model higher, orange = second; one scale clipped at +-0.3; number in a panel = gauges > 0 of gauges scored; row labels vertical; data `s5_pergauge_nse_leads.csv`, `s5_pergauge_nse.csv` | main text: lead-time breakdown of Figure 6 |
| Figure 10 | `figures/fig10_gauge_protocol_leads` | the same maps for M0 - LSTM in the three protocols (gauged n = 101, random hold-out n = 98, blocked n = 29), columns as in Figure 9 | main text: protocol breakdown by lead |
| Figure S3 | `figures/figS3_example_event` | example event (fixed rule): observed discharge and NWM, LSTM, M0, M1 forecasts issued 6, 12, 18 and 24 h earlier, ten days | supplement |
| Figure S5 a-e | `figures/figS5a_reach_map` ... `figS5e_reach_map` | reach-level maps of the days 1-5 NSE difference M0 - LSTM against the NWM retrospective, in five steps of drainage area (a: >= 5,000 km2; b: >= 500; c: >= 50; d: >= 5; e: all 58,834 reaches); purple = M0 higher, orange = LSTM higher, scale clipped at +-0.5, gray = reach not scored (e.g. many large reaches have no usable reference: 529 of 1,569 reaches >= 5,000 km2 are scored); line width grows with area; data `s5_perreach_nse.csv` | supplement, with the caveat of A2 rule 6 (the reference is the NWM retrospective) |
| Figure S5 grid | `figures/figS5_reach_grid` | 5 x 4 grid of reach-level maps of the NSE difference M0 - LSTM against the NWM retrospective: rows = reaches >= 5,000 / 500 / 50 / 5 km2 / all (1,569 / 5,263 / 13,131 / 31,236 / 58,834 reaches), columns = lead 6 h, 12 h, 18 h and days 1-5; purple = M0 higher, orange = LSTM higher, clipped at +-0.5, gray = not scored; the number in each panel is the median over scored reaches; data `s5_perreach_nse_leads.csv`, `s5_perreach_nse.csv` | supplement, same caveat as Figure S5 a-e |
| Figure S4 | `figures/figS4_area_ratio` | exploratory: M0 - M1 gain at days 1-2 against the area ratio of the nearest visible upstream gauge, with fits and Spearman rho | supplement |
| Figure S6 a-e | `figures/figS6a_headwater_ecdf`, `figS6b_headwater_leads`, `figS6c1_headwater_map`, `figS6c2_headwater_map`, `figS6d_headwater_area`, `figS6e_headwater_elevation` | headwater reaches (no reach upstream; 23,172 of 58,834; median area 1.96 km2), days 1-5 NSE against the NWM retrospective for M0, LSTM and the NWM forecast: (a) ECDF by area class (< 1, 1-3, >= 3 km2); (b) median NSE by lead for headwater / other < 50 km2 / >= 50 km2 reaches; (c1) map of M0 - LSTM; (c2) map of M0 NSE (gray: NSE <= 0 or not scored); (d) median NSE and share with NSE > 0 against drainage area; (e) median and interquartile range by mean catchment elevation. Axes clipped (NSE -3 to 1); data `s5_perreach_nse.csv`, `s5_headwater_lead_curves.csv` | supplement, with the caveat of A2 rule 6 (the reference is the NWM retrospective; the NWM forecast is scored against its own retrospective) |
| Table 3 | `reports/section5/table3_skill.tex` | median NSE and KGE, days 1-5, by protocol and model with n | 5.1 |
| Table S6 | `reports/section5/tableS6_contrasts.tex` | every paired contrast: n, difference, 5-95% interval, Holm-corrected p | supplement |

Suggested captions (adapt):
- **Figure 4.** Median NSE against lead time at the evaluation gauges for (a) the gauged protocol, (b) the random hold-out, and (c) the blocked basins. Shading marks days 1-5 (24-120 h). The NWM is an operational reference nudged toward observations. M4 was not trained for the gauged protocol.
- **Figure 5.** Paired differences in days 1-5 NSE (first minus second model), mean over gauges with 90% gauge-bootstrap intervals, for each research question and protocol. The grey band marks the +-0.02 equivalence margin. Each model was trained once.
- **Figure 6.** Per-gauge difference in days 1-5 NSE in the random hold-out for (a) M0 - LSTM, (b) M0 - M2, (c) M1 - M4, (d) M0 - M1. Red: the first model is better; blue: the second. Values beyond +-0.15 are shown at the ends of the scale.
- **Figure 7.** Difference in NSE between M0 and M1 against lead time at nested and headwater gauges, (a) gauged protocol, (b) random hold-out (mean and 90% gauge-bootstrap band). Insertion can act only at nested gauges with a visible upstream gauge.
- **Figure 8.** Skill by drainage area. (a) Agreement of M0 and the LSTM with the NWM v3.0 retrospective at every reach (median and interquartile range); the retrospective is the reference on which the reach-scale models were pretrained. (b) Median NSE against observed discharge at the evaluation gauges of the random hold-out.

## Table 3 (LaTeX, ready to paste)

```latex
{tex3}```

## Table S6 (LaTeX, ready to paste)

```latex
{texS6}```

---

"""
brief = (HERE / "brief_header.md").read_text() + "\n# PART B. Numbers (generated; every table is also a CSV in `reports/section5/`)\n\n" + body.rstrip() + "\n\n---\n\n" + partC + (HERE / "brief_footer.md").read_text()
(R / "section5_brief.md").write_text(brief)
print("wrote", R / "section5_brief.md", f"({len(brief.splitlines())} lines); reports/section5_facts.md is kept")
