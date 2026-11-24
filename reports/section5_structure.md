# Section 5 "Results and Discussion": readiness check and structure

Built on `reports/paper.tex` (Sections 1-4, unchanged) and `reports/section5_facts.md` (regenerated 23 Nov 2026 by `scripts/section5/s5_facts.py`). Every number below is from the facts sheet or from a CSV in `reports/section5/` that the same script writes (file named where it is not in the facts sheet). The LaTeX skeleton that follows this structure is `reports/section5_skeleton.tex`.

---

## 1. Are we ready to write?

**Yes.** All runs that the three research questions need are finished and scored:
- M0, M1, M2 and the LSTM in all three protocols.
- M4 in the random hold-out and the blocked basins.
- M0, the LSTM and the NWM forecast at every reach.

Every figure and table is produced by a script from those numbers. Nothing in the structure below depends on a run that has not been made.

Settle five things before or while writing. None of them needs a new run.

| # | Item | Where | What to do |
|---|---|---|---|
| 1 | **Section numbering.** The Introduction promises "Section 5 presents the results, Section 6 discusses ..., Section 7 concludes". With a combined "Results and Discussion", the paper has Sections 5 and 6 only. | paper.tex line 113; "Section 6" in lines 374 and 379 | Change line 113 to "Section 5 presents and discusses the results, and Section 6 concludes"; point the two "(Section 6)" references to the limitations subsection (`\ref{sec:limits}`). |
| 2 | **RQ1 wording.** "a lumped LSTM with matched inputs and outputs" contradicts Section 4.4 ("It is not matched to them input for input"). | line 107 | "... than a lumped regional LSTM driven by the same meteorological products and predicting the same quantity". |
| 3 | **M4 coverage.** Section 4.5 still carries a VERIFY comment. | line 390 | One sentence: M4 is trained and evaluated for the random hold-out and the blocked basins; there is no all-gauge M4, so M1 - M4 is not available in the gauged protocol. |
| 4 | **NWM calibration gauges.** Still unverified. | line 380 | The limitations subsection needs either the result of the check or the sentence that it could not be determined. |
| 5 | **Figure overlap.** Figure 6 (days 1-5 maps, four contrasts, random hold-out, scale +-0.15) repeats the last column of Figure 9 (same maps, scale +-0.3). | Figures 6 and 9 | Recommended: move Figure 6 to the Supporting Information. Figures 7-10 then become 6-9. The skeleton keeps the current numbers, so renumber once you decide. |

### Optional runs, not needed to write

These would strengthen the text but are not required (facts sheet section 10). If they are not run, the limitations subsection states the gap.

- **LSTM seeds 2-5.** About 1.5 h. Gives a seed spread for M0 - LSTM. The gauged M0 - LSTM difference (+0.020) is about as large as the only noise-floor estimate (+0.012, one pair of runs).
- **Second stage-B seed for M0, M1, M2 (gauged).** About 9 h. Gives a data-based justification of the +-0.02 equivalence margin.
- **Every-reach evaluation of M1, M2, M4.** About 2.2 h. Gives within-family contrasts by drainage-area class that the shared NWM reference does not bias.

### What the results say (the frame for the whole section)

The network-aware model and the LSTM are close in skill. The differences are small compared with the spread across gauges, and none of the three M0 - LSTM contrasts survives the Holm correction.

Of the three mechanisms, only routing gives a difference classified as *different*: random hold-out, M1 - M4 +0.033 [+0.016, +0.053]. Message passing is equivalent within 0.01 NSE in the gauged protocol and undetermined elsewhere. Upstream-observation insertion is equivalent within 0.02 NSE at days 1-5 and helps only at the first lead (random hold-out nested gauges, +0.054 at 6 h).

The honest answer to the title question is therefore: a little, mainly through travel time, and mainly at short leads. Section 5 and the abstract should be written around this answer.

---

## 2. Structure

Each results subsection follows the same order:
1. one-sentence answer;
2. numbers (difference, 90% interval, n, test, status);
3. dependence on lead time and protocol;
4. figure reference;
5. "Discussion" paragraphs (interpretation, comparison with the literature of Section 2, what it means for forecasting).

Mechanisms and explanations appear only in the discussion paragraphs. Suggested length is 4,000-5,000 words in total.

### 5 Results and Discussion (opening, about 150 words, no subsection)
- **The comparisons:**
  - M0 - LSTM (RQ1);
  - M0 - M2 and M1 - M4 (RQ2);
  - M0 - M1 (RQ3);
  - NWM as reference only.
- **Gauge counts per protocol:** gauged n = 101, random hold-out n = 98, blocked n = 29. M4 is absent in the gauged protocol.
- **Status rule:**
  - *equivalent* = Holm-corrected TOST p < 0.05 at the stated margin;
  - *different* = Holm-corrected Wilcoxon p < 0.05 and interval excludes zero;
  - *undetermined* = neither.
- **Reporting conventions:**
  - The primary quantity is the mean paired difference in days 1-5 NSE, with the median beside it.
  - Intervals are 90% gauge-bootstrap intervals.
  - Intervals describe gauges, not training runs (one seed).
- **Pointer:** every contrast is in Table S6.

### 5.1 Skill across lead times and protocols
Figure 4, Table 3, Figure S3.

**Results**
- **Answer:** skill falls with the strictness of the hold-out for every model. M0 and the LSTM are close at all 40 leads, and both are above the NWM forecast in NSE.
- **Days 1-5 median NSE (gauged / random / blocked):**

  | Model | Gauged | Random | Blocked |
  |---|---|---|---|
  | M0 | 0.430 | 0.385 | 0.305 |
  | LSTM | 0.428 | 0.367 | 0.280 |
  | NWM | 0.255 | 0.255 | 0.183 |

- **Lead dependence:** use the full 10 days, Figure 4 top row (`s5_2_lead_curves.csv`). Curves converge with lead time.
- **KGE does not follow NSE:**
  - Blocked KGE: LSTM 0.381 and NWM 0.385 against M0 0.334.
  - Variability ratio alpha: M0 0.530 against LSTM 0.626 (blocked). In the gauged protocol M0 has the higher KGE (0.526 against 0.481).
  - Section 2.4 promised this check, so state it here because it qualifies every later NSE statement.
- **NWM better than M0** at 6 of 101, 17 of 98 and 5 of 29 gauges.
- **Peak flows:**
  - Every model underestimates the top 5% of flows: median bias -0.37 to -0.48 across models and protocols.
  - Peak timing for peaks within 24 h: M0 3 h against 6 h for the LSTM and the NWM (gauged and random); 1.5 h against 6 h (blocked).
  - Figure S3 gives one event, chosen by a fixed rule.

**Discussion**
- **The NWM** is nudged toward observations, including at gauges withheld from our models. It is a reference for operational skill, not a competitor (Section 4.4). Its NSE deficit grows with lead time (days 1-5 table, 24 h to 120 h).
- **Convergence with lead time:** precipitation-forecast error dominates at medium range (Cloke2009, Wood2008). At long leads all models share the GEFS control member, so structure matters less.
- **NSE rewards damped forecasts** (Gupta2009). The lower alpha of M0 means part of its NSE advantage comes from smoother forecasts. Connect to the peak underestimation and to flood warning (Introduction).

### 5.2 RQ1: The reach-scale model against the regional LSTM
Figure 4 bottom row, Figure 5 row 1, Figure 10, Table S6; Figure 8c for area.

**Results**
- **Answer:** M0 has higher NSE than the LSTM on average in all three protocols, but no contrast is classified as *different*. All three are *undetermined*.
- **Days 1-5 contrasts:**

  | Protocol | Mean [90% interval] | Holm p | Median | Share of gauges with M0 higher |
  |---|---|---|---|---|
  | Gauged | +0.020 [+0.007, +0.035] | 0.053 | | |
  | Random hold-out | +0.015 [-0.010, +0.037] | 0.012 | +0.035 | 66% |
  | Blocked (n = 29) | +0.066 [+0.026, +0.105] | 0.089 | | 69% |

  In the random hold-out the interval includes zero although the Wilcoxon test is significant after Holm correction; report both.
- **Lead dependence:**
  - Random hold-out: +0.031 at 24 h falling to -0.003 at 120 h.
  - Blocked: +0.057 to +0.076 at the five leads.
  - Over all 40 leads (Figure 4 bottom; `s5_4_contrasts_40_leads.csv`):

    | Protocol | Leads with mean > 0 | Leads with 90% band above zero |
    |---|---|---|
    | Gauged | 36 | 19 |
    | Random hold-out | 13 | 5 (all at or before 66 h) |
    | Blocked | 40 | 34 |

- **Where (Figure 10):** gauges with M0 higher:

  | Protocol | 6 h | 12 h | 18 h | Days 1-5 |
  |---|---|---|---|---|
  | Gauged | 84 / 101 | 71 / 101 | 71 / 101 | 60 / 101 |
  | Random hold-out | 70 / 98 | 75 / 98 | 79 / 98 | 65 / 98 |
  | Blocked | 19 / 29 | 18 / 29 | 22 / 29 | 20 / 29 |

  Before quoting these counts, take them from `s5_pergauge_nse_leads.csv` and `s5_pergauge_nse.csv` and add them to the facts sheet. They are read here from the figure panels.
- **By drainage area (Figure 8c; facts 5g):** M0 and the LSTM are equal below 50 km² (n = 3-4). M0 is higher above 600 km²: gauged 0.511 against 0.482 (600-2,000 km²), 0.559 against 0.489 (> 2,000 km²).

**Discussion**
- **A system difference:** M0 - LSTM compares two systems (Section 4.4). M0 also has other gauges' observations, NWM pretraining, the seasonal input and a different training schedule. The small gain is an upper bound on what the network alone contributes.
- **Literature:** the network gave no clear benefit here, as in Kirschstein2024. It contrasts with the gains of WangH2025 and Mosaffa2026, which were evaluated under observed meteorology and/or at training gauges. Our evaluation uses GEFS forecasts and withheld gauges (Section 2.4).
- **Blocked basins:** the gain is largest where neighbours are withheld. This is consistent with a regional LSTM being weaker in unseen terrain (Feng2021; the blocked basins are higher and more forested, Section 3.5). It rests on 29 gauges in 3 basins, and the LSTM has the higher blocked KGE, so it is suggestive only.

### 5.3 RQ2: Message passing or routing?
Figure 5 rows 2-3, Figure 9 rows 2-3, facts 6a, 6b, 6c.

**Results: message passing (M0 - M2)**
- **Gauged:** +0.005 [+0.002, +0.009], *equivalent* within 0.01.
- **Random hold-out:** +0.023 [-0.002, +0.054], median +0.001, *undetermined*. The mean is carried by a few gauges, as the gap between mean and median shows.
- **Blocked:** -0.013 [-0.038, +0.010], *undetermined*.
- **Sign** changes across protocols (+, +, -).

**Results: routing (M1 - M4)**
- **Answer:** routing gives the only mechanism contrast classified as *different*.
- **Random hold-out:** +0.033 [+0.016, +0.053], median +0.026, 76% of gauges, Holm p 1.4e-5. It is positive at every lead from 24 h to 120 h (+0.042 to +0.025). M1 is higher at 73-81 of 98 gauges at 6, 12 and 18 h (Figure 9).
- **By area (random):**
  - 600-2,000 km²: +0.055 [+0.026, +0.094], 86% of gauges.
  - 200-600 km²: +0.041.
  - < 200 km²: +0.005.
  - > 2,000 km²: +0.019.
- **Blocked:** -0.006 [-0.037, +0.021], *undetermined*. The sign changes between every pair of consecutive leads. Small basins are negative (-0.046, n = 12) and the 6 + 3 large ones positive.
- **Physical check (6b):**
  - Learned travel times scale with river length (Spearman 0.88).
  - Median 10.3 h for a median 34.3 km.
  - Implied celerity 1.06 m/s.
- **Peak timing:** short-lead peaks are timed at 3 h by M1 against 6 h by M4 (random hold-out, 6c).

**Discussion**
- **Why routing matters:** travel time, not learned neighbour mixing, is the network property that the gauges detect. Travel times of hours to days are comparable to the 6 h step and the first leads, so instantaneous routing mistimes the flood wave. This fits the gain peaking at intermediate basin sizes. Small basins have travel times below the time step, and in large basins the accumulated flow is smooth.
- **Why message passing adds little:** the routed flow and the specific discharge broadcast from the containing gauge already pass neighbour information. Three graph layers see only three links, which ties in with over-squashing (Alon2021, WangH2025).
- **Comparison with differentiable routing** (Bindas2024, Song2025): the learned parameters give physically plausible celerity.
- **Blocked:** routing parameters are predicted from reach attributes, so transfer to unseen terrain is not guaranteed. n = 29 is small.

### 5.4 RQ3: The value of upstream observations
Figure 5 row 4, Figure 7, Figure 9 row 4; Figures S3, S4.

**Results**
- **Answer:** inserting upstream observations is equivalent to not inserting them at days 1-5. It improves forecasts only at the first lead, and only at nested gauges in the random hold-out.
- **Days 1-5 contrasts (M0 - M1):**

  | Protocol | Mean [90% interval] | Status |
  |---|---|---|
  | Gauged | +0.002 [+0.001, +0.004] | *equivalent* within 0.01 |
  | Random hold-out | +0.000 [-0.007, +0.007] | *equivalent* within 0.02 |

- **Blocked (+0.012):** not a test of insertion. No gauge has a visible upstream gauge, so the value is a one-pair estimate of training-to-training differences (noise floor; B1 +0.004, B2 +0.019, B3 +0.018).
- **Where insertion can act** (random hold-out, 40 nested gauges with a visible upstream gauge; `s5_5a_position_by_lead.csv`):

  | Lead | Mean [90% interval] |
  |---|---|
  | 6 h | +0.054 [+0.037, +0.072] |
  | 12 h | +0.009 [-0.027, +0.035] |
  | 24 h | -0.011 [-0.037, +0.011] |
  | Days 1-5 | -0.003 [-0.018, +0.015] |

  Headwater gauges: about zero at all leads. The gauged protocol shows no 6 h gain at nested gauges (+0.001).
- **Area-matched** (12 pairs within a factor 1.41): difference in gain about 0 (-0.0001 [-0.014, +0.013]).
- **Figure 9 row 4:** M0 higher at 60 / 98 gauges at 6 h, 46 / 98 at days 1-5.
- **Exploratory, Supporting Information only** (one sentence here):
  - Gain at days 1-2 against the upstream gauge's area share: Spearman -0.43 in the random hold-out, +0.10 when gauged (5d, 5f).
  - The within-gauge test is not identifiable, because upstream gauges are visible at almost every issue (5b).

**Discussion**
- **Other routes:** observations reach every variant through three other routes (Section 4.3). M0 - M1 is the added value of routing a correction downstream, and the other routes already carry most of it.
- **Travel time:** the median learned travel time from upstream gauge to gauge is 10.3 h (5.3). A correction made at the issue time has passed the downstream gauge before the first daily lead. This is consistent with the gain being confined to 6 h.
- **An upper bound:** we assume observations arrive without delay (Section 3.4). Real-time NWIS latency would shrink the 6 h gain.
- **Contrast with expectation:** Section 2.3 expected the value to grow with the share of flow passing the gauge. We do not see this, and in the random hold-out the sign is reversed. It is exploratory, rests on one seed, and changes sign between protocols.
- **Practice:** for daily-to-weekly forecasts the information in upstream gauges is already used through the state of the model. Insertion matters for nowcasting (hours), not for days 1-5.

### 5.5 Forecasts at every reach
Figure 8a-b, Figures S5, S5 grid, S6.

**Results**
- **First sentence (caveat):** the reference is the NWM v3.0 retrospective. M0 was pretrained on it and the LSTM never saw it, so this measures agreement with that simulation, not accuracy. Compare drainage-area classes, not models.
- **Median NSE by area class** (< 5 / 5-50 / 50-500 / 500-5,000 / > 5,000 km²):

  | Model | < 5 | 5-50 | 50-500 | 500-5,000 | > 5,000 |
  |---|---|---|---|---|---|
  | M0 | 0.011 | 0.112 | 0.211 | 0.306 | -0.290 |
  | LSTM | -0.723 | 0.050 | 0.197 | 0.249 | -1.449 |
  | NWM forecast | -1.730 | -0.696 | -0.150 | 0.211 | -0.238 |

- **Share of reaches with NSE > 0 (M0):** 0.55, 0.81, 0.81, 0.72, 0.38.
- **Unscored large rivers:** only 529 of the 1,569 reaches above 5,000 km² are scored, because the retrospective has no usable series at most large reaches.
- **Headwater reaches** (no reach upstream; 23,172 reaches; median 1.96 km²; `s5_7_headwater_summary.csv`):

  | Model | Median NSE | Share with NSE > 0 |
  |---|---|---|
  | M0 | 0.014 | 0.56 |
  | LSTM | -0.650 | 0.27 |
  | NWM forecast | -1.643 | 0.07 |

- **Observation-based check** (Figure 8c, 5g): skill at gauges rises with area, from 0.12 (< 50 km², n = 3) to 0.48 (600-2,000 km², random hold-out). Only 4 evaluation gauges drain less than 50 km², against 77.7% of the reaches (Section 3.2).

**Discussion**
- **Circularity:** M0 tracks the retrospective at small streams because of stage-A pretraining. This shows that it inherits the reference's behaviour, not that it is right at small streams.
- **Area scaling:** the LSTM and the NWM forecast depart most where areas are smallest, which is where area scaling and forecast precipitation are least constrained.
- **The evidence gap:** below 50 km² no observation-based evidence exists in this basin, and that is where most reaches are. Unmonitored-reach forecasting needs evaluation data at small streams (Krabbenhoft2022). The every-reach evaluation is the only view of these streams, and it is model-referenced.

### 5.6 Limitations
One paragraph per item. These replace the "(Section 6)" pointers in Section 4.

1. **One training run per model and protocol.**
   - Intervals are across gauges.
   - The only estimate of run-to-run differences is the blocked M0 - M1 value (+0.012), which is as large as several reported effects.
   - The +-0.02 margin is pre-chosen, not derived from seeds.
2. **LSTM not input-matched** (Section 4.4). Its selected checkpoints are early in training (update 13,754 of 68,767, gauged).
3. **NWM as reference:**
   - The forecast is nudged at withheld gauges.
   - The retrospective is the pretraining target and possibly calibrated at withheld gauges (VERIFY, item 4 of section 1).
4. **Blocked protocol:** three basins, all in the eastern highlands, n = 29 (Section 3.5).
5. **Forecast inputs:**
   - Deterministic GEFS control member only.
   - No other forecast weather variables.
   - Observations assumed available without delay.
6. **Coverage gaps:**
   - No gauged M4.
   - Every-reach runs only for M0 and the LSTM.
   - The retrospective is missing at most large reaches.

### 5.7 What the river network contributes (synthesis, about 250 words)
- **RQ1:** a small, undetermined gain over a regional LSTM. It is larger in unseen basins, but those contain few gauges.
- **RQ2:** the part that can be measured comes from routing (travel time), not from learned message passing.
- **RQ3:** upstream observations routed downstream help only in the first hours.
- **Implication for operational forecasting at unmonitored reaches:** the network structure costs little and helps timing. The bottleneck at days 1-5 is the precipitation forecast and the lack of small-stream observations.

### 6 Conclusions (separate section, about 300 words)
Answer each RQ in one or two sentences with its headline number, then one sentence on what would change the answers (seeds, more blocked basins, small-stream gauges).

---

## 3. Figures and tables in Section 5 (current files)

| Paper number | File | Used in | Note |
|---|---|---|---|
| Figure 4 | `figures/fig4_skill_lead` | 5.1, 5.2 | full 10 days; GNN (M0), LSTM, NWM; bottom row GNN - LSTM |
| Figure 5 | `figures/fig5_contrasts` | 5.2-5.4 | the central figure: four contrasts x three protocols, +-0.02 band |
| Figure 6 | `figures/fig6_map_contrasts` | 5.2-5.4 | recommended for the SI (repeats Figure 9, last column) |
| Figure 7 | `figures/fig7_upstream` | 5.4 | M0 - M1 by lead, nested vs headwater |
| Figure 8 | `figures/fig8_skill_area` | 5.5, 5.2 | every reach (a, b) and gauges (c) by drainage area |
| Figure 9 | `figures/fig9_gauge_contrast_leads` | 5.2-5.4 | four contrasts x 6/12/18 h/days 1-5, random hold-out |
| Figure 10 | `figures/fig10_gauge_protocol_leads` | 5.2 | M0 - LSTM x three protocols x leads |
| Table 3 | `reports/section5/table3_skill.tex` | 5.1 | median NSE and KGE, days 1-5 |
| Figure S3 | `figures/figS3_example_event` | 5.1, 5.4 | example event, leads 6-24 h |
| Figure S4 | `figures/figS4_area_ratio` | 5.4 | exploratory area-ratio result |
| Figure S5 a-e, S5 grid | `figures/figS5*_reach_*` | 5.5 | M0 - LSTM at every reach by area and lead |
| Figure S6 a-e | `figures/figS6*_headwater_*` | 5.5 | headwater reaches |
| Table S6 | `reports/section5/tableS6_contrasts.tex` | 5.2-5.4 | every paired contrast |
