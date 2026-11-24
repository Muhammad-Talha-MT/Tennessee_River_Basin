# Section 5 (Results) brief: everything needed to write the section

**Paper.** "Does the River Network Help Forecast Streamflow at Unmonitored Reaches?" (Water Resources Research). Sections 1-4 are written; this file is all that is needed to write Section 5 (Results) from scratch, and it can be given to a writer who has not seen the repository. Public data only; do not mention any dam, reservoir-operator or utility data or name. Do not invent numbers: every number to be used is in PART B; if a number is not in this file, say it is not available.

How to use: PART A is context and instructions (what the paper has already said, how to write). PART B is the numbers (generated from the saved forecasts; the script and command for each table are named). PART C lists the figures and tables with captions to adapt. PART D lists what is still open.

---

# PART A. Context and instructions

## A1. The paper in brief (Sections 1-4, as written)

- **Question.** Does the river network improve streamflow forecasts at unmonitored reaches, and through which mechanism? Setting: Tennessee River Basin (HUC 06, about 106,000 km2), 58,834 NHDPlus V2.1 reaches, 60,267 links; six-hourly discharge forecasts at every reach to 240 h (40 leads); observed weather (AORC) before the 00Z issue time, archived operational GEFSv12 control-member precipitation forecasts after it (the only future weather any model sees); test period 2020-09-23 to 2025-12-22 (1,917 daily issues); training issues 2000-01-01 to 2018-12-22 (6,931), validation 2019 (355).
- **Research questions (the structure of Section 5).**
  - RQ1: does a reach-scale model that represents the river network forecast streamflow at unmonitored reaches better than a lumped regional LSTM? (Note: the LSTM is NOT input-matched; see A3.)
  - RQ2: if so, does the benefit come from learned message passing or from physically based (Manning-type) routing?
  - RQ3: do observations at upstream gauges, carried downstream through the network (insertion), improve forecasts at ungauged reaches below them?
  - For each: how does the answer change with lead time and with the strictness of the spatial hold-out?
- **Evaluation gauges.** 128 USGS gauges matched to reaches (125 CAMELS-H sites, 3 NWIS-only); 101 evaluation gauges (CAMELS-H sites with at least 100 forecast-observation pairs at 24 h, positive mean, std >= 1e-3 m3/s). They are not representative: median drainage area 382 km2 against 5.8 km2 for all reaches; 4.0% of the evaluation gauges and 77.7% of the reaches drain less than 50 km2; the smallest gauge drains 13.9 km2. Nested gauge = another of the 128 matched gauges lies upstream (44 of 101; median area 1,238 km2); headwater gauge = none (57; median 275 km2). Sensitivity split using the 101 evaluation gauges only: 39 nested / 62 headwater. Nested gauges are larger, so compare by position only at similar drainage areas.
- **Protocols (all forecasts use GEFS rain; a withheld gauge contributes nothing, as a target, input, or for model selection).**
  - Gauged: one model trained with all gauges (122 with at least 1,460 six-hour values before 2019), evaluated at the 101 gauges (3 of them have no training data).
  - Random hold-out: 122 gauges in 5 area-stratified folds; one model per fold trained without that fold's gauges and scored at them; 98 of the 101 evaluation gauges.
  - Blocked basins: three whole upstream basins withheld (B1 French Broad near Newport, USGS 03455000; B2 Holston near Knoxville, 03495500; B3 Hiwassee at Charleston, 03566000; 4.6, 9.1, 5.7% of the network area); 29 scored gauges (12, 11, 6); gauges downstream of a block stay in training. Insertion cannot act here (no scored gauge has a visible upstream gauge). The blocked basins are higher (691-940 m against 623 m) and more forested (66-81% against 64%) than the other evaluation gauges.
  - Every reach: evaluation against the NWM v3.0 retrospective simulation at all 58,834 reaches, 425 issues (every second test issue to 2023-01-20); reaches inside the basin of a randomly withheld gauge (31,684) use that gauge's fold model, the other 27,150 (25,588 outside every gauge basin + 1,562 in the basin of a gauge in no fold) use the all-gauge model. The reference is itself a model.
- **Models (Section 4).** All reach-scale models share the inputs, encoder, area scaling, NWM pretraining and gauge head and differ only in three mechanisms:

  | Model | Message passing | Routing | Insertion | Parameters | Contrast it enables |
  |---|---|---|---|---|---|
  | M0 | 3 graph layers | Manning-type | yes | 932,347 | full model |
  | M1 | 3 graph layers | Manning-type | no | 932,347 | M0 - M1: insertion (RQ3) |
  | M2 | none | Manning-type | yes | 919,675 | M0 - M2: message passing (RQ2) |
  | M4 | 3 graph layers | instantaneous (no travel time) | no | 928,654 | M1 - M4: routing (RQ2) |
  | LSTM | lumped | | | 810,369 | M0 - LSTM: the network (RQ1) |
  | NWM | operational medium-range forecast (deterministic member, 00Z, 3-hourly to 240 h, six-hour means by trapezoid) | | | | reference only |

  Mechanisms: message passing = 3 residual graph layers mixing the mean state of upstream and downstream neighbours (acts before the runoff head); Manning-type routing = lateral inflow q_j = (a_j/21.6) x runoff depth routed by Q_i(t) = sum_j f_ij sum_s h_ij(s) q_j(t-s) with kernels from Manning travel times (learned roughness, depth coefficient, lake wave speed, kernel spread); insertion = at each visible gauge a correction (observed minus routed before the issue; gauge-head forecast minus routed after it) is routed downstream with the same kernels and added to every reach below the gauge, never to the gauge itself. A gauge is visible if it is not withheld and reported at least 8 of the last 32 six-hour values. Observations of visible gauges also reach every model through three other routes (own history, specific discharge of the containing gauge broadcast to its basin, the gauge head), so M0 - M1 measures the additional value of routing a correction downstream, not the total value of upstream observations. Insertion can act at 44 of 101 gauged, 40 of 98 random hold-out and 0 of 29 blocked gauges.
- **LSTM baseline (A3 below), NWM reference (nudged to observations, including at gauges withheld from the other models; reference, not a competitor).**
- **Training.** Stage A (NWM retrospective pretraining, 5,000 updates, no gauge data) then stage B (observed discharge, 3,000 updates; loss NSE* + pinball + 0.3 x NWM prior); one random seed (seed 1) per model and protocol; checkpoint chosen on 2019 validation issues. So every interval describes variability across gauges, not across training runs.
- **Metrics.** Per gauge and lead, NSE and KGE (Gupta et al. 2009: r, alpha = std ratio, beta = mean ratio) are computed over all test issues where the observation, the observation at the issue time and the NWM forecast are present. Results: median over gauges at each lead; "days 1-5" = mean over leads 24, 48, 72, 96, 120 h of the median (for a median) or per-gauge mean over those leads (for a paired difference). Every comparison uses the gauges scored by all models in it (n stated in every table). Statistics: gauge-level bootstrap (10,000 resamples, seed 0, 5-95% interval = 90% interval), Wilcoxon signed-rank, TOST equivalence at margins 0.01 and 0.02 NSE, Holm correction within each protocol.

## A2. How to write Section 5

Structure: one subsection per research question, in the order of the Introduction (plan in A4). Each subsection opens with the one-sentence answer, then the number with n, interval and test, then the lead dependence, then the figure reference.

Rules:
1. Report numbers and what they show; interpretation and mechanisms belong in Section 6 (Discussion). No causal language in Section 5.
2. Classify each paired contrast with the `status` column of the contrast tables (PART B, section 3): **equivalent (0.02)** = Holm-corrected TOST p < 0.05 at the 0.02 margin; **different** = Holm-corrected Wilcoxon p < 0.05 and the 5-95% interval excludes zero; **undetermined** = neither. Write "M0 is better than X" only for "different". A difference whose interval includes zero is "undetermined", not "no difference".
3. The primary quantity is the mean paired difference in days 1-5 NSE; give the median difference beside it. They can disagree (random hold-out: M0 - M2 mean +0.023, median +0.001). Intervals are 90% (5-95%) gauge-bootstrap intervals, the usual display for equivalence tests; say so once.
4. One seed per model: say once, at the start of Section 5, that intervals describe variability across gauges, not across training runs. Use the noise floor (PART B, 5e) when claiming that a difference is "within 0.02".
5. NSE and KGE can disagree: report both in the overall-skill subsection (blocked protocol: the LSTM and the NWM have higher KGE than M0 although M0 has higher NSE, because M0 has a lower variability ratio alpha). Section 2.4 of the paper promised this check.
6. The every-reach evaluation is not an RQ1 test: the reference (NWM v3.0 retrospective) is the target the reach-scale models were pretrained on and the LSTM never saw, and the LSTM at reaches runs with flow history hidden everywhere. Say so in the first sentence of that subsection; compare drainage-area classes only.
7. Exploratory results (area-ratio correlation, within-gauge visible-upstream test) go to the Supporting Information, not the main text.
8. Do not use the words "significant" without the test and corrected p; do not describe non-equivalent, undetermined contrasts as "no effect".
9. Public data only; no mention of reservoir or dam operation, operators, or any utility.

## A3. The LSTM is not input-matched (must be reflected in how RQ1 is worded)

The regional LSTM reads 30 days of basin-mean weather and its own gauge's discharge, basin-mean attributes and antecedent indices, basin-mean forecast rain at 40 leads; predicts log specific discharge times area (area-scaled like the reach models). Differences from the reach-scale models: no observations from other gauges; no NWM pretraining or prior; no season input; basin means over 30 days instead of per-reach series over 36 days; different lead weights and a larger number of smaller updates (68,767 updates of 48 pairs against 3,000 updates of 8 whole-network samples); a different validation set (356 against 355 issues); selected checkpoints early in training (update 13,754 of 68,767 for the gauged model). So M0 - LSTM is a difference between two systems, of which the river network is the main but not the only part. Word RQ1 as "a reach-scale model against a regional LSTM driven by the same meteorological products", never "matched inputs".

## A4. Plan for Section 5 (from the planning discussion; revisions applied)

- **5.0 Opening (one paragraph):** state the seven or so comparisons (M0 - LSTM; M0 - M2; M1 - M4; M0 - M1; M0 - NWM as reference), the three protocols, n for each, one seed, the status rule.
- **5.1 Overall skill and the NWM reference (Figure 4, Table 3):** lead curves; days 1-5 median NSE and KGE by protocol; skill falls with the strictness of the hold-out for every model; the reach-scale variants and the LSTM lie close together; the NWM is lower in NSE but competitive in KGE and better than M0 at some gauges (counts in PART B, 9); state the KGE reversal here; state peak underestimation here (negative bias for every model, PART B 6c).
- **5.2 RQ1, reach-scale model against the LSTM (Figure 5 row M0 - LSTM, Table S6):** all three M0 - LSTM contrasts are undetermined (none survives Holm correction; the random-hold-out interval includes zero; the intervals overlap). The blocked result is suggestive only: n = 29 in three basins and the LSTM has the higher blocked KGE. Describe lead dependence (random hold-out: +0.031 at 24 h falling to -0.003 at 120 h; blocked: roughly +0.06 to +0.08 at all five leads). Do not write a trend with the strictness of the hold-out.
- **5.3 RQ2, message passing and routing (Figure 5 rows M0 - M2 and M1 - M4; area strata; learned travel times):** message passing: equivalent within 0.02 in the gauged protocol, undetermined in the other two, sign changes across protocols. Routing (M1 - M4): see PART B sections 3, 4, 6 once M4 is in them (if the contrast says MISSING, this subsection cannot be written yet). One sentence on learned travel times (they scale with river length; implied celerity about 1.1 m/s).
- **5.4 RQ3, upstream observations (Figure 6a; Table S6):** M0 - M1 is equivalent within 0.02 in the gauged and random protocols; in the blocked protocol insertion cannot act, so the difference there (+0.012; per basin +0.004, +0.019, +0.018) is the noise floor (PART B 5e). Where insertion can act (nested gauges with a visible upstream gauge) describe the difference by lead (Figure 6a: a gain only in the first hours; nothing at days 1-5; PART B 5a). Use the area-matched comparison (PART B 5c), not the raw nested/headwater split (Section 3 of the paper promises comparison at similar areas). The within-gauge test (5b) is not identifiable (upstream gauges are visible at almost every issue): one sentence, supplement. The area-ratio correlation (5d, 5f, Figure 6b) is exploratory: supplement only.
- **5.5 Behaviour at every reach (Figure 7):** only M0 and the LSTM have reach-level runs; agreement with the NWM retrospective by drainage-area class (M0 highest at 500-5,000 km2, lowest at the smallest and largest classes); the caveat in A2 rule 6. Add gauge-based skill by drainage-area stratum (PART B 5g), the only observation-based evidence below 200 km2.
- **Supporting Information:** Table S6 (every contrast), the example event (Figure S3), the area-ratio analysis, the within-gauge test, area bins, lead-by-lead contrast tables.
- **Framing decision (author's):** the honest summary may be that the river network gives no clear benefit beyond the first hours. If so the abstract and the Discussion should be written around that; Section 5 should still only report numbers and statuses.

---
