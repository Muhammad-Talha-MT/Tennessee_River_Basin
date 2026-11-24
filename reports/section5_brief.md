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

# PART B. Numbers (generated; every table is also a CSV in `reports/section5/`)

Every number is computed by `scripts/section5/s5_facts.py` (run: `python scripts/section5/s5_facts.py`) from the saved forecasts in `<data>/models/attr` and the checkpoints, with the scoring and statistics of `scripts/v3_common.py` (`scores`: NSE, KGE (2009) and its parts, a gauge scored with at least 100 values, positive mean and std >= 1e-3 m3/s; `boot_mean`: 10,000 gauge resamples, seed 0, 5-95%; `wilcoxon_p`; `tost`; `holm`). Every table is also a CSV in `reports/section5/` (file named in each heading). Days 1-5 = leads 24, 48, 72, 96, 120 h. A median over gauges is taken per lead and then averaged over the five leads; a paired difference is the per-gauge mean over the five leads. No dam or operator data is used. MISSING = forecasts not saved; nothing is filled in.

#### 1. Run inventory (CSV `s5_1_inventory.csv`)

Selected update = the checkpoint `best.pt` (M models: one of 500, 1,000, ..., 3,000 stage-B updates; LSTM: one of 10 evaluations of 68,767 updates). One seed (seed 1) per run.

| protocol | model | runs_expected | runs_done | selected_updates | predictions |
|---|---|---|---|---|---|
| Gauged | M0 | 1 | 1 | 2500 | yes |
| Gauged | M1 | 1 | 1 | 2500 | yes |
| Gauged | M2 | 1 | 1 | 2500 | yes |
| Gauged | M4 | 0 | 0 | - | MISSING (not planned: no all-gauge M4) |
| Gauged | LSTM | 1 | 1 | 13754 | yes |
| Gauged | NWM | 0 | 0 | - | yes (operational archive) |
| Random hold-out | M0 | 5 | 5 | 2000, 2500, 2500, 2000, 2500 | yes |
| Random hold-out | M1 | 5 | 5 | 2000, 2500, 2500, 2000, 2500 | yes |
| Random hold-out | M2 | 5 | 5 | 2000, 2500, 500, 2000, 2500 | yes |
| Random hold-out | M4 | 5 | 5 | 2000, 1500, 2500, 2000, 2500 | yes |
| Random hold-out | LSTM | 5 | 5 | 34385, 20631, 13754, 20631, 20631 | yes |
| Random hold-out | NWM | 0 | 0 | - | yes (operational archive) |
| Blocked basins | M0 | 3 | 3 | 2000, 1500, 2500 | yes |
| Blocked basins | M1 | 3 | 3 | 2000, 1500, 2500 | yes |
| Blocked basins | M2 | 3 | 3 | 2500, 2500, 2500 | yes |
| Blocked basins | M4 | 3 | 3 | 2000, 1500, 2500 | yes |
| Blocked basins | LSTM | 3 | 3 | 61893, 13754, 34385 | yes |
| Blocked basins | NWM | 0 | 0 | - | yes (operational archive) |

M4: trained and evaluated for the random hold-out (5 folds) and the blocked basins (3 basins) (`eval_M4_s1`, `eval_S1_M4_s1`, both `--heldout-only`); no all-gauge M4 was planned, so M4 has no gauged-protocol entries.

### 2. Headline skill (CSV `s5_2_headline.csv`; lead curves over all 40 leads: `s5_2_lead_curves.csv`)

**Gauged** (n = 101 gauges scored by every model shown)

| model | nse_days1_5 | kge_days1_5 | r_days1_5 | alpha_days1_5 | beta_days1_5 | nse_24h | nse_48h | nse_72h | nse_96h | nse_120h | kge_24h | kge_48h | kge_72h | kge_96h | kge_120h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 0.430 | 0.526 | 0.664 | 0.677 | 0.983 | 0.614 | 0.459 | 0.401 | 0.368 | 0.307 | 0.702 | 0.584 | 0.512 | 0.436 | 0.396 |
| M1 | 0.428 | 0.526 | 0.662 | 0.679 | 0.983 | 0.612 | 0.450 | 0.402 | 0.366 | 0.310 | 0.696 | 0.581 | 0.521 | 0.436 | 0.398 |
| M2 | 0.426 | 0.517 | 0.659 | 0.662 | 0.975 | 0.608 | 0.452 | 0.400 | 0.362 | 0.306 | 0.703 | 0.574 | 0.505 | 0.422 | 0.382 |
| LSTM | 0.428 | 0.481 | 0.670 | 0.632 | 0.935 | 0.595 | 0.487 | 0.403 | 0.349 | 0.308 | 0.551 | 0.537 | 0.505 | 0.431 | 0.379 |
| NWM | 0.255 | 0.443 | 0.578 | 0.772 | 0.861 | 0.461 | 0.335 | 0.244 | 0.157 | 0.079 | 0.568 | 0.488 | 0.433 | 0.362 | 0.362 |

**Random hold-out** (n = 98 gauges scored by every model shown)

| model | nse_days1_5 | kge_days1_5 | r_days1_5 | alpha_days1_5 | beta_days1_5 | nse_24h | nse_48h | nse_72h | nse_96h | nse_120h | kge_24h | kge_48h | kge_72h | kge_96h | kge_120h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 0.385 | 0.429 | 0.661 | 0.617 | 0.933 | 0.567 | 0.443 | 0.372 | 0.294 | 0.248 | 0.591 | 0.479 | 0.413 | 0.343 | 0.317 |
| M1 | 0.383 | 0.428 | 0.663 | 0.618 | 0.944 | 0.569 | 0.452 | 0.369 | 0.291 | 0.235 | 0.592 | 0.485 | 0.410 | 0.342 | 0.308 |
| M2 | 0.386 | 0.421 | 0.657 | 0.628 | 0.954 | 0.567 | 0.440 | 0.366 | 0.303 | 0.253 | 0.555 | 0.468 | 0.437 | 0.348 | 0.295 |
| M4 | 0.379 | 0.421 | 0.647 | 0.604 | 0.915 | 0.543 | 0.443 | 0.354 | 0.304 | 0.249 | 0.570 | 0.481 | 0.419 | 0.328 | 0.306 |
| LSTM | 0.367 | 0.396 | 0.634 | 0.578 | 0.918 | 0.518 | 0.429 | 0.361 | 0.293 | 0.236 | 0.482 | 0.432 | 0.414 | 0.344 | 0.308 |
| NWM | 0.255 | 0.448 | 0.579 | 0.778 | 0.865 | 0.463 | 0.337 | 0.245 | 0.158 | 0.070 | 0.574 | 0.491 | 0.444 | 0.368 | 0.365 |

**Blocked basins** (n = 29 gauges scored by every model shown)

| model | nse_days1_5 | kge_days1_5 | r_days1_5 | alpha_days1_5 | beta_days1_5 | nse_24h | nse_48h | nse_72h | nse_96h | nse_120h | kge_24h | kge_48h | kge_72h | kge_96h | kge_120h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 0.305 | 0.334 | 0.634 | 0.530 | 0.893 | 0.457 | 0.380 | 0.275 | 0.243 | 0.169 | 0.504 | 0.404 | 0.311 | 0.259 | 0.190 |
| M1 | 0.297 | 0.334 | 0.633 | 0.550 | 0.912 | 0.462 | 0.360 | 0.272 | 0.230 | 0.159 | 0.451 | 0.416 | 0.339 | 0.266 | 0.197 |
| M2 | 0.322 | 0.341 | 0.645 | 0.486 | 0.885 | 0.463 | 0.387 | 0.305 | 0.254 | 0.201 | 0.488 | 0.410 | 0.347 | 0.266 | 0.196 |
| M4 | 0.307 | 0.324 | 0.620 | 0.509 | 0.866 | 0.485 | 0.381 | 0.293 | 0.219 | 0.159 | 0.489 | 0.396 | 0.325 | 0.234 | 0.175 |
| LSTM | 0.280 | 0.381 | 0.575 | 0.626 | 0.880 | 0.482 | 0.340 | 0.242 | 0.209 | 0.126 | 0.494 | 0.424 | 0.380 | 0.335 | 0.273 |
| NWM | 0.183 | 0.385 | 0.536 | 0.744 | 0.855 | 0.401 | 0.309 | 0.083 | 0.088 | 0.031 | 0.536 | 0.452 | 0.402 | 0.296 | 0.238 |

### 3. Paired contrasts, days 1-5 (CSV `s5_3_contrasts.csv`)

Difference = first minus second model, per gauge, on the gauges both score. Interval = bootstrap 5-95% of the mean. Holm correction across the contrasts of one protocol (Wilcoxon p, and separately the TOST p at each margin). Equivalent = Holm-corrected TOST p < 0.05. Status: equivalent (0.02) = equivalent at the 0.02 margin; different = Holm-corrected Wilcoxon p < 0.05 and the interval excludes zero; undetermined = neither. The primary quantity is the mean paired difference; the median difference is given beside it (they can disagree).

**Gauged**

| contrast | n | mean_diff | ci5 | ci95 | median_diff | share_first_better | wilcoxon_p | wilcoxon_p_holm | tost_p_0p01_holm | tost_p_0p02_holm | equivalent_0p01 | equivalent_0p02 | status | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 - LSTM | 101 | 0.020 | 0.007 | 0.035 | 0.012 | 0.594 | 0.018 | 0.053 | 1.000 | 1.000 | False | False | undetermined |  |
| M0 - NWM | 101 | 0.303 | 0.226 | 0.398 | 0.176 | 0.941 | 7.73e-17 | 4.64e-16 | 1.000 | 1.000 | False | False | different |  |
| M0 - M2 | 101 | 0.005 | 0.002 | 0.009 | 0.005 | 0.584 | 0.013 | 0.051 | 0.045 | 2.91e-10 | True | True | equivalent (0.02) |  |
| M1 - M4 | 0 |  |  |  |  |  |  |  |  |  | False | False | MISSING | MISSING: M4 |
| M0 - M1 | 101 | 0.002 | 0.001 | 0.004 | 0.001 | 0.663 | 6.34e-04 | 0.003 | 8.58e-16 | 1.48e-40 | True | True | equivalent (0.02) |  |
| M1 - LSTM | 101 | 0.017 | 0.004 | 0.032 | 0.013 | 0.584 | 0.041 | 0.082 | 1.000 | 1.000 | False | False | undetermined |  |
| M2 - LSTM | 101 | 0.015 | 0.002 | 0.028 | 0.011 | 0.564 | 0.083 | 0.083 | 1.000 | 0.985 | False | False | undetermined |  |
| M4 - LSTM | 0 |  |  |  |  |  |  |  |  |  | False | False | MISSING | MISSING: M4 |

**Random hold-out**

| contrast | n | mean_diff | ci5 | ci95 | median_diff | share_first_better | wilcoxon_p | wilcoxon_p_holm | tost_p_0p01_holm | tost_p_0p02_holm | equivalent_0p01 | equivalent_0p02 | status | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 - LSTM | 98 | 0.015 | -0.010 | 0.037 | 0.035 | 0.663 | 0.002 | 0.012 | 1.000 | 1.000 | False | False | undetermined |  |
| M0 - NWM | 98 | 0.207 | 0.125 | 0.298 | 0.145 | 0.827 | 6.78e-11 | 5.42e-10 | 1.000 | 1.000 | False | False | different |  |
| M0 - M2 | 98 | 0.023 | -0.002 | 0.054 | 6.82e-04 | 0.510 | 0.493 | 0.986 | 1.000 | 1.000 | False | False | undetermined |  |
| M1 - M4 | 98 | 0.033 | 0.016 | 0.053 | 0.026 | 0.755 | 1.93e-06 | 1.35e-05 | 1.000 | 1.000 | False | False | different |  |
| M0 - M1 | 98 | 1.26e-04 | -0.007 | 0.007 | -5.83e-04 | 0.469 | 0.906 | 0.986 | 0.091 | 4.11e-05 | False | True | equivalent (0.02) |  |
| M1 - LSTM | 98 | 0.015 | -0.014 | 0.040 | 0.038 | 0.694 | 0.002 | 0.012 | 1.000 | 1.000 | False | False | undetermined |  |
| M2 - LSTM | 98 | -0.009 | -0.052 | 0.027 | 0.020 | 0.684 | 0.004 | 0.015 | 1.000 | 1.000 | False | False | undetermined |  |
| M4 - LSTM | 98 | -0.018 | -0.059 | 0.016 | 0.020 | 0.633 | 0.091 | 0.274 | 1.000 | 1.000 | False | False | undetermined |  |

**Blocked basins**

| contrast | n | mean_diff | ci5 | ci95 | median_diff | share_first_better | wilcoxon_p | wilcoxon_p_holm | tost_p_0p01_holm | tost_p_0p02_holm | equivalent_0p01 | equivalent_0p02 | status | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 - LSTM | 29 | 0.066 | 0.026 | 0.105 | 0.040 | 0.690 | 0.018 | 0.089 | 1.000 | 1.000 | False | False | undetermined |  |
| M0 - NWM | 29 | 0.390 | 0.186 | 0.643 | 0.161 | 0.828 | 1.19e-05 | 9.54e-05 | 1.000 | 1.000 | False | False | different |  |
| M0 - M2 | 29 | -0.013 | -0.038 | 0.010 | -0.002 | 0.483 | 0.579 | 1.000 | 1.000 | 1.000 | False | False | undetermined |  |
| M1 - M4 | 29 | -0.006 | -0.037 | 0.021 | 0.023 | 0.655 | 0.508 | 1.000 | 1.000 | 1.000 | False | False | undetermined |  |
| M0 - M1 | 29 | 0.012 | 0.002 | 0.026 | -6.37e-04 | 0.448 | 0.369 | 1.000 | 1.000 | 1.000 | False | False | undetermined |  |
| M1 - LSTM | 29 | 0.053 | 0.015 | 0.090 | 0.043 | 0.655 | 0.037 | 0.146 | 1.000 | 1.000 | False | False | undetermined |  |
| M2 - LSTM | 29 | 0.079 | 0.042 | 0.116 | 0.054 | 0.759 | 0.001 | 0.010 | 1.000 | 1.000 | False | False | different |  |
| M4 - LSTM | 29 | 0.059 | 0.020 | 0.100 | 0.048 | 0.655 | 0.014 | 0.083 | 1.000 | 1.000 | False | False | undetermined |  |

### 4. Contrasts by lead (CSV `s5_4_contrasts_by_lead.csv`; all 40 leads: `s5_4_contrasts_40_leads.csv`)

**Gauged** (mean paired difference [5-95%])

| contrast | n | 24 h | 48 h | 72 h | 96 h | 120 h |
|---|---|---|---|---|---|---|
| M0 - LSTM | 101 | +0.030 [+0.008, +0.055] | +0.010 [-0.006, +0.028] | +0.027 [+0.012, +0.043] | +0.019 [+0.007, +0.032] | +0.012 [+0.003, +0.022] |
| M0 - NWM | 101 | +0.330 [+0.236, +0.436] | +0.229 [+0.166, +0.307] | +0.228 [+0.165, +0.303] | +0.270 [+0.209, +0.344] | +0.458 [+0.315, +0.632] |
| M0 - M2 | 101 | +0.005 [-0.001, +0.011] | +0.005 [+0.000, +0.010] | +0.004 [-0.000, +0.009] | +0.005 [+0.002, +0.009] | +0.005 [+0.001, +0.009] |
| M0 - M1 | 101 | +0.003 [+0.000, +0.005] | +0.005 [+0.002, +0.009] | +0.002 [+0.001, +0.003] | +0.001 [-0.000, +0.002] | +0.001 [+0.000, +0.003] |
| M1 - LSTM | 101 | +0.027 [+0.005, +0.052] | +0.005 [-0.011, +0.023] | +0.025 [+0.011, +0.041] | +0.018 [+0.006, +0.031] | +0.010 [+0.001, +0.021] |
| M2 - LSTM | 101 | +0.025 [+0.004, +0.048] | +0.005 [-0.010, +0.021] | +0.023 [+0.010, +0.037] | +0.014 [+0.002, +0.026] | +0.007 [-0.003, +0.017] |

**Random hold-out** (mean paired difference [5-95%])

| contrast | n | 24 h | 48 h | 72 h | 96 h | 120 h |
|---|---|---|---|---|---|---|
| M0 - LSTM | 98 | +0.031 [+0.002, +0.059] | +0.020 [-0.003, +0.043] | +0.017 [-0.008, +0.040] | +0.008 [-0.017, +0.031] | -0.003 [-0.036, +0.024] |
| M0 - NWM | 98 | +0.247 [+0.166, +0.337] | +0.152 [+0.091, +0.223] | +0.124 [+0.043, +0.203] | +0.159 [+0.068, +0.246] | +0.351 [+0.197, +0.530] |
| M0 - M2 | 98 | +0.006 [-0.016, +0.030] | +0.024 [+0.002, +0.051] | +0.032 [-0.001, +0.073] | +0.027 [+0.001, +0.059] | +0.027 [-0.004, +0.067] |
| M1 - M4 | 98 | +0.042 [+0.021, +0.064] | +0.039 [+0.021, +0.059] | +0.029 [+0.012, +0.050] | +0.029 [+0.015, +0.045] | +0.025 [+0.006, +0.050] |
| M0 - M1 | 98 | -0.005 [-0.016, +0.004] | -0.007 [-0.017, +0.002] | +0.000 [-0.007, +0.009] | +0.006 [-0.001, +0.015] | +0.007 [+0.002, +0.014] |
| M1 - LSTM | 98 | +0.036 [+0.007, +0.064] | +0.028 [+0.002, +0.053] | +0.017 [-0.013, +0.044] | +0.003 [-0.029, +0.029] | -0.010 [-0.048, +0.020] |
| M2 - LSTM | 98 | +0.025 [-0.006, +0.053] | -0.004 [-0.040, +0.028] | -0.015 [-0.067, +0.025] | -0.019 [-0.066, +0.018] | -0.031 [-0.098, +0.018] |
| M4 - LSTM | 98 | -0.006 [-0.042, +0.027] | -0.011 [-0.050, +0.023] | -0.012 [-0.049, +0.020] | -0.026 [-0.066, +0.007] | -0.036 [-0.093, +0.007] |

**Blocked basins** (mean paired difference [5-95%])

| contrast | n | 24 h | 48 h | 72 h | 96 h | 120 h |
|---|---|---|---|---|---|---|
| M0 - LSTM | 29 | +0.076 [+0.022, +0.131] | +0.057 [+0.003, +0.110] | +0.069 [+0.018, +0.119] | +0.065 [+0.028, +0.103] | +0.061 [+0.022, +0.103] |
| M0 - NWM | 29 | +0.423 [+0.214, +0.663] | +0.275 [+0.136, +0.458] | +0.275 [+0.131, +0.457] | +0.313 [+0.141, +0.526] | +0.664 [+0.223, +1.207] |
| M0 - M2 | 29 | -0.020 [-0.060, +0.015] | -0.018 [-0.048, +0.010] | -0.018 [-0.044, +0.008] | -0.003 [-0.023, +0.016] | -0.007 [-0.024, +0.010] |
| M1 - M4 | 29 | -0.023 [-0.080, +0.026] | +0.000 [-0.044, +0.038] | -0.009 [-0.040, +0.019] | +0.008 [-0.010, +0.025] | -0.008 [-0.025, +0.008] |
| M0 - M1 | 29 | +0.022 [+0.003, +0.047] | +0.010 [-0.003, +0.029] | +0.011 [+0.001, +0.024] | +0.009 [+0.002, +0.016] | +0.010 [+0.005, +0.017] |
| M1 - LSTM | 29 | +0.055 [+0.005, +0.102] | +0.047 [-0.012, +0.105] | +0.057 [+0.007, +0.108] | +0.056 [+0.021, +0.092] | +0.050 [+0.015, +0.090] |
| M2 - LSTM | 29 | +0.097 [+0.040, +0.160] | +0.075 [+0.033, +0.119] | +0.086 [+0.040, +0.133] | +0.068 [+0.031, +0.106] | +0.068 [+0.029, +0.112] |
| M4 - LSTM | 29 | +0.077 [+0.010, +0.151] | +0.047 [+0.003, +0.091] | +0.066 [+0.024, +0.110] | +0.048 [+0.011, +0.088] | +0.059 [+0.019, +0.104] |

Sign changes across the five leads (CSV `s5_4_sign_changes.csv`):

| protocol | contrast | sign_changes |
|---|---|---|
| Gauged | M0 - LSTM | none |
| Gauged | M0 - NWM | none |
| Gauged | M0 - M2 | none |
| Gauged | M0 - M1 | none |
| Gauged | M1 - LSTM | none |
| Gauged | M2 - LSTM | none |
| Random hold-out | M0 - LSTM | 96->120 h |
| Random hold-out | M0 - NWM | none |
| Random hold-out | M0 - M2 | none |
| Random hold-out | M1 - M4 | none |
| Random hold-out | M0 - M1 | 48->72 h |
| Random hold-out | M1 - LSTM | 96->120 h |
| Random hold-out | M2 - LSTM | 24->48 h |
| Random hold-out | M4 - LSTM | none |
| Blocked basins | M0 - LSTM | none |
| Blocked basins | M0 - NWM | none |
| Blocked basins | M0 - M2 | none |
| Blocked basins | M1 - M4 | 24->48 h, 48->72 h, 72->96 h, 96->120 h |
| Blocked basins | M0 - M1 | none |
| Blocked basins | M1 - LSTM | none |
| Blocked basins | M2 - LSTM | none |
| Blocked basins | M4 - LSTM | none |

### 5. Upstream observations (RQ3)

### 5a. M0 - M1 by position (CSV `s5_5a_position.csv`; by lead: `s5_5a_position_by_lead.csv`)

Three definitions of nested: (1) a catalog gauge upstream that the model scoring the gauge may see (not withheld in its fold): the definition the insertion can use; (2) another of the 128 catalog gauges upstream; (3) another of the 101 evaluation gauges upstream (sensitivity).

| protocol | definition | group | n | mean_diff | ci5 | ci95 | median_diff | share_M0_better | wilcoxon_p |
|---|---|---|---|---|---|---|---|---|---|
| Gauged | visible upstream gauge (protocol) | nested | 44 | 0.003 | 9.25e-04 | 0.005 | 0.001 | 0.659 | 0.014 |
| Gauged | visible upstream gauge (protocol) | headwater | 57 | 0.002 | 4.00e-04 | 0.004 | 0.001 | 0.667 | 0.015 |
| Gauged | 128-gauge definition | nested | 44 | 0.003 | 9.25e-04 | 0.005 | 0.001 | 0.659 | 0.014 |
| Gauged | 128-gauge definition | headwater | 57 | 0.002 | 4.00e-04 | 0.004 | 0.001 | 0.667 | 0.015 |
| Gauged | 101-gauge definition | nested | 39 | 0.003 | 0.001 | 0.006 | 0.002 | 0.667 | 0.014 |
| Gauged | 101-gauge definition | headwater | 62 | 0.002 | 4.02e-04 | 0.003 | 0.001 | 0.661 | 0.014 |
| Random hold-out | visible upstream gauge (protocol) | nested | 40 | -0.003 | -0.018 | 0.015 | 9.77e-04 | 0.500 | 0.735 |
| Random hold-out | visible upstream gauge (protocol) | headwater | 58 | 0.002 | -0.001 | 0.006 | -6.81e-04 | 0.448 | 0.874 |
| Random hold-out | 128-gauge definition | nested | 44 | -0.002 | -0.016 | 0.013 | 9.64e-04 | 0.500 | 0.858 |
| Random hold-out | 128-gauge definition | headwater | 54 | 0.002 | -0.002 | 0.006 | -6.81e-04 | 0.444 | 0.820 |
| Random hold-out | 101-gauge definition | nested | 39 | -0.003 | -0.019 | 0.014 | -7.17e-04 | 0.462 | 0.566 |
| Random hold-out | 101-gauge definition | headwater | 59 | 0.002 | -9.96e-04 | 0.006 | -5.53e-04 | 0.475 | 0.827 |
| Blocked basins | visible upstream gauge (protocol) | nested | 0 |  |  |  |  |  |  |
| Blocked basins | visible upstream gauge (protocol) | headwater | 29 | 0.012 | 0.002 | 0.026 | -6.37e-04 | 0.448 | 0.369 |
| Blocked basins | 128-gauge definition | nested | 11 | 0.002 | -0.008 | 0.014 | -0.009 | 0.364 | 0.638 |
| Blocked basins | 128-gauge definition | headwater | 18 | 0.019 | 0.004 | 0.039 | 0.002 | 0.500 | 0.167 |
| Blocked basins | 101-gauge definition | nested | 10 | 0.002 | -0.009 | 0.015 | -0.010 | 0.300 | 0.695 |
| Blocked basins | 101-gauge definition | headwater | 19 | 0.018 | 0.004 | 0.037 | 0.004 | 0.526 | 0.145 |

### 5b. Within nested gauges: issues with a visible upstream gauge against issues without (CSV `s5_5b_within_gauge.csv`, `s5_5b_within_gauge_summary.csv`)

Visible = at least 8 of the last 32 six-hour values at an upstream catalog gauge that is not withheld (the model's rule). NSE per gauge on each subset of issues (subset variance; at least 30 values per lead), days 1-5 mean of M0 - M1. Difference = gain on visible issues minus gain on the other issues, per gauge.

| protocol | nested_gauges | with_both_subsets | median_share_issues_visible | mean_gain_visible | mean_gain_not_visible | mean_diff | ci5 | ci95 | wilcoxon_p |
|---|---|---|---|---|---|---|---|---|---|
| Gauged | 44 | 1 | 1.000 | -0.015 | -0.023 | 0.008 | 0.008 | 0.008 |  |
| Random hold-out | 40 | 2 | 1.000 | 0.003 | -0.013 | 0.016 | 0.010 | 0.022 |  |

Gauged: share of test issues with a visible upstream gauge, per nested gauge: min 0.000, quartiles 0.999 / 1.000 / 1.000; 5 of 44 gauges below 0.9. Only the gauges with at least 30 values per lead in both subsets give a within-gauge difference (column with_both_subsets).

Random hold-out: share of test issues with a visible upstream gauge, per nested gauge: min 0.000, quartiles 0.999 / 1.000 / 1.000; 4 of 40 gauges below 0.9. Only the gauges with at least 30 values per lead in both subsets give a within-gauge difference (column with_both_subsets).

### 5c. Area-matched comparison (CSV `s5_5c_area_bins.csv`, `s5_5c_matched.csv`)

Nested and headwater as in 5a definition (1). Area bins (NHDPlus drainage area):

| protocol | area_km2 | n_nested | mean_gain_nested | n_headwater | mean_gain_headwater |
|---|---|---|---|---|---|
| Gauged | < 200 | 3 | 2.54e-05 | 24 | 0.004 |
| Gauged | 200-600 | 4 | 7.60e-04 | 29 | 3.31e-04 |
| Gauged | 600-2,000 | 25 | 0.002 | 4 | 0.003 |
| Gauged | > 2,000 | 12 | 0.006 | 0 |  |
| Random hold-out | < 200 | 2 | 0.001 | 24 | -0.002 |
| Random hold-out | 200-600 | 3 | 0.007 | 28 | 0.006 |
| Random hold-out | 600-2,000 | 24 | 0.008 | 5 | 0.001 |
| Random hold-out | > 2,000 | 11 | -0.030 | 1 | -0.002 |
| Blocked basins | < 200 | 0 |  | 12 | 0.023 |
| Blocked basins | 200-600 | 0 |  | 8 | 0.017 |
| Blocked basins | 600-2,000 | 0 |  | 6 | -0.002 |
| Blocked basins | > 2,000 | 0 |  | 3 | -0.011 |

Nearest-neighbour matching on log10 area, greedy without replacement from the smallest nested gauge, caliper 0.15 (a factor 1.41); difference = gain of the nested gauge minus gain of its matched headwater gauge:

| protocol | nested_gauges | matched_pairs | caliper_log10 | mean_gain_nested | mean_gain_headwater | mean_diff | ci5 | ci95 | wilcoxon_p | median_area_nested | median_area_headwater |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gauged | 44 | 12 | 0.150 | 9.77e-04 | 0.001 | -2.66e-04 | -0.004 | 0.003 | 0.791 | 358.038 | 364.275 |
| Random hold-out | 40 | 12 | 0.150 | 2.60e-04 | 3.71e-04 | -1.10e-04 | -0.014 | 0.013 | 0.910 | 686.401 | 694.617 |
| Blocked basins | 0 | 0 | 0.150 |  |  |  |  |  |  |  |  |

### 5d. Gain at days 1-2 against the upstream gauge's area share and travel time (CSV `s5_5d_gain_vs_ratio.csv`, `s5_5d_rank_correlations.csv`)

Gain = per-gauge mean of M0 - M1 NSE at leads 24 and 48 h. Upstream gauge = the visible upstream catalog gauge with the largest drainage area. Area ratio = its NHDPlus drainage area over the gauge's. Travel time = pair travel time of the trained M0 router that scores the gauge (gauged: M0_s1; random hold-out: the fold model) at the mean NWM retrospective flow of each reach, from the middle of the upstream gauge reach to the end of the gauge reach along the main path (`H/channel.py: reach_time, pair_time`).

| protocol | variable | n | spearman_rho | p_value |
|---|---|---|---|---|
| Gauged | area_ratio | 44 | 0.097 | 0.529 |
| Gauged | travel_time_h_M0 | 43 | -0.051 | 0.747 |
| Random hold-out | area_ratio | 40 | -0.432 | 0.005 |
| Random hold-out | travel_time_h_M0 | 39 | 0.398 | 0.012 |

### 5e. Noise floor: M0 - M1 in the blocked protocol (CSV `s5_5e_noise_floor.csv`)

In the blocked protocol no scored gauge has a visible upstream gauge, so insertion cannot act at test time and M0 and M1 can differ only through training (each was trained once). The difference below is therefore an estimate, from one pair of runs, of the size of training-to-training differences; it is not a variance estimate. Compare every 'within 0.02' statement with it.

| unit | n | mean_diff | median_diff |
|---|---|---|---|
| all blocked gauges | 29 | 0.012 | -6.37e-04 |
| basin B1 | 12 | 0.004 | -0.005 |
| basin B2 | 11 | 0.019 | -0.006 |
| basin B3 | 6 | 0.018 | 0.017 |

### 5f. Area-ratio correlation: robustness, exploratory (CSV `s5_5f_area_ratio_robustness.csv`)

Spearman correlation of the M0 - M1 gain at days 1-2 with the area ratio (5d), with leave-one-out range, the correlation after dropping the three gauges with the largest absolute gain, and a 5-95% gauge bootstrap (2,000 resamples, seed 0). One seed per model and many looks at the data: exploratory.

| protocol | n | rho | loo_min | loo_max | rho_without_3_largest_gains | p_without_3 | boot_ci5 | boot_ci95 | largest_gains |
|---|---|---|---|---|---|---|---|---|---|
| Gauged | 44 | 0.097 | 0.035 | 0.141 | 0.126 | 0.431 | -0.163 | 0.323 | 03592000 (-0.019), 03559500 (-0.019), 03592500 (-0.018) |
| Random hold-out | 40 | -0.432 | -0.464 | -0.393 | -0.346 | 0.036 | -0.613 | -0.186 | 03461500 (-0.362), 03453500 (-0.184), 03597860 (+0.150) |

### 5g. Gauge-based skill by drainage-area stratum (CSV `s5_5g_skill_by_area.csv`)

Median NSE (days 1-5) of the gauges in each stratum, from observations. The smallest evaluation gauge drains 13.9 km2; 4 evaluation gauges drain less than 50 km2 and 27 less than 200 km2. This is the only observation-based evidence below 200 km2 (the every-reach reference is a model). n is the number of gauges in the stratum scored by every model shown.

| protocol | area_km2 | n | M0_median_nse | M1_median_nse | M2_median_nse | LSTM_median_nse | NWM_median_nse | M4_median_nse |
|---|---|---|---|---|---|---|---|---|
| Gauged | < 50 | 4 | 0.158 | 0.161 | 0.171 | 0.167 | -0.279 |  |
| Gauged | 50-200 | 23 | 0.330 | 0.330 | 0.335 | 0.320 | 0.166 |  |
| Gauged | 200-600 | 33 | 0.420 | 0.419 | 0.432 | 0.430 | 0.218 |  |
| Gauged | 600-2,000 | 29 | 0.511 | 0.507 | 0.514 | 0.482 | 0.330 |  |
| Gauged | > 2,000 | 12 | 0.559 | 0.551 | 0.556 | 0.489 | 0.446 |  |
| Random hold-out | < 50 | 3 | 0.119 | 0.122 | 0.119 | 0.122 | -0.368 | 0.123 |
| Random hold-out | 50-200 | 23 | 0.305 | 0.306 | 0.296 | 0.270 | 0.166 | 0.310 |
| Random hold-out | 200-600 | 31 | 0.382 | 0.378 | 0.394 | 0.374 | 0.203 | 0.367 |
| Random hold-out | 600-2,000 | 29 | 0.484 | 0.489 | 0.486 | 0.442 | 0.330 | 0.453 |
| Random hold-out | > 2,000 | 12 | 0.477 | 0.530 | 0.475 | 0.453 | 0.446 | 0.510 |
| Blocked basins | < 50 | 3 | 0.105 | 0.107 | 0.115 | 0.058 | -0.368 | 0.119 |
| Blocked basins | 50-200 | 9 | 0.320 | 0.319 | 0.299 | 0.285 | 0.174 | 0.314 |
| Blocked basins | 200-600 | 8 | 0.308 | 0.286 | 0.344 | 0.289 | 0.086 | 0.331 |
| Blocked basins | 600-2,000 | 6 | 0.334 | 0.341 | 0.396 | 0.307 | 0.295 | 0.328 |
| Blocked basins | > 2,000 | 3 | 0.496 | 0.507 | 0.495 | 0.315 | 0.533 | 0.440 |

### 6. Routing and message passing (RQ2)

### 6a. By drainage-area stratum (CSV `s5_6_area_strata.csv`); by lead: section 4

| protocol | contrast | area_km2 | n | note | mean_diff | ci5 | ci95 | share_first_better |
|---|---|---|---|---|---|---|---|---|
| Gauged | M0 - M2 | < 200 | 27 |  | 0.003 | -0.002 | 0.008 | 0.593 |
| Gauged | M0 - M2 | 200-600 | 33 |  | 0.008 | 9.59e-04 | 0.017 | 0.515 |
| Gauged | M0 - M2 | 600-2,000 | 29 |  | 0.002 | -0.004 | 0.008 | 0.586 |
| Gauged | M0 - M2 | > 2,000 | 12 |  | 0.008 | 0.004 | 0.012 | 0.750 |
| Gauged | M1 - M4 | all | 0 | MISSING |  |  |  |  |
| Random hold-out | M0 - M2 | < 200 | 26 |  | 0.005 | -0.008 | 0.018 | 0.538 |
| Random hold-out | M0 - M2 | 200-600 | 31 |  | 0.037 | -0.017 | 0.109 | 0.419 |
| Random hold-out | M0 - M2 | 600-2,000 | 29 |  | 0.027 | -0.024 | 0.101 | 0.552 |
| Random hold-out | M0 - M2 | > 2,000 | 12 |  | 0.021 | -0.002 | 0.056 | 0.583 |
| Random hold-out | M1 - M4 | < 200 | 26 |  | 0.005 | -0.013 | 0.020 | 0.692 |
| Random hold-out | M1 - M4 | 200-600 | 31 |  | 0.041 | 0.004 | 0.092 | 0.677 |
| Random hold-out | M1 - M4 | 600-2,000 | 29 |  | 0.055 | 0.026 | 0.094 | 0.862 |
| Random hold-out | M1 - M4 | > 2,000 | 12 |  | 0.019 | -0.011 | 0.042 | 0.833 |
| Blocked basins | M0 - M2 | < 200 | 12 |  | -0.003 | -0.050 | 0.040 | 0.500 |
| Blocked basins | M0 - M2 | 200-600 | 8 |  | -0.017 | -0.051 | 0.019 | 0.250 |
| Blocked basins | M0 - M2 | 600-2,000 | 6 |  | -0.037 | -0.111 | 0.005 | 0.500 |
| Blocked basins | M0 - M2 | > 2,000 | 3 |  | 0.004 | 0.002 | 0.007 | 1.000 |
| Blocked basins | M1 - M4 | < 200 | 12 |  | -0.046 | -0.107 | 0.007 | 0.583 |
| Blocked basins | M1 - M4 | 200-600 | 8 |  | -0.016 | -0.045 | 0.012 | 0.500 |
| Blocked basins | M1 - M4 | 600-2,000 | 6 |  | 0.045 | 0.008 | 0.086 | 0.833 |
| Blocked basins | M1 - M4 | > 2,000 | 3 |  | 0.076 | 0.065 | 0.087 | 1.000 |

### 6b. Learned travel times (CSV `s5_6_travel_times.csv`)

Nested evaluation gauges and their upstream catalog gauge with the largest drainage area; pair travel time at the mean NWM flow from the gauged-protocol models M0_s1 and M2_s1; river length along the main path from the middle of the upstream reach to the end of the gauge reach.

Pairs: 44 (43 on the main path). River length median 34.3 km (range 4.8-239.3). Travel time M0 median 10.3 h (range 0.3-124.7); M2 median 9.6 h (range 0.2-126.8). Implied celerity (length / time) median M0 1.06 m/s, M2 1.06 m/s. Spearman rho of travel time and length: M0 0.88, M2 0.86.

### 6c. High flows (CSV `s5_6_highflow.csv`)

Logic of `scripts/v3_kge_highflow.py`: peak bias = mean over leads 24-120 h of (mean forecast / mean observation - 1) over the top 5% of observed flows; events = 10-day windows whose observed maximum reaches the 2-year flow (median annual maximum, water years with at least 70% of values, at least 10 years), timing error = forecast minus observed peak step x 6 h, median over the gauge's events (at least 3), then median over gauges.

Peak-flow bias (median over gauges, top 5% of observed flows) is negative for every model in every protocol: Gauged M0 -0.37; Gauged M1 -0.37; Gauged M2 -0.39; Gauged LSTM -0.41; Gauged NWM -0.41; Random hold-out M0 -0.42; Random hold-out M1 -0.41; Random hold-out M2 -0.42; Random hold-out M4 -0.43; Random hold-out LSTM -0.45; Random hold-out NWM -0.41; Blocked basins M0 -0.42; Blocked basins M1 -0.40; Blocked basins M2 -0.48; Blocked basins M4 -0.48; Blocked basins LSTM -0.42; Blocked basins NWM -0.44.

| protocol | model | n | median_peak_bias_days1_5 | gauges_peak <= 24 h | median_timing_abs_h_peak <= 24 h | median_magnitude_rel_error_peak <= 24 h | gauges_peak 24-72 h | median_timing_abs_h_peak 24-72 h | median_magnitude_rel_error_peak 24-72 h | gauges_peak 72-120 h | median_timing_abs_h_peak 72-120 h | median_magnitude_rel_error_peak 72-120 h | gauges_peak 120-240 h | median_timing_abs_h_peak 120-240 h | median_magnitude_rel_error_peak 120-240 h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gauged | M0 | 101 | -0.373 | 62 | 3.000 | -0.510 | 75 | 6.000 | -0.726 | 75 | 6.000 | -0.802 | 83 | 24.000 | -0.882 |
| Gauged | M1 | 101 | -0.373 | 62 | 3.000 | -0.516 | 75 | 6.000 | -0.727 | 75 | 9.000 | -0.795 | 83 | 24.000 | -0.881 |
| Gauged | M2 | 101 | -0.388 | 62 | 4.500 | -0.529 | 75 | 6.000 | -0.742 | 75 | 9.000 | -0.823 | 83 | 24.000 | -0.892 |
| Gauged | LSTM | 101 | -0.410 | 62 | 6.000 | -0.591 | 75 | 6.000 | -0.726 | 75 | 12.000 | -0.807 | 83 | 30.000 | -0.884 |
| Gauged | NWM | 101 | -0.414 | 62 | 6.000 | -0.398 | 75 | 15.000 | -0.651 | 75 | 24.000 | -0.771 | 83 | 36.000 | -0.856 |
| Random hold-out | M0 | 98 | -0.422 | 62 | 3.000 | -0.500 | 75 | 6.000 | -0.732 | 75 | 9.000 | -0.807 | 83 | 24.000 | -0.894 |
| Random hold-out | M1 | 98 | -0.410 | 62 | 3.000 | -0.458 | 75 | 6.000 | -0.720 | 75 | 9.000 | -0.814 | 83 | 24.000 | -0.897 |
| Random hold-out | M2 | 98 | -0.420 | 62 | 3.000 | -0.521 | 75 | 6.000 | -0.748 | 75 | 12.000 | -0.820 | 83 | 24.000 | -0.895 |
| Random hold-out | M4 | 98 | -0.426 | 62 | 6.000 | -0.489 | 75 | 6.000 | -0.736 | 75 | 9.000 | -0.842 | 83 | 24.000 | -0.903 |
| Random hold-out | LSTM | 98 | -0.452 | 62 | 6.000 | -0.614 | 75 | 6.000 | -0.748 | 75 | 12.000 | -0.831 | 83 | 30.000 | -0.900 |
| Random hold-out | NWM | 98 | -0.412 | 62 | 6.000 | -0.398 | 75 | 15.000 | -0.651 | 75 | 24.000 | -0.771 | 83 | 36.000 | -0.856 |
| Blocked basins | M0 | 29 | -0.420 | 18 | 1.500 | -0.595 | 22 | 6.000 | -0.743 | 22 | 12.000 | -0.851 | 24 | 24.000 | -0.912 |
| Blocked basins | M1 | 29 | -0.403 | 18 | 1.500 | -0.565 | 22 | 6.000 | -0.732 | 22 | 12.000 | -0.835 | 24 | 24.000 | -0.903 |
| Blocked basins | M2 | 29 | -0.479 | 18 | 4.500 | -0.646 | 22 | 6.000 | -0.773 | 22 | 12.000 | -0.859 | 24 | 24.000 | -0.913 |
| Blocked basins | M4 | 29 | -0.480 | 18 | 3.000 | -0.694 | 22 | 6.000 | -0.814 | 22 | 12.000 | -0.871 | 24 | 24.000 | -0.910 |
| Blocked basins | LSTM | 29 | -0.421 | 18 | 6.000 | -0.600 | 22 | 6.000 | -0.725 | 22 | 12.000 | -0.832 | 24 | 30.000 | -0.906 |
| Blocked basins | NWM | 29 | -0.444 | 18 | 6.000 | -0.392 | 22 | 18.000 | -0.703 | 22 | 25.500 | -0.792 | 24 | 40.500 | -0.885 |

### 7. Every reach (CSV `s5_7_every_reach.csv`, `s5_7_routes.csv`)

Against the NWM v3.0 retrospective, 425 issues (every second test issue to 2023-01-20; M0 manifest: 425). Per-reach NSE and KGE, days 1-5 mean, then median (and quartiles) over the reaches of a class. M0: `eval_M0_reach/reach_metrics.npz`; LSTM (L4): `ladder_L4_s1_reaches/metrics.npz` (the LSTM can be run at reaches: `scripts/lstm_every_reach.py`, basin-mean inputs of each reach, flow history hidden everywhere). M1, M2, M4: MISSING (no every-reach evaluation run).

| route | reaches |
|---|---|
| inside the basin of a random-fold gauge (fold model) | 31684 |
| outside every gauge basin (all-gauge model) | 25588 |
| inside the basin of a gauge in no fold (all-gauge model) | 1562 |

| scope | area_km2 | reaches | M0_scored | M0_median_nse | M0_nse_q25 | M0_nse_q75 | M0_median_kge | M0_share_nse_above_0 | M1_median_nse | M2_median_nse | M4_median_nse | LSTM_scored | LSTM_median_nse | LSTM_nse_q25 | LSTM_nse_q75 | LSTM_median_kge | LSTM_share_nse_above_0 | NWM_scored | NWM_median_nse | NWM_nse_q25 | NWM_nse_q75 | NWM_median_kge | NWM_share_nse_above_0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all reaches | < 5 | 27598 | 25212 | 0.011 | -0.072 | 0.085 | -0.147 | 0.548 | MISSING | MISSING | MISSING | 25212 | -0.723 | -6.170 | -0.003 | -0.304 | 0.248 | 25212 | -1.730 | -6.889 | -0.412 |  | 0.066 |
| all reaches | 5-50 | 18105 | 16464 | 0.112 | 0.030 | 0.220 | 0.115 | 0.808 | MISSING | MISSING | MISSING | 16464 | 0.050 | -0.256 | 0.176 | 0.163 | 0.584 | 16464 | -0.696 | -2.144 | -0.160 |  | 0.144 |
| all reaches | 50-500 | 7868 | 6901 | 0.211 | 0.059 | 0.359 | 0.303 | 0.807 | MISSING | MISSING | MISSING | 6901 | 0.197 | 0.038 | 0.329 | 0.281 | 0.784 | 6901 | -0.150 | -0.691 | 0.136 |  | 0.386 |
| all reaches | 500-5,000 | 3694 | 3020 | 0.306 | -0.122 | 0.449 | 0.461 | 0.724 | MISSING | MISSING | MISSING | 3020 | 0.249 | -0.333 | 0.408 | 0.406 | 0.717 | 3020 | 0.211 | -0.205 | 0.446 |  | 0.670 |
| all reaches | > 5,000 | 1569 | 529 | -0.290 | -3.967 | 0.418 | 0.258 | 0.380 | MISSING | MISSING | MISSING | 529 | -1.449 | -3.597 | -0.221 | 0.070 | 0.231 | 529 | -0.238 | -1.624 | 0.355 |  | 0.393 |
| inside the basin of a withheld gauge (reach_fold > 0) | < 5 | 15006 | 14201 | 0.017 | -0.089 | 0.098 | -0.097 | 0.567 | MISSING | MISSING | MISSING | 14201 | -0.860 | -6.716 | -0.037 | -0.312 | 0.221 | 14201 | -1.492 | -5.764 | -0.391 |  | 0.061 |
| inside the basin of a withheld gauge (reach_fold > 0) | 5-50 | 9599 | 9192 | 0.122 | 0.038 | 0.226 | 0.140 | 0.829 | MISSING | MISSING | MISSING | 9192 | 0.042 | -0.242 | 0.159 | 0.170 | 0.572 | 9192 | -0.615 | -1.716 | -0.143 |  | 0.139 |
| inside the basin of a withheld gauge (reach_fold > 0) | 50-500 | 4223 | 3940 | 0.239 | 0.095 | 0.361 | 0.320 | 0.865 | MISSING | MISSING | MISSING | 3940 | 0.196 | 0.042 | 0.320 | 0.301 | 0.795 | 3940 | -0.098 | -0.569 | 0.142 |  | 0.405 |
| inside the basin of a withheld gauge (reach_fold > 0) | 500-5,000 | 2575 | 2245 | 0.319 | 0.006 | 0.466 | 0.476 | 0.754 | MISSING | MISSING | MISSING | 2245 | 0.267 | -0.365 | 0.417 | 0.413 | 0.706 | 2245 | 0.230 | -0.172 | 0.455 |  | 0.682 |
| inside the basin of a withheld gauge (reach_fold > 0) | > 5,000 | 281 | 168 | -0.345 | -2.258 | 0.198 | 0.207 | 0.280 | MISSING | MISSING | MISSING | 168 | -2.705 | -4.800 | -0.882 | -0.308 | 0.054 | 168 | -1.594 | -2.441 | 0.211 |  | 0.298 |
| outside (reach_fold <= 0) | < 5 | 12592 | 11011 | 0.005 | -0.059 | 0.071 | -0.217 | 0.524 | MISSING | MISSING | MISSING | 11011 | -0.566 | -5.500 | 0.025 | -0.295 | 0.282 | 11011 | -2.061 | -8.354 | -0.459 |  | 0.073 |
| outside (reach_fold <= 0) | 5-50 | 8506 | 7272 | 0.100 | 0.017 | 0.211 | 0.080 | 0.781 | MISSING | MISSING | MISSING | 7272 | 0.060 | -0.277 | 0.205 | 0.153 | 0.600 | 7272 | -0.879 | -2.744 | -0.182 |  | 0.151 |
| outside (reach_fold <= 0) | 50-500 | 3645 | 2961 | 0.167 | -0.038 | 0.352 | 0.270 | 0.730 | MISSING | MISSING | MISSING | 2961 | 0.201 | 0.027 | 0.337 | 0.240 | 0.770 | 2961 | -0.233 | -1.012 | 0.131 |  | 0.361 |
| outside (reach_fold <= 0) | 500-5,000 | 1119 | 775 | 0.228 | -0.409 | 0.381 | 0.246 | 0.637 | MISSING | MISSING | MISSING | 775 | 0.229 | -0.005 | 0.383 | 0.345 | 0.747 | 775 | 0.138 | -0.242 | 0.415 |  | 0.634 |
| outside (reach_fold <= 0) | > 5,000 | 1288 | 361 | -0.182 | -3.967 | 0.437 | 0.272 | 0.427 | MISSING | MISSING | MISSING | 361 | -0.624 | -3.597 | 0.278 | 0.176 | 0.313 | 361 | -0.126 | -0.566 | 0.436 |  | 0.438 |


Headwater reaches (CSV `s5_7_headwater_summary.csv`; no reach upstream of the reach in the network graph): per-reach days 1-5 NSE against the NWM retrospective; NWM = the NWM operational forecast scored against the retrospective. Figures S6 a-e; lead curves `s5_headwater_lead_curves.csv`.

| reaches | n | median_area_km2 | M0_scored | M0_median_nse | M0_share_nse_above_0 | LSTM_scored | LSTM_median_nse | LSTM_share_nse_above_0 | NWM_scored | NWM_median_nse | NWM_share_nse_above_0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| headwater reaches (no reach upstream) | 23172 | 1.960 | 22124 | 0.014 | 0.557 | 22124 | -0.650 | 0.265 | 22124 | -1.643 | 0.069 |
| all other reaches | 35662 | 21.644 | 30002 | 0.118 | 0.759 | 30002 | 0.065 | 0.590 | 30002 | -0.477 | 0.247 |
| all reaches | 58834 | 5.826 | 52126 | 0.062 | 0.673 | 52126 | -0.064 | 0.452 | 52126 | -0.800 | 0.172 |


### 8. Example event (CSV `s5_8_example_event.csv`)

Rule: random hold-out; among the nested gauges with a visible upstream gauge, the gauge with the (lower) median M0 - M1 days 1-5 gain; within the test period (2020-09-23 to 2025-12-31) its largest observed six-hour flow; 5 days either side.

Gauge 03513000 (USGS 03513000), 1,695.6 km2, fold 2, gain -0.0003 (rank 20 of 40); visible upstream gauges: 03508050 (03508050), 03509000 (03509000). Peak 920.7 m3/s at 2024-09-27 18:00:00 (six-hour mean). The CSV gives, for leads 6, 12, 18 and 24 h, the issue, the valid time and the observed, NWM, LSTM, M0 and M1 values (NaN where an issue has no forecast or is not scored).

### 9. Numbers for the caveats

Test pairs per scored gauge at 24 h (CSV `s5_9_test_pairs.csv`):

| protocol | gauges | min_pairs_24h | gauges_under_300_pairs | gauges_under_500_pairs |
|---|---|---|---|---|
| Gauged | 101 | 355 | 0 | 1 |
| Random hold-out | 98 | 355 | 0 | 1 |
| Blocked basins | 29 | 1459 | 0 | 0 |

Gauges where the NWM has the higher days 1-5 NSE than M0 (CSV `s5_9_nwm_better.csv`):

| protocol | n | gauges_NWM_better |
|---|---|---|
| Gauged | 101 | 6 |
| Random hold-out | 98 | 17 |
| Blocked basins | 29 | 5 |

Sign of each contrast across protocols (CSV `s5_9_sign_by_protocol.csv`):

| contrast | protocols | signs | flips |
|---|---|---|---|
| M0 - LSTM | Gauged, Random hold-out, Blocked basins | +, +, + | False |
| M0 - NWM | Gauged, Random hold-out, Blocked basins | +, +, + | False |
| M0 - M2 | Gauged, Random hold-out, Blocked basins | +, +, - | True |
| M0 - M1 | Gauged, Random hold-out, Blocked basins | +, +, + | False |
| M1 - LSTM | Gauged, Random hold-out, Blocked basins | +, +, + | False |
| M2 - LSTM | Gauged, Random hold-out, Blocked basins | +, -, + | True |
| M1 - M4 | Random hold-out, Blocked basins | +, - | True |
| M4 - LSTM | Random hold-out, Blocked basins | -, + | True |

Contrasts within the 0.02 NSE equivalence margin (Holm-corrected TOST p < 0.05): M0 - M2 (Gauged); M0 - M1 (Gauged); M0 - M1 (Random hold-out).

Within the 0.01 margin: M0 - M2 (Gauged); M0 - M1 (Gauged).

### 9. Caveats for the text

- One seed (seed 1) per model and protocol; no seed spread is available, so no difference can be compared with training noise.
- Missing runs: M4 Gauged (MISSING (not planned: no all-gauge M4)). Every-reach: M1, M2, M4 MISSING.
- Blocked protocol: n = 29 gauges in 3 basins (12, 11, 6 scored); no scored gauge there has a visible upstream gauge, so M0 - M1 there is not a test of the insertion.
- Test pairs: the fewest at 24 h is 355 (section 9 table); 1 gauge has fewer than 500 in the gauged and random protocols.
- NWM: the operational NWM is nudged toward observations at gauges before each forecast, including gauges withheld from the other models; Gauged: NWM better than M0 at 6 of 101 gauges; Random hold-out: NWM better than M0 at 17 of 98 gauges; Blocked basins: NWM better than M0 at 5 of 29 gauges.
- Sign changes between protocols: M0 - M2 (+, +, - for Gauged, Random hold-out, Blocked basins); M2 - LSTM (+, -, + for Gauged, Random hold-out, Blocked basins); M1 - M4 (+, - for Random hold-out, Blocked basins); M4 - LSTM (-, + for Random hold-out, Blocked basins).
- Sign changes across leads 24-120 h: M0 - LSTM, Random hold-out: 96->120 h; M0 - M1, Random hold-out: 48->72 h; M1 - LSTM, Random hold-out: 96->120 h; M2 - LSTM, Random hold-out: 24->48 h; M1 - M4, Blocked basins: 24->48 h, 48->72 h, 72->96 h, 96->120 h.
- Within the 0.02 NSE margin (Holm-corrected TOST): M0 - M2 (Gauged); M0 - M1 (Gauged); M0 - M1 (Random hold-out). Not shown equivalent at 0.02 (and not different either) is every other contrast whose interval includes zero.
- Noise floor (5e): the blocked M0 - M1 difference, where insertion cannot act, is all blocked gauges +0.012; basin B1 +0.004; basin B2 +0.019; basin B3 +0.018 (one pair of runs). Every statement that a difference is within 0.02 NSE has to be read against it.
- The +-0.02 equivalence margin is not yet tied to a seed spread; it should be justified once second seeds exist (section 10).
- 5f: the area-ratio correlation changes sign between the gauged and random protocols and rests on one seed; exploratory only.
- 5b: upstream gauges are visible at nearly every test issue (median share 1.0), so the within-gauge test has 1-2 gauges per protocol; it cannot separate the insertion from other differences between M0 and M1.
- Every reach: the reference is the NWM retrospective (a model) that M0 (and M1, M2, M4) were pretrained on and the LSTM never saw, and the LSTM at reaches runs with the flow history hidden everywhere. Contrasts within the M family (M0 - M1, M0 - M2 by area class) would be fair; only M0 has a reach-level run, so none can be computed (MISSING); the M0 - LSTM comparison at reaches is biased toward M0.
- Holm correction here runs across the contrasts available in each protocol (6 while M4 is missing); earlier reports used other families, so corrected p values differ from them.

### 10. Extra runs that would make the key contrasts interpretable (not started)

Wall-clock from the logged runs (`<data>/models/attr/*/validation.json` and logs): stage B M0 178 min on 4 GPUs (11.9 GPU-h), M1 167 min (11.1 GPU-h), M2 119 min (8.0 GPU-h), M4 246 min on 2 GPUs (8.2 GPU-h); one gauge-protocol evaluation about 30 min on 4 GPUs; one every-reach evaluation (M0) about 44 min on 4 GPUs; L4 LSTM about 17 min on one GPU plus a CPU benchmark. Four L40S GPUs; runs of different sizes can share the machine.

| Priority | Runs | GPU-hours | Wall-clock on 4 GPUs | What it enables |
|---|---|---|---|---|
| 1 | M4 evaluations (random hold-out, blocked): DONE 2026-11-22 | about 4 | about 30 min (parallel) | M1 - M4 (RQ2) in both protocols |
| 2 | L4 LSTM seeds 2-5, gauged protocol (4 runs, 17 min each) + CPU benchmark evaluation | about 1.2 (GPU) | about 1.5 h plus the CPU benchmark | seed spread of the baseline and of its early-checkpoint selection; seed spread for M0 - LSTM |
| 3 | Second stage-B seed, gauged protocol: M0, M1, M2 (reuse stage A) + evaluation | about 31 | about 7.7 h + 1.5 h | seed spread for the contrasts that carry the equivalence claims (M0 - M1, M0 - M2); a data-based justification of the +-0.02 margin |
| 4 | Every-reach evaluation of M1, M2 and M4 (existing models, no training) | about 12 | about 2.2 h (3 x 44 min) | within-family contrasts by area class that are not biased by the shared NWM reference (Figure 7) |
| 5 | Seed 2, blocked basins: M0, M1, M2, M4 x 3 basins + evaluations | about 120 | about 31 h | seed spread for the blocked contrasts (n = 29), after items 2-3 |
| 6 | Seed 2, random hold-out: M0, M1, M2, M4 x 5 folds + evaluation | about 197 | about 50 h | seed spread for the 98-gauge contrasts, if still needed |

Retraining stage A for a second seed would add 15.9 (M0/M1), 9.1 (M2) and 12.0 (M4) GPU-hours; items 3, 5 and 6 above reuse the existing stage A, so they measure stage-B noise only.

---

# PART C. Figures and tables

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
\begin{table}
\caption{Median NSE and KGE, Days 1--5, by Protocol and Model}
\label{tab:skill}
\footnotesize
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}llrrr@{}}
\toprule
Protocol & Model & $n$ & NSE & KGE \\
\midrule
Gauged & GNN & 101 & \textbf{0.430} & \textbf{0.526} \\
 & GNN-noIns & 101 & 0.428 & \textbf{0.526} \\
 & GNN-noMP & 101 & 0.426 & 0.517 \\
 & GNN-Instat & --- & \multicolumn{2}{c}{not trained} \\
 & LSTM & 101 & 0.428 & 0.481 \\
 & NWM & 101 & 0.255 & 0.443 \\
\addlinespace
Random hold-out & GNN & 98 & 0.385 & 0.429 \\
 & GNN-noIns & 98 & 0.383 & 0.428 \\
 & GNN-noMP & 98 & \textbf{0.386} & 0.421 \\
 & GNN-Instat & 98 & 0.379 & 0.421 \\
 & LSTM & 98 & 0.367 & 0.396 \\
 & NWM & 98 & 0.255 & \textbf{0.448} \\
\addlinespace
Blocked basins & GNN & 29 & 0.305 & 0.334 \\
 & GNN-noIns & 29 & 0.297 & 0.334 \\
 & GNN-noMP & 29 & \textbf{0.322} & 0.341 \\
 & GNN-Instat & 29 & 0.307 & 0.324 \\
 & LSTM & 29 & 0.280 & 0.381 \\
 & NWM & 29 & 0.183 & \textbf{0.385} \\
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize \emph{Note.} Median over the $n$ gauges scored by every model of the protocol, per lead, averaged over the leads 24, 48, 72, 96 and 120~h. Bold: the highest value per protocol. Differences below 0.02 NSE are not distinguishable (equivalence margin, Table~S6). GNN-Instat was not trained for the gauged protocol. NWM: operational forecast, nudged toward observations at gauges, a reference only.
\end{flushleft}
\end{table}
```

## Table S6 (LaTeX, ready to paste)

```latex
\begin{table}
\caption{Paired Differences in Days 1--5 NSE}
\label{tab:contrasts}
\footnotesize
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}llrrrr@{}}
\toprule
Protocol & Contrast & $n$ & Difference & 5--95\% interval & Holm $p$ \\
\midrule
Gauged & GNN $-$ LSTM & 101 & +0.020 & [+0.007, +0.035] & 0.053 \\
 & GNN $-$ NWM & 101 & +0.303 & [+0.226, +0.398] & $<$0.001 \\
 & GNN $-$ GNN-noMP & 101 & +0.005 & [+0.002, +0.009] & 0.051 \\
 & GNN-noIns $-$ GNN-Instat & --- & \multicolumn{3}{c}{GNN-Instat not trained} \\
 & GNN $-$ GNN-noIns & 101 & +0.002 & [+0.001, +0.004] & 0.003 \\
 & GNN-noIns $-$ LSTM & 101 & +0.017 & [+0.004, +0.032] & 0.082 \\
 & GNN-noMP $-$ LSTM & 101 & +0.015 & [+0.002, +0.028] & 0.083 \\
 & GNN-Instat $-$ LSTM & --- & \multicolumn{3}{c}{GNN-Instat not trained} \\
\addlinespace
Random hold-out & GNN $-$ LSTM & 98 & +0.015 & [-0.010, +0.037] & 0.012 \\
 & GNN $-$ NWM & 98 & +0.207 & [+0.125, +0.298] & $<$0.001 \\
 & GNN $-$ GNN-noMP & 98 & +0.023 & [-0.002, +0.054] & 0.986 \\
 & GNN-noIns $-$ GNN-Instat & 98 & +0.033 & [+0.016, +0.053] & $<$0.001 \\
 & GNN $-$ GNN-noIns & 98 & +0.000 & [-0.007, +0.007] & 0.986 \\
 & GNN-noIns $-$ LSTM & 98 & +0.015 & [-0.014, +0.040] & 0.012 \\
 & GNN-noMP $-$ LSTM & 98 & -0.009 & [-0.052, +0.027] & 0.015 \\
 & GNN-Instat $-$ LSTM & 98 & -0.018 & [-0.059, +0.016] & 0.274 \\
\addlinespace
Blocked basins & GNN $-$ LSTM & 29 & +0.066 & [+0.026, +0.105] & 0.089 \\
 & GNN $-$ NWM & 29 & +0.390 & [+0.186, +0.643] & $<$0.001 \\
 & GNN $-$ GNN-noMP & 29 & -0.013 & [-0.038, +0.010] & 1.000 \\
 & GNN-noIns $-$ GNN-Instat & 29 & -0.006 & [-0.037, +0.021] & 1.000 \\
 & GNN $-$ GNN-noIns & 29 & +0.012 & [+0.002, +0.026] & 1.000 \\
 & GNN-noIns $-$ LSTM & 29 & +0.053 & [+0.015, +0.090] & 0.146 \\
 & GNN-noMP $-$ LSTM & 29 & +0.079 & [+0.042, +0.116] & 0.010 \\
 & GNN-Instat $-$ LSTM & 29 & +0.059 & [+0.020, +0.100] & 0.083 \\
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize \emph{Note.} Difference = first minus second model, mean over the $n$ gauges both score of the per-gauge mean NSE over leads 24--120~h. Interval: gauge bootstrap (10,000 resamples). Holm $p$: Wilcoxon signed-rank test, Holm-corrected across the contrasts of each protocol.
\end{flushleft}
\end{table}
```

---


---

# PART D. Open items (must be resolved or stated in the text)

1. **M4 and routing (RQ2).** M4 is evaluated for the random hold-out and the blocked basins (done 2026-11-22); there is no gauged-protocol M4, so M1 - M4 exists in those two protocols only.
2. **Seeds.** One seed per model and protocol. Planned in order of cost: LSTM seeds 2-5 (about 1.5 h plus the CPU benchmark); a second stage-B seed for gauged M0, M1, M2 (about 9 h); then blocked seed 2 (about 31 h) and random hold-out seed 2 (about 50 h). The +-0.02 equivalence margin should be justified from the seed spread once these exist; until then say that it is a pre-chosen margin and give the noise floor (PART B 5e).
3. **Every-reach contrasts within the model family** (M0 - M1, M0 - M2 by area class) need reach-level runs of M1, M2, M4 (existing models, about 2.2 h of evaluation); until then Figure 8 and Figures S5-S6 show M0, the LSTM and the NWM forecast only.
4. **Figures** are final in layout (journal style); Figure 5 already groups the rows by research question and leaves M0 - NWM to Table 3 and Table S6.
5. **Paper housekeeping:** (a) Introduction, RQ1 says "a lumped LSTM with matched inputs and outputs", which contradicts Section 4.4: reword as in A3. (b) The paper's Figure 3 is the model schematic (file `fig4_model`); rename it `fig3_model` so the results figures are 4-10. (c) Section 4.5 contains a VERIFY comment on M4 coverage: adjust the sentence and Table 2 to the runs that exist: M4 in the random hold-out and blocked protocols only (inventory in PART B, section 1).
6. **Numbers not in this file** (e.g. any comparison not in PART B): say they are not available; do not compute or estimate them.
