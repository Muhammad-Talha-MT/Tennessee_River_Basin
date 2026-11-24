# Test-set comparison v3: M0, M1, M2, LSTM and NWM

Test issues 2020-09-23 to 2025-12-22 (1,917 00Z issues), GEFSv12 operational rain to 240 h, 6-hourly leads. Protocols: **gauged** (101 gauges, model trained on all gauges), **held-out** (98 gauges, 5 random folds, each scored at gauges it never saw), **blocked** (29 gauges in 3 whole tributary basins withheld). Metric: median NSE over gauges; *days 1-5* is the mean over the leads 24, 48, 72, 96 and 120 h of that median. One seed for every model.

| Model | What it is |
|---|---|
| M0 | Manning routing + gauge insertion + 3 graph layers |
| M1 | M0 without insertion |
| M2 | M0 without graph layers |
| LSTM | area-scaled LSTM (output = specific discharge x basin area), the only baseline |
| NWM | National Water Model v3.0 forecast, reference only |

**NWM is a reference, not a held-out competitor.** The NWM analysis nudges toward USGS observations at every gauge, including the gauges withheld from M0, M1, M2 and the LSTM, so its scores at held-out and blocked gauges are not comparable to a model that never saw those gauges.

*Not run (marked in the tables):* M4 = M1 without routing (planned: held-out and blocked only); the reach level for M1, M2 and M4; multiple seeds.

## 1. Headline: days 1-5 median NSE

| Protocol (gauges) | M0 | M1 | M2 | LSTM | NWM |
|---|---:|---:|---:|---:|---:|
| Gauged (101) | **0.430** | 0.428 | 0.426 | 0.428 | 0.255 |
| Held-out (random folds) (98) | 0.385 | 0.383 | **0.386** | 0.367 | 0.255 |
| Blocked basins (29) | 0.305 | 0.297 | **0.322** | 0.280 | 0.183 |

Bold: best of M0, M1, M2 and LSTM.

## 2. NSE by lead

Gauged (101 gauges):

| Model | 6 h | 24 h | 72 h | 120 h | 240 h | Days 1-5 |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.919 | 0.614 | 0.401 | 0.307 | 0.153 | 0.430 |
| M1 | 0.916 | 0.612 | 0.402 | 0.310 | 0.151 | 0.428 |
| M2 | 0.918 | 0.608 | 0.400 | 0.306 | 0.149 | 0.426 |
| LSTM | 0.804 | 0.595 | 0.403 | 0.308 | 0.145 | 0.428 |
| NWM | 0.837 | 0.461 | 0.244 | 0.079 | -0.331 | 0.255 |

Held-out (random folds) (98 gauges):

| Model | 6 h | 24 h | 72 h | 120 h | 240 h | Days 1-5 |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.770 | 0.567 | 0.372 | 0.248 | 0.115 | 0.385 |
| M1 | 0.769 | 0.569 | 0.369 | 0.235 | 0.115 | 0.383 |
| M2 | 0.754 | 0.567 | 0.366 | 0.253 | 0.111 | 0.386 |
| LSTM | 0.690 | 0.518 | 0.361 | 0.236 | 0.099 | 0.367 |
| NWM | 0.842 | 0.463 | 0.245 | 0.070 | -0.331 | 0.255 |

Blocked basins (29 gauges):

| Model | 6 h | 24 h | 72 h | 120 h | 240 h | Days 1-5 |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.710 | 0.457 | 0.275 | 0.169 | 0.045 | 0.305 |
| M1 | 0.718 | 0.462 | 0.272 | 0.159 | 0.041 | 0.297 |
| M2 | 0.688 | 0.463 | 0.305 | 0.201 | 0.041 | 0.322 |
| LSTM | 0.719 | 0.482 | 0.242 | 0.126 | 0.011 | 0.280 |
| NWM | 0.773 | 0.401 | 0.083 | 0.031 | -0.187 | 0.183 |

## 3. Paired tests

Per-gauge days 1-5 NSE (mean over the leads 24-120 h), first model minus second. Gauge-level 90% bootstrap interval (10,000 resamples) of the mean difference, gauges where the first model is better, two-sided Wilcoxon signed-rank p, and Holm-adjusted p across all six paired tests of the protocol's table.

| Comparison | Gauged | Held-out | Blocked |
|---|---|---|---|
| M0 - LSTM | +0.020 [+0.007, +0.035]; 60/101; p 0.018, Holm 0.071 | +0.015 [-0.010, +0.037]; 65/98; p 0.002, Holm 0.012 | +0.066 [+0.026, +0.105]; 20/29; p 0.018, Holm 0.089 |
| M2 - LSTM | +0.015 [+0.002, +0.028]; 57/101; p 0.083, Holm 0.165 | -0.009 [-0.052, +0.027]; 67/98; p 0.004, Holm 0.015 | +0.079 [+0.042, +0.116]; 22/29; p 0.001, Holm 0.008 |
| M1 - LSTM | +0.017 [+0.004, +0.032]; 59/101; p 0.041, Holm 0.123 | +0.015 [-0.014, +0.040]; 68/98; p 0.002, Holm 0.012 | +0.053 [+0.015, +0.090]; 19/29; p 0.037, Holm 0.146 |
| M0 - M2 | +0.005 [+0.002, +0.009]; 59/101; p 0.013, Holm 0.064 | +0.023 [-0.002, +0.054]; 50/98; p 0.493, Holm 0.986 | -0.013 [-0.038, +0.010]; 14/29; p 0.579, Holm 1.000 |
| M0 - M1 | +0.002 [+0.001, +0.004]; 67/101; p < 0.001, Holm 0.004 | +0.000 [-0.007, +0.007]; 46/98; p 0.906, Holm 0.986 | +0.012 [+0.002, +0.026]; 13/29; p 0.369, Holm 1.000 |
| M1 - M2 | +0.003 [-0.001, +0.006]; 62/101; p 0.119, Holm 0.165 | +0.023 [+0.000, +0.050]; 50/98; p 0.125, Holm 0.376 | -0.026 [-0.056, +0.002]; 13/29; p 0.670, Holm 1.000 |

Note: M2 - LSTM on held-out (random folds) has a mean difference of -0.009 but a median difference of +0.020 and is better on 67 of 98 gauges: a few gauges with large losses pull the mean below zero while most gauges gain, so the Wilcoxon result and the mean point in different directions.

M0 against LSTM is not significant after Holm correction on gauged (Holm p 0.071), significant after Holm correction on held-out (Holm p 0.012) and not significant after Holm correction on blocked (Holm p 0.089); the blocked test treats the 29 gauges as independent, but they sit in only 3 basins (section 3.2).

### 3.1 Equivalence tests for the null claims

Paired TOST on the mean difference of per-gauge days 1-5 NSE with equivalence margins of +-0.01 and +-0.02 NSE (two one-sided t-tests; the larger p is reported), Holm-adjusted across the three equivalence tests of a protocol at each margin. Equivalence is concluded when the adjusted p is below 0.05. The 90% intervals are t-based and bootstrap.

| Pair | Protocol | Mean diff | 90% CI (t) / (bootstrap) | TOST p, +-0.01 (Holm) | Equivalent +-0.01 | TOST p, +-0.02 (Holm) | Equivalent +-0.02 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 - M2 | Gauged | +0.005 | [+0.002, +0.008] / [+0.002, +0.009] | 0.009 | yes | < 0.001 | yes |
| M0 - M2 | Held-out (random folds) | +0.023 | [-0.005, +0.052] / [-0.002, +0.054] | 1.000 | no | 1.000 | no |
| M0 - M2 | Blocked basins | -0.013 | [-0.039, +0.012] / [-0.038, +0.010] | 1.000 | no | 0.659 | no |
| M0 - M1 | Gauged | +0.002 | [+0.001, +0.004] / [+0.001, +0.004] | < 0.001 | yes | < 0.001 | yes |
| M0 - M1 | Held-out (random folds) | +0.000 | [-0.007, +0.007] / [-0.007, +0.007] | 0.034 | yes | < 0.001 | yes |
| M0 - M1 | Blocked basins | +0.012 | [-0.000, +0.025] / [+0.002, +0.026] | 1.000 | no | 0.478 | no |
| M1 - M2 | Gauged | +0.003 | [-0.001, +0.006] / [-0.001, +0.006] | < 0.001 | yes | < 0.001 | yes |
| M1 - M2 | Held-out (random folds) | +0.023 | [-0.002, +0.049] / [+0.000, +0.050] | 1.000 | no | 1.000 | no |
| M1 - M2 | Blocked basins | -0.026 | [-0.056, +0.005] / [-0.056, +0.002] | 1.000 | no | 0.659 | no |

Equivalence within +-0.01 is established for: M0 vs M2 (gauged), M0 vs M1 (gauged), M1 vs M2 (gauged), M0 vs M1 (held-out (random folds)). Equivalence is **not** established, even at +-0.02, for: M0 vs M2 (held-out (random folds)), M1 vs M2 (held-out (random folds)), M0 vs M2 (blocked basins), M0 vs M1 (blocked basins), M1 vs M2 (blocked basins); their intervals reach beyond the margin, so for those pairs the claim can only be stated as *no detectable effect*.

### 3.2 Blocked basins, per basin

| Basin | Gauges | Median area km2 | M0 | M2 | LSTM | NWM | M0 - LSTM mean (gauges better) | M2 - LSTM mean (gauges better) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 375 | 0.371 | 0.363 | 0.270 | 0.248 | +0.128 (12/12) | +0.127 (12/12) |
| 2 | 11 | 340 | 0.271 | 0.346 | 0.346 | 0.211 | -0.023 (4/11) | +0.037 (7/11) |
| 3 | 6 | 231 | 0.297 | 0.271 | 0.282 | -0.058 | +0.103 (4/6) | +0.059 (3/6) |

Median days 1-5 NSE: M0 is above LSTM in 2 of 3 basins and M2 in 1 of 3. With three basins no cluster-level inference is possible; the gauge-level tests in section 3 treat nested gauges of one basin as independent and should be read as descriptive. NWM is shown as a reference.

## 4. Gap to the LSTM by lead

Difference in median NSE (model minus LSTM) with a 95% gauge bootstrap interval (4,000 resamples); data for figure (a) in `lead_curves.csv`.

| Protocol | Lead | M0 - LSTM | M2 - LSTM |
|---|---:|---:|---:|
| Gauged | 6 h | +0.114 [+0.084, +0.147] | +0.114 [+0.084, +0.143] |
| Gauged | 24 h | +0.019 [-0.024, +0.053] | +0.013 [-0.035, +0.054] |
| Gauged | 72 h | -0.002 [-0.017, +0.033] | -0.003 [-0.021, +0.032] |
| Gauged | 120 h | -0.001 [-0.023, +0.034] | -0.002 [-0.022, +0.015] |
| Gauged | 240 h | +0.008 [-0.009, +0.016] | +0.004 [-0.014, +0.012] |
| Held-out (random folds) | 6 h | +0.080 [+0.036, +0.128] | +0.064 [+0.027, +0.117] |
| Held-out (random folds) | 24 h | +0.049 [+0.006, +0.105] | +0.048 [+0.002, +0.092] |
| Held-out (random folds) | 72 h | +0.012 [-0.018, +0.051] | +0.005 [-0.025, +0.045] |
| Held-out (random folds) | 120 h | +0.012 [-0.026, +0.046] | +0.017 [-0.017, +0.039] |
| Held-out (random folds) | 240 h | +0.016 [-0.012, +0.038] | +0.012 [-0.016, +0.042] |
| Blocked basins | 6 h | -0.009 [-0.137, +0.093] | -0.032 [-0.131, +0.054] |
| Blocked basins | 24 h | -0.024 [-0.097, +0.114] | -0.018 [-0.088, +0.087] |
| Blocked basins | 72 h | +0.033 [-0.016, +0.122] | +0.063 [+0.009, +0.110] |
| Blocked basins | 120 h | +0.042 [+0.002, +0.084] | +0.074 [+0.005, +0.108] |
| Blocked basins | 240 h | +0.035 [-0.029, +0.056] | +0.030 [-0.037, +0.045] |

- Gauged: mean M0 - LSTM difference in median NSE is +0.028 over leads up to 48 h and +0.009 over leads of 144 h and beyond.
- Held-out (random folds): mean M0 - LSTM difference in median NSE is +0.050 over leads up to 48 h and +0.016 over leads of 144 h and beyond.
- Blocked basins: mean M0 - LSTM difference in median NSE is +0.007 over leads up to 48 h and +0.044 over leads of 144 h and beyond.

### 4.1 Status of H1 to H3

**H1** (a reach-scale routed model transfers better than a lumped LSTM, with a gap growing from gauged to held-out to blocked and with lead). Gap in days 1-5 median NSE, M0 minus LSTM: gauged +0.001, held-out +0.017, blocked +0.025; M2 minus LSTM: -0.003, +0.018, +0.043. The ordering of the medians follows the prediction; the per-gauge mean differences do not (M0 - LSTM +0.020, +0.015, +0.066). With lead the M0 - LSTM gap shrinks on gauged, shrinks on held-out (random folds), grows on blocked basins (leads up to 48 h against 144 h and beyond, section 4). On blocked basins M0 is above the LSTM in 2 of 3 basins and M2 in 1 of 3 (section 3.2). The comparison is also not matched in every input (section 9): the network models see other gauges and have NWM pretraining.

**H2** (any network effect comes through physical routing, not learned message passing). Message passing, M0 against M2: +0.005 gauged, +0.023 held-out, -0.013 blocked (section 3.1): no detectable benefit of the graph layers; equivalence within +-0.01 holds on gauged only. The routing test, M1 against M4 (M1 without routing: instantaneous summation of upstream lateral inflow), has not been run yet.

**H3** (observations at upstream gauges, carried downstream by the routing, add skill at ungauged reaches below them). Tested at the 40 held-out gauges that have a visible gauge upstream (the blocked protocol cannot test it, section 8). With M0's own weights, switching insertion off at test time lowers 6 h NSE by 0.057 on average (p < 0.001), so insertion adds skill at short leads. At 24 h the change is -0.008 (p 0.646) and for days 1-5 +0.003 (p 0.581): no detectable benefit beyond the first hours. The trained comparison agrees: M0 minus M1 is +0.000 on all held-out gauges for days 1-5, and -0.003 at these 40 gauges. So H3 is supported at 6 h and not for days 1-5.

## 5. KGE and its components

KGE is the variant of Gupta et al. (2009): KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2) with alpha = std(sim) / std(obs) and beta = mean(sim) / mean(obs) (the 2012 variant, which uses coefficient-of-variation ratios, is not used). The repository's reach metrics (`evaluation/hydrology.py`) use the same definition. Validation against an independent numpy implementation (np.corrcoef, np.std, np.mean): 101 gauges x 5 leads x 2 models, maximum absolute difference 1.8e-14. `hydroeval` is not installed. Values are the median over gauges, averaged over leads 24-120 h.

Gauged (101 gauges), days 1-5:

| Model | KGE | r | alpha (variability) | beta (bias) | NSE |
|---|---:|---:|---:|---:|---:|
| M0 | 0.526 | 0.664 | 0.677 | 0.983 | 0.430 |
| M1 | 0.526 | 0.662 | 0.679 | 0.983 | 0.428 |
| M2 | 0.517 | 0.659 | 0.662 | 0.975 | 0.426 |
| LSTM | 0.481 | 0.670 | 0.632 | 0.935 | 0.428 |
| NWM | 0.443 | 0.578 | 0.772 | 0.861 | 0.255 |

Held-out (random folds) (98 gauges), days 1-5:

| Model | KGE | r | alpha (variability) | beta (bias) | NSE |
|---|---:|---:|---:|---:|---:|
| M0 | 0.429 | 0.661 | 0.617 | 0.933 | 0.385 |
| M1 | 0.428 | 0.663 | 0.618 | 0.944 | 0.383 |
| M2 | 0.421 | 0.657 | 0.628 | 0.954 | 0.386 |
| LSTM | 0.396 | 0.634 | 0.578 | 0.918 | 0.367 |
| NWM | 0.448 | 0.579 | 0.778 | 0.865 | 0.255 |

Blocked basins (29 gauges), days 1-5:

| Model | KGE | r | alpha (variability) | beta (bias) | NSE |
|---|---:|---:|---:|---:|---:|
| M0 | 0.334 | 0.634 | 0.530 | 0.893 | 0.305 |
| M1 | 0.334 | 0.633 | 0.550 | 0.912 | 0.297 |
| M2 | 0.341 | 0.645 | 0.486 | 0.885 | 0.322 |
| LSTM | 0.381 | 0.575 | 0.626 | 0.880 | 0.280 |
| NWM | 0.385 | 0.536 | 0.744 | 0.855 | 0.183 |

KGE by lead:

| Protocol | Model | 6 h | 24 h | 72 h | 120 h | 240 h | Days 1-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gauged | M0 | 0.926 | 0.702 | 0.512 | 0.396 | 0.150 | 0.526 |
| Gauged | M1 | 0.919 | 0.696 | 0.521 | 0.398 | 0.156 | 0.526 |
| Gauged | M2 | 0.923 | 0.703 | 0.505 | 0.382 | 0.150 | 0.517 |
| Gauged | LSTM | 0.695 | 0.551 | 0.505 | 0.379 | 0.157 | 0.481 |
| Gauged | NWM | 0.820 | 0.568 | 0.433 | 0.362 | 0.186 | 0.443 |
| Held-out (random folds) | M0 | 0.718 | 0.591 | 0.413 | 0.317 | 0.086 | 0.429 |
| Held-out (random folds) | M1 | 0.681 | 0.592 | 0.410 | 0.308 | 0.094 | 0.428 |
| Held-out (random folds) | M2 | 0.731 | 0.555 | 0.437 | 0.295 | 0.107 | 0.421 |
| Held-out (random folds) | LSTM | 0.602 | 0.482 | 0.414 | 0.308 | 0.099 | 0.396 |
| Held-out (random folds) | NWM | 0.821 | 0.574 | 0.444 | 0.365 | 0.187 | 0.448 |
| Blocked basins | M0 | 0.566 | 0.504 | 0.311 | 0.190 | 0.025 | 0.334 |
| Blocked basins | M1 | 0.551 | 0.451 | 0.339 | 0.197 | 0.012 | 0.334 |
| Blocked basins | M2 | 0.552 | 0.488 | 0.347 | 0.196 | 0.018 | 0.341 |
| Blocked basins | LSTM | 0.649 | 0.494 | 0.380 | 0.273 | 0.057 | 0.381 |
| Blocked basins | NWM | 0.780 | 0.536 | 0.402 | 0.238 | 0.116 | 0.385 |

On blocked basins KGE ranks the LSTM (0.381) above M2 (0.341) and M0 (0.334) although NSE ranks them the other way. The components show why: the correlation of M0 and M2 (0.634, 0.645) is higher than the LSTM's (0.575), but their variability ratio alpha (0.530, 0.486) is lower than the LSTM's (0.626) and NWM's (0.744); bias ratios are similar (0.893, 0.885, 0.880). The M-family forecasts are better timed but more damped on blocked basins. All models have alpha below 1 in every protocol (flows are under-dispersed), the M-family most on blocked basins.

## 6. High-flow measures

Threshold: the **2-year flow**, the median of each gauge's annual maxima (water years, Oct-Sep) of the 6-hourly observed flow before 2019-01-01, for gauges with at least 10 water years with >= 70% of values observed. Few gauges have enough 2-year events in the test period for a CSI, so CSI at the gauge's **95th percentile** of training-period flow is reported as well. Median over gauges, mean over leads 24-120 h; `n` = gauges with enough events (>= 5 event pairs at the 2-year flow, >= 20 at the 95th percentile).

| Protocol | Model | Peak-flow bias, top 5% flows (sim/obs - 1) | CSI > p95 | n | CSI > 2-year flow | n |
|---|---:|---:|---:|---:|---:|---:|
| Gauged | M0 | -0.373 | 0.397 | 97 | 0.115 | 28 |
| Gauged | M1 | -0.373 | 0.393 | 97 | 0.113 | 28 |
| Gauged | M2 | -0.388 | 0.390 | 97 | 0.110 | 28 |
| Gauged | LSTM | -0.410 | 0.388 | 97 | 0.037 | 28 |
| Gauged | NWM | -0.414 | 0.306 | 97 | 0.135 | 28 |
| Held-out (random folds) | M0 | -0.422 | 0.351 | 97 | 0.113 | 28 |
| Held-out (random folds) | M1 | -0.410 | 0.344 | 97 | 0.114 | 28 |
| Held-out (random folds) | M2 | -0.420 | 0.350 | 97 | 0.088 | 28 |
| Held-out (random folds) | LSTM | -0.452 | 0.321 | 97 | 0.040 | 28 |
| Held-out (random folds) | NWM | -0.412 | 0.306 | 97 | 0.135 | 28 |
| Blocked basins | M0 | -0.420 | 0.330 | 29 | 0.164 | 8 |
| Blocked basins | M1 | -0.403 | 0.343 | 29 | 0.172 | 8 |
| Blocked basins | M2 | -0.479 | 0.281 | 29 | 0.124 | 8 |
| Blocked basins | LSTM | -0.421 | 0.318 | 29 | 0.098 | 8 |
| Blocked basins | NWM | -0.444 | 0.257 | 29 | 0.224 | 8 |

Peak timing and magnitude: for 10-day windows whose observed maximum is at least the 2-year flow, the error in the lead of the forecast maximum (hours, forecast minus observed) and the relative error of the forecast maximum, by the lead of the observed peak; median over gauges with >= 3 such windows.

| Protocol | Model | Peak within 24 h | 24-72 h | 72-120 h | 120-240 h |
|---|---:|---:|---:|---:|---:|
| Gauged | M0 | 3.0 h / -0.51 (n 62) | 6.0 h / -0.73 (n 75) | 6.0 h / -0.80 (n 75) | 24.0 h / -0.88 (n 83) |
| Gauged | M1 | 3.0 h / -0.52 (n 62) | 6.0 h / -0.73 (n 75) | 9.0 h / -0.79 (n 75) | 24.0 h / -0.88 (n 83) |
| Gauged | M2 | 4.5 h / -0.53 (n 62) | 6.0 h / -0.74 (n 75) | 9.0 h / -0.82 (n 75) | 24.0 h / -0.89 (n 83) |
| Gauged | LSTM | 6.0 h / -0.59 (n 62) | 6.0 h / -0.73 (n 75) | 12.0 h / -0.81 (n 75) | 30.0 h / -0.88 (n 83) |
| Gauged | NWM | 6.0 h / -0.40 (n 62) | 15.0 h / -0.65 (n 75) | 24.0 h / -0.77 (n 75) | 36.0 h / -0.86 (n 83) |
| Held-out (random folds) | M0 | 3.0 h / -0.50 (n 62) | 6.0 h / -0.73 (n 75) | 9.0 h / -0.81 (n 75) | 24.0 h / -0.89 (n 83) |
| Held-out (random folds) | M1 | 3.0 h / -0.46 (n 62) | 6.0 h / -0.72 (n 75) | 9.0 h / -0.81 (n 75) | 24.0 h / -0.90 (n 83) |
| Held-out (random folds) | M2 | 3.0 h / -0.52 (n 62) | 6.0 h / -0.75 (n 75) | 12.0 h / -0.82 (n 75) | 24.0 h / -0.89 (n 83) |
| Held-out (random folds) | LSTM | 6.0 h / -0.61 (n 62) | 6.0 h / -0.75 (n 75) | 12.0 h / -0.83 (n 75) | 30.0 h / -0.90 (n 83) |
| Held-out (random folds) | NWM | 6.0 h / -0.40 (n 62) | 15.0 h / -0.65 (n 75) | 24.0 h / -0.77 (n 75) | 36.0 h / -0.86 (n 83) |
| Blocked basins | M0 | 1.5 h / -0.60 (n 18) | 6.0 h / -0.74 (n 22) | 12.0 h / -0.85 (n 22) | 24.0 h / -0.91 (n 24) |
| Blocked basins | M1 | 1.5 h / -0.57 (n 18) | 6.0 h / -0.73 (n 22) | 12.0 h / -0.84 (n 22) | 24.0 h / -0.90 (n 24) |
| Blocked basins | M2 | 4.5 h / -0.65 (n 18) | 6.0 h / -0.77 (n 22) | 12.0 h / -0.86 (n 22) | 24.0 h / -0.91 (n 24) |
| Blocked basins | LSTM | 6.0 h / -0.60 (n 18) | 6.0 h / -0.72 (n 22) | 12.0 h / -0.83 (n 22) | 30.0 h / -0.91 (n 24) |
| Blocked basins | NWM | 6.0 h / -0.39 (n 18) | 18.0 h / -0.70 (n 22) | 25.5 h / -0.79 (n 22) | 40.5 h / -0.89 (n 24) |

Entries are median absolute timing error / median relative peak-magnitude error. All models underestimate flood peaks, increasingly with the lead of the peak.

## 7. Stratification

Area bins (km2) are fixed near the 25th, 60th and 90th percentiles of the gauged-protocol gauge areas (14 to 5,308 km2): < 200, 200-600, 600-2,000, > 2,000.

Gauged, days 1-5 median NSE by drainage area:

| Area km2 | Gauges | M0 | M1 | M2 | LSTM | NWM |
|---|---:|---:|---:|---:|---:|---:|
| < 200 | 27 | 0.304 | **0.304** | 0.300 | 0.302 | 0.138 |
| 200-600 | 34 | 0.416 | 0.416 | 0.427 | **0.428** | 0.203 |
| 600-2,000 | 28 | 0.516 | 0.511 | **0.516** | 0.491 | 0.344 |
| > 2,000 | 12 | **0.559** | 0.551 | 0.556 | 0.489 | 0.446 |

Held-out (random folds), days 1-5 median NSE by drainage area:

| Area km2 | Gauges | M0 | M1 | M2 | LSTM | NWM |
|---|---:|---:|---:|---:|---:|---:|
| < 200 | 26 | **0.281** | 0.276 | 0.265 | 0.254 | 0.146 |
| 200-600 | 32 | 0.378 | 0.371 | **0.389** | 0.370 | 0.192 |
| 600-2,000 | 28 | 0.492 | **0.497** | 0.495 | 0.446 | 0.344 |
| > 2,000 | 12 | 0.477 | **0.530** | 0.475 | 0.453 | 0.446 |

Blocked basins, days 1-5 median NSE by drainage area:

| Area km2 | Gauges | M0 | M1 | M2 | LSTM | NWM |
|---|---:|---:|---:|---:|---:|---:|
| < 200 | 12 | **0.279** | 0.277 | 0.268 | 0.244 | 0.030 |
| 200-600 | 8 | 0.308 | 0.286 | **0.344** | 0.289 | 0.086 |
| 600-2,000 | 6 | 0.334 | 0.341 | **0.396** | 0.307 | 0.295 |
| > 2,000 | 3 | 0.496 | **0.507** | 0.495 | 0.315 | 0.533 |

By whether a gauge visible to the model lies upstream (the condition under which insertion can act):

| Protocol | Visible upstream gauge | Gauges | M0 | M1 | M2 | LSTM | NWM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gauged | yes | 44 | **0.496** | 0.488 | 0.494 | 0.462 | 0.364 |
| Gauged | no | 57 | **0.404** | 0.400 | 0.392 | 0.399 | 0.163 |
| Held-out (random folds) | yes | 40 | 0.447 | **0.463** | 0.440 | 0.416 | 0.361 |
| Held-out (random folds) | no | 58 | 0.353 | 0.353 | **0.365** | 0.329 | 0.177 |
| Blocked basins | yes | 0 | - | - | - | - | - |
| Blocked basins | no | 29 | 0.305 | 0.297 | **0.322** | 0.280 | 0.183 |

On blocked basins no scored gauge has a visible upstream gauge: every gauge upstream of a blocked gauge lies inside the same withheld basin.

Area and nesting together (per-gauge days 1-5 NSE of M0 and M2 minus LSTM regressed on log10(area) and on having any gauge upstream, visible or withheld):

| Protocol | Difference | Term | Coefficient | SE | p | Gauges |
|---|---:|---:|---:|---:|---:|---:|
| Gauged | M0 - LSTM | log10(area km2) | -0.012 | 0.021 | 0.579 | 101 |
| Gauged | M0 - LSTM | any upstream gauge | +0.046 | 0.023 | 0.048 | 101 |
| Gauged | M2 - LSTM | log10(area km2) | -0.014 | 0.019 | 0.453 | 101 |
| Gauged | M2 - LSTM | any upstream gauge | +0.047 | 0.021 | 0.026 | 101 |
| Held-out (random folds) | M0 - LSTM | log10(area km2) | +0.011 | 0.037 | 0.756 | 98 |
| Held-out (random folds) | M0 - LSTM | any upstream gauge | -0.012 | 0.039 | 0.768 | 98 |
| Held-out (random folds) | M2 - LSTM | log10(area km2) | -0.005 | 0.062 | 0.942 | 98 |
| Held-out (random folds) | M2 - LSTM | any upstream gauge | -0.017 | 0.067 | 0.797 | 98 |
| Blocked basins | M0 - LSTM | log10(area km2) | +0.007 | 0.055 | 0.903 | 29 |
| Blocked basins | M0 - LSTM | any upstream gauge | +0.038 | 0.071 | 0.593 | 29 |
| Blocked basins | M2 - LSTM | log10(area km2) | -0.018 | 0.050 | 0.721 | 29 |
| Blocked basins | M2 - LSTM | any upstream gauge | +0.082 | 0.064 | 0.213 | 29 |

On blocked basins the M0 - LSTM advantage is not explained by area (p 0.903) and shows no reliable dependence on having a gauge upstream (p 0.593); with 29 gauges in 3 basins this test has little power. The cross-tabulation by area and nesting is in `strata_area_by_upstream_gauge.csv`.

By dam category (days 1-5 median NSE; the 2-gauge blocked dam category is only in `by_dam_category.csv`):

| Protocol | Category | Gauges | M0 | M1 | M2 | LSTM | NWM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gauged | headwater, no dam | 54 | **0.405** | 0.401 | 0.392 | 0.399 | 0.170 |
| Gauged | below a gauge, no dam | 29 | 0.510 | 0.505 | **0.512** | 0.477 | 0.406 |
| Gauged | dam in the basin | 18 | 0.421 | 0.418 | 0.425 | **0.442** | 0.209 |
| Held-out (random folds) | headwater, no dam | 52 | 0.352 | 0.348 | **0.365** | 0.326 | 0.170 |
| Held-out (random folds) | below a gauge, no dam | 29 | 0.486 | **0.498** | 0.485 | 0.455 | 0.406 |
| Held-out (random folds) | dam in the basin | 17 | 0.371 | **0.383** | 0.343 | 0.378 | 0.221 |
| Blocked basins | headwater, no dam | 18 | **0.295** | 0.286 | 0.277 | 0.257 | 0.043 |
| Blocked basins | below a gauge, no dam | 9 | 0.461 | **0.467** | 0.460 | 0.332 | 0.391 |

## 8. Insertion audit

**What the code does** (`hydro/model.py` `HydroModel.forward` and `insertion`, `hydro/samples.py`, `hydro/evaluate.py`):

1. At test time each fold model scores its own held-out gauges with `visible = Samples.eligible(step, withheld[k])`: not withheld, and at least 8 observed values in the last 32 steps. Gauges of the training folds, and non-CAMELS-H catalog gauges, that lie upstream of a withheld gauge are therefore visible and are inserted.
2. `Samples.gauge_view` sets the withheld gauges' flow to NaN at every time and marks them not visible; `insertion` uses only visible gauges and only the steps before the issue. The withheld gauge's own observation never enters its forecast, nor does that of any other withheld gauge.
3. The corrections are routed down the network by the Manning router (`insert` and `insert_gauges` operations): to every reach and to every gauge *below* the correcting gauge, never to the gauge itself.
4. At a gauge that is visible at the issue the forecast over the lead window is the gauge head's output, which reads the routed flow *before* insertion, so insertion cannot change it. A gauge is visible only if it has at least 8 values in the last 32 steps; a gauge without recent flow is treated like an ungauged reach (routed flow plus corrections from visible gauges above it), also in the gauged protocol. The counterfactual below measures this directly.

**Tests.** `tests/test_hydro.py` now has 23 tests (23 pass; `unit_tests.csv`). The new ones, on a synthetic 5-reach network: an inserted observation changes the downstream forecast and nothing upstream or beside it; a withheld gauge's observation, whatever its value, never changes the insertion output; observed flow after the issue is ignored; the correction is routed with the travel time and conserves mass; `Samples.gauge_view` hides withheld gauges at all times and `eligible` excludes them.

**Check on real checkpoints** (`scripts/v3_leak_check.py`: M0 and M1 fold 1, 2 held-out gauges with visible upstream gauges, 12 gauge-issue cases, fp32). Re-running the same inputs changes the forecast by up to 0.0005 m3/s (floating-point noise from sparse operations). Changing the observed flow after the issue changes it by up to 0.0007 m3/s, and replacing the withheld gauges' true flow by 1e6 by up to 0.0005 m3/s: both at the noise level, so no leakage. Hiding the visible upstream gauges changes the target forecast by 0.02 to 10.69 m3/s, so the models do use upstream observations. (With bf16, as in the evaluation, the noise floor is 0.14 m3/s; `leak_check_bf16.csv`.)

**How many scored gauges have a visible gauge upstream** (`insertion_upstream_visible_summary.csv`; insertion can only act on them):

| Protocol | Scored gauges | With any gauge upstream | With a visible gauge upstream | Mean share of issues with a visible upstream gauge | Median share of basin area upstream of visible gauges |
|---|---:|---:|---:|---:|---:|
| Gauged | 101 | 44 | 44 | 0.39 | 0.54 |
| Held-out (random folds) | 98 | 44 | 40 | 0.37 | 0.51 |
| Blocked basins | 29 | 11 | 0 | 0.00 | n/a |

On the blocked folds **0 of the 29** scored gauges have a visible gauge upstream (11 have gauges upstream, but those lie in the same withheld basin). The blocked protocol therefore cannot test insertion, and H3 can only be tested on the gauged and held-out protocols.

**M0 against M1 at gauged-protocol sites below another gauge** (all gauges visible; `insertion_M0_vs_M1_gauged_sites_below_gauges.csv`). Insertion cannot act here, so any difference reflects training:

| Sites | Gauges | Lead | M0 | M1 | Mean diff | M0 better | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|
| all gauged sites | 101 | 6 h | 0.919 | 0.916 | +0.002 | 66/101 | 0.002 |
| all gauged sites | 101 | 24 h | 0.614 | 0.612 | +0.003 | 61/101 | 0.017 |
| below at least one gauge | 44 | 6 h | 0.947 | 0.943 | +0.001 | 25/44 | 0.066 |
| below at least one gauge | 44 | 24 h | 0.718 | 0.706 | +0.005 | 27/44 | 0.010 |
| upstream gauges cover >= 50% of basin area | 27 | 6 h | 0.926 | 0.918 | +0.003 | 15/27 | 0.055 |
| upstream gauges cover >= 50% of basin area | 27 | 24 h | 0.724 | 0.717 | +0.004 | 17/27 | 0.052 |
| no gauge upstream (headwater) | 57 | 6 h | 0.909 | 0.907 | +0.003 | 41/57 | 0.029 |
| no gauge upstream (headwater) | 57 | 24 h | 0.546 | 0.539 | +0.001 | 34/57 | 0.338 |

**Counterfactual: M0 with insertion switched off at test time, same weights** (`eval_M0_s1_insertoff`, `scripts/v3_insertion_effect.py`; GEFS rain, gauged and held-out protocols).

- Gauged protocol: 99.93% of gauge-issue-lead values are at gauges visible at the issue; insertion changes them by a median of 0.0004 m3/s (99.9th percentile 0.146, maximum 4.98 m3/s), which is not an insertion effect: the fp32 toggle below changes visible gauges by exactly zero, so these differences come from bf16 run-to-run variation between the two evaluation runs. The remaining 0.07% are gauges with no recent flow, treated as hidden; there insertion changes the forecast by up to 25.6 m3/s.
- Same weights, insertion toggled on one real model in fp32 (two test issues, `insertion_toggle_check.csv`): visible gauges: largest change 0.0000 m3/s; hidden, no visible gauge upstream: largest change 0.0000 m3/s; hidden, visible gauge upstream: largest change 13.9697 m3/s.

Held-out gauges. Medians over gauges of the quantity in the third column; mean paired differences with Wilcoxon p:

| Gauges | n | Quantity | M0 | M0, insertion off | M1 | M0 minus M0 off | M0 minus M1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all held-out gauges | 98 | NSE at 6 h | 0.770 | 0.763 | 0.769 | +0.023 (p < 0.001) | +0.021 (p 0.004) |
| all held-out gauges | 98 | NSE at 24 h | 0.567 | 0.568 | 0.569 | -0.003 (p 0.714) | -0.005 (p 0.922) |
| all held-out gauges | 98 | per-gauge days 1-5 NSE | 0.397 | 0.397 | 0.394 | +0.001 (p 0.454) | +0.000 (p 0.906) |
| with a visible gauge upstream | 40 | NSE at 6 h | 0.844 | 0.787 | 0.778 | +0.057 (p < 0.001) | +0.054 (p < 0.001) |
| with a visible gauge upstream | 40 | NSE at 24 h | 0.652 | 0.702 | 0.695 | -0.008 (p 0.646) | -0.011 (p 0.725) |
| with a visible gauge upstream | 40 | per-gauge days 1-5 NSE | 0.460 | 0.496 | 0.499 | +0.003 (p 0.581) | -0.003 (p 0.735) |
| no visible gauge upstream | 58 | NSE at 6 h | 0.751 | 0.751 | 0.760 | +0.000 (p 0.813) | -0.002 (p 0.115) |
| no visible gauge upstream | 58 | NSE at 24 h | 0.525 | 0.525 | 0.521 | -0.000 (p 0.339) | -0.001 (p 0.363) |
| no visible gauge upstream | 58 | per-gauge days 1-5 NSE | 0.360 | 0.360 | 0.357 | -0.000 (p 0.097) | +0.002 (p 0.874) |

Size of the correction: mean absolute change of the forecast caused by insertion, as a share of the gauge's mean flow (median and 90th percentile over gauges):

| Gauges | Lead | n | Median | 90th percentile |
|---|---:|---:|---:|---:|
| with a visible gauge upstream | 6 h | 40 | 10.9% | 32.0% |
| no visible gauge upstream | 6 h | 58 | 0.0% | 0.0% |
| with a visible gauge upstream | 24 h | 40 | 10.3% | 30.8% |
| no visible gauge upstream | 24 h | 58 | 0.0% | 0.0% |
| with a visible gauge upstream | 120 h | 40 | 3.6% | 10.9% |
| no visible gauge upstream | 120 h | 58 | 0.0% | 0.0% |

Rank correlation, across the gauges with a visible gauge upstream, between the share of the basin lying upstream of visible gauges and the NSE gain from insertion: 6: rho +0.24 (p 0.138, n 40); 24: rho -0.36 (p 0.021, n 40); days 1-5: rho -0.45 (p 0.003, n 40).

**Why M0 and M1 differ by only 0.001 in median NSE at 6 h on held-out gauges.** The median over all 98 gauges is M0 0.770 and M1 0.769, but the mean paired difference is +0.021 (p 0.004). Insertion can only act at the 40 gauges that have a visible gauge upstream; at the other 58 M0 with insertion off is identical to M0 (median change 0.0% of mean flow), so they cannot differ because of insertion, and there M1 is slightly better than M0 (median 0.760 against 0.751, training differences). At the 40 gauges insertion matters: median 6 h NSE 0.844 with insertion and 0.787 without (mean gain +0.057, p < 0.001); M1, which has no insertion, sits at 0.778. The median over all gauges is set by the majority that insertion cannot reach, which cancels the gain. The gain does not persist: at 24 h the mean change is -0.008 (p 0.646) and for days 1-5 +0.003 (p 0.581), and the larger the share of the basin covered by upstream gauges the smaller the days 1-5 gain (rho -0.45).

## 9. Inputs audit

What each model receives (`inputs_audit.csv`, from the code constants, checkpoint configs and `arguments.json`). The comparison is matched in forcing, static attributes, antecedent storage, the forecast-rain source and the output units. It is **not** matched in everything: the network models see other gauges' observations and have NWM pretraining and a different loss and training volume.

| Input / setting | M0, M1, M2 | LSTM | Matched? |
|---|---|---|---|
| Weather before the issue | 144 steps (36 days) of 8 AORC variables + 8 validity flags per reach (reach scale) | 120 steps (30 days) of the same 8 variables + 8 validity flags, basin-mean over the gauge's upstream basin | same variables and source; different window length and spatial aggregation |
| Rain after the issue | GEFSv12 control precipitation to 240 h (40 steps) per reach; the other 7 weather variables are unknown and flagged invalid; known-rain flag per step | GEFSv12 control rain + known-share, basin-mean, 40 leads (2 channels); reforecast in training, operational at test | same forecast source and horizon (samples.py:build; lstm_baseline.py:batch) |
| Static attributes | 48 per reach (REACH_INPUTS) | 48 basin-mean (static_dim 55 = 48 + 7) | same attribute set, reach vs basin mean |
| Antecedent storage | 7 per reach (REACH_INPUTS) | 7 basin-mean, appended to the attributes (--antecedent) | same features |
| Season | sin and cos of the day of year (2 step channels) | none (no explicit season input) | differs |
| Own flow of the target gauge when it is withheld | not available: gauge_view sets it to NaN and visible False, at every time (tests/test_hydro.py) | not available: history zeroed and flag 0 (fold models trained with that gauge removed) | matched |
| Own flow of a visible gauge | log flow + shown flag at the gauge's own reach (2 step channels), history only; the gauge head reads the last 120 steps (30 days; HydroModel.forward) | flow + flag in the 120-step history | matched |
| Flow of other gauges | visible gauges only, before the issue: the owner gauge's specific discharge + validity (2 step channels) and the area ratio + flag (2 reach inputs) on every reach inside the owner's basin; carried by the graph layers (M0, M1: 3; M2: 0) and, in M0 and M2 only, by insertion | none | DIFFERS: the network models see other gauges, the LSTM does not (by design) |
| Network structure | river graph with 60,267 links over 58,834 reaches; Manning router inputs per reach (CHANNEL_INPUTS = 53) | none (lumped basin mean) | differs by design |
| NWM | stage A pretraining on NWM retrospective flows at every reach (5000 updates), plus a prior term (nwm_weight 0.3) in stage B; not an input at test | not used (no NWM pretraining, no NWM input) | DIFFERS: the LSTM has no NWM pretraining |
| Output | area-scaled runoff routed down the network; a visible gauge's forecast is the gauge head's; any other reach is routed flow (+ insertion in M0, M2) | specific discharge x basin area / 21.6 (area_output = True) | matched at the gauge (same units: flow in m3/s) |
| Training loss | nse-q: NSE* on gauge and routed forecasts + pinball tail (weight 1.0) + NWM prior (weight 0.3) | nse on gauge flow; high-flow weight 2.0, raw weight 1.0 | differs |
| Gauge visibility in training | 15% of samples show no gauge; otherwise each eligible gauge is shown with probability 0.5 (train.py:107-108) | own history hidden in 57.5% of samples | comparable intent, different mechanism |
| Training volume (stage B) | 3000 updates x 8 whole-network samples, lr 0.0003 | 68,767 updates x 48 (issue, gauge) pairs, lr 0.0005 | differs |
| Model selection | 2019 validation: mean of gauged and hidden median NSE at training gauges, leads 24-120 h | training.selection: mean over 6-120 h of median NSE over camelsh gauges, 00Z validation issues; GEFS reforecast rain | similar (validation year 2019, median NSE over gauges) |
| Recurrent size | LSTM hidden 128 -> 32 channels | LSTM hidden 256 | differs |
| Parameters | M0 932,347; M1 932,347; M2 919,675 | 810,369 | differs |
| Step inputs / reach inputs (network models) | 23 step channels, 57 reach inputs | 18 dynamic channels + 55 static | - |

Settings that differ among M0, M1 and M2 (`config_audit.csv`; the other 35 of 41 recorded settings, including loss, updates, learning rate and router, are identical):

| Setting | M0 | M1 | M2 |
|---|---|---|---|
| config.graph_layers | 3 | 3 | 0 |
| config.insert | True | (absent) | True |
| args.init | /data/tva_data/models/attr/manning_A/best.pt | /data/tva_data/models/attr/manning_A/best.pt | /data/tva_data/models/attr/manning0_A/best.pt |
| args.insert | True | False | True |
| args.graph_layers | 3 | 3 | 0 |
| parameters | 932347 | 932347 | 919675 |

## 10. Reach level: every reach against the NWM v3.0 retrospective

58,834 reaches, 425 issues to 2023-01-20, GEFS rain; median NSE over reaches. The target is the NWM retrospective, not observed flow, so this measures consistency with NWM, not skill. M1 and M2 have not been evaluated at the reach level, and none is planned for M1, M2 or M4.

| Lead | Scope | Reaches | M0 | LSTM | NWM | M1, M2 |
|---|---:|---:|---:|---:|---:|---:|
| 24 h | all reaches | 58,834 | **0.062** | -0.017 | -0.059 | not run |
| 24 h | held-out basins | 31,684 | **0.081** | 0.002 | -0.087 | not run |
| 24 h | outside every gauge basin | 25,588 | **0.042** | -0.054 | -0.032 | not run |
| 24 h | held-out basins, < 50 km2 | 24,605 | **0.054** | -0.086 | -0.174 | not run |
| 24 h | held-out basins, 50-500 | 4,223 | **0.320** | 0.264 | 0.238 | not run |
| 24 h | held-out basins, 500-5,000 | 2,575 | **0.435** | 0.428 | 0.491 | not run |
| 24 h | held-out basins, > 5,000 | 281 | **-0.315** | -3.282 | -0.869 | not run |
| 24 h | behind dams (> 50% of basin) | 2,405 | **-0.123** | -0.508 | -0.054 | not run |
| 24 h | free-flowing, > 500 km2 | 2,286 | **0.481** | 0.435 | 0.628 | not run |
| 72 h | all reaches | 58,834 | **0.024** | -0.118 | -0.192 | not run |
| 72 h | held-out basins | 31,684 | **0.029** | -0.153 | -0.264 | not run |
| 72 h | outside every gauge basin | 25,588 | **0.018** | -0.078 | -0.112 | not run |
| 72 h | held-out basins, < 50 km2 | 24,605 | **0.014** | -0.296 | -0.409 | not run |
| 72 h | held-out basins, 50-500 | 4,223 | **0.171** | 0.116 | 0.040 | not run |
| 72 h | held-out basins, 500-5,000 | 2,575 | **0.270** | 0.214 | 0.331 | not run |
| 72 h | held-out basins, > 5,000 | 281 | **-0.324** | -2.027 | -1.364 | not run |
| 72 h | behind dams (> 50% of basin) | 2,405 | **-0.106** | -0.593 | -0.102 | not run |
| 72 h | free-flowing, > 500 km2 | 2,286 | **0.259** | 0.217 | 0.331 | not run |
| 120 h | all reaches | 58,834 | **0.026** | -0.087 | -1.231 | not run |
| 120 h | held-out basins | 31,684 | **0.031** | -0.107 | -1.180 | not run |
| 120 h | outside every gauge basin | 25,588 | **0.020** | -0.063 | -1.349 | not run |
| 120 h | held-out basins, < 50 km2 | 24,605 | **0.015** | -0.217 | -1.648 | not run |
| 120 h | held-out basins, 50-500 | 4,223 | **0.169** | 0.111 | -0.446 | not run |
| 120 h | held-out basins, 500-5,000 | 2,575 | **0.292** | 0.207 | 0.170 | not run |
| 120 h | held-out basins, > 5,000 | 281 | **-0.340** | -1.792 | -1.781 | not run |
| 120 h | behind dams (> 50% of basin) | 2,405 | **-0.050** | -0.399 | -0.177 | not run |
| 120 h | free-flowing, > 500 km2 | 2,286 | **0.274** | 0.221 | 0.143 | not run |

## 11. Pending

- M4 (M1 without routing) on the held-out and blocked protocols; seed 2 on the blocked folds. Not planned: gauged-protocol runs of new models, reach-level runs of M1, M2 and M4, gauged and held-out seeds, seed 3.

## 12. Figures, tables and scripts

Draft figures in `figures/paper/` (PDF and PNG): `fig_a_skill_vs_lead`, `fig_b_gap_vs_area`, `fig_c_ablation_ladder`, `fig_d_kge_components`, `fig_e_reach_level`.

Every table here is a CSV in `reports/v3_tables/`; the scripts are `scripts/v3_common.py` (loaders, scores, statistics), `v3_insertion_audit.py`, `v3_leak_check.py`, `v3_insertion_effect.py`, `v3_config_audit.py`, `v3_inputs_audit.py`, `v3_kge_highflow.py`, `v3_stats.py`, `v3_figures.py` and this report builder `v3_report.py`; unit tests: `tests/run_all.py`. Commands and jobs are in `RUNLOG.md`.

Inputs: saved forecasts in `models/attr/` (`eval_M0_s1`, `eval_M1_s1`, `eval_M2_s1`, `eval_S1_M0_s1`, `eval_S1_M2_s1`, `eval_M0_s1_insertoff`, `eval_M0_reach`), the LSTM benchmarks (`ladder_L4_s1_bench`, `_bench_f{1..5}`, `_bench_b{1..3}`, `ladder_L4_s1_reaches`), and the NWM rows of the M0 gauged evaluation.
