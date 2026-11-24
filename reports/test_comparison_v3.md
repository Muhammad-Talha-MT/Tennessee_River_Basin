# Working report: results against the research questions

Test issues 2020-09-23 to 2025-12-22 (1,917 issues), GEFSv12 rain, leads 6-240 h. Metric: median NSE over gauges; *days 1-5* = mean over the leads 24-120 h. One seed. Protocols: gauged (101 gauges), held-out (98, 5 random folds), blocked (29, 3 whole basins). **A dash (-) marks a result that is still pending.**

| Model | Definition | Status |
|---|---|---|
| M0 | Manning routing + gauge insertion + 3 graph layers | done |
| M1 | M0 without insertion | done |
| M2 | M0 without graph layers | done |
| M4 | M1 without routing (instantaneous summation of upstream lateral inflow) | stage A done; held-out and blocked runs in progress (-) |
| LSTM | area-scaled LSTM, the only baseline | done |
| NWM | v3.0 forecast, reference only: it nudges toward every USGS gauge, including withheld ones | done |

## 1. Headline: days 1-5 median NSE

| Protocol | M0 | M1 | M2 | M4 | LSTM | NWM |
|---|---:|---:|---:|---:|---:|---:|
| Gauged (101) | 0.430 | 0.428 | 0.426 | - | 0.428 | 0.255 |
| Held-out (98) | 0.385 | 0.383 | 0.386 | - | 0.367 | 0.255 |
| Blocked (29) | 0.305 | 0.297 | 0.322 | - | 0.280 | 0.183 |

M4 has no gauged runs by design.

## 2. RQ1: does a reach-scale routed model transfer better than the LSTM?

Per-gauge days 1-5 NSE, model minus LSTM: mean difference [90% gauge bootstrap]; gauges where the model is better; Holm-adjusted Wilcoxon p (six tests per protocol).

| Comparison | Gauged | Held-out | Blocked |
|---|---:|---:|---:|
| M0 - LSTM | +0.020 [+0.007, +0.035]; 60/101; Holm p 0.071 | +0.015 [-0.010, +0.037]; 65/98; Holm p 0.012 | +0.066 [+0.026, +0.105]; 20/29; Holm p 0.089 |
| M2 - LSTM | +0.015 [+0.002, +0.028]; 57/101; Holm p 0.165 | -0.009 [-0.052, +0.027]; 67/98; Holm p 0.015 | +0.079 [+0.042, +0.116]; 22/29; Holm p 0.008 |
| M1 - LSTM | +0.017 [+0.004, +0.032]; 59/101; Holm p 0.123 | +0.015 [-0.014, +0.040]; 68/98; Holm p 0.012 | +0.053 [+0.015, +0.090]; 19/29; Holm p 0.146 |
| M4 - LSTM | - | - | - |

Note: M2 - LSTM on held-out has a mean of -0.009 but a median of +0.020 and wins on 67 of 98 gauges: a few large losses pull the mean below zero.

Gap in median NSE (M0 minus LSTM): gauged +0.001, held-out +0.017, blocked +0.025. With lead the gap shrinks on gauged, shrinks on held-out, grows on blocked (leads up to 48 h against 144 h and beyond).

Blocked basins one by one (the three basins are the only independent units, so the gauge-level tests above are descriptive):

| Basin | Gauges | M0 | M1 | M2 | LSTM | M0 - LSTM (mean) | M0 better |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 0.371 | 0.367 | 0.363 | 0.270 | +0.128 | 12/12 |
| 2 | 11 | 0.271 | 0.275 | 0.346 | 0.346 | -0.023 | 4/11 |
| 3 | 6 | 0.297 | 0.293 | 0.271 | 0.282 | +0.103 | 4/6 |

Caveat for the Methods: the comparison is matched in forcing, attributes, antecedent storage, rain source and output units, but the network models also see other gauges' observations and have NWM pretraining; the LSTM has neither.

## 3. RQ2: is any network effect carried by physical routing rather than learned message passing?

Component effect = the full model minus the model without the component, days 1-5 NSE, mean paired difference [90% bootstrap]; gauges better; Holm p; equivalence within +-0.01 (paired TOST, Holm-adjusted).

| Component (comparison) | Held-out | Blocked |
|---|---:|---:|
| Routing (M1 - M4) | - | - |
| Message passing (M0 - M2) | +0.023 [-0.002, +0.054]; 50/98; Holm p 0.986; equivalent within +-0.01: no | -0.013 [-0.038, +0.010]; 14/29; Holm p 1.000; equivalent within +-0.01: no |

## 4. RQ3: do upstream gauge observations, carried downstream by the routing, add skill?

| Component (comparison) | Held-out | Blocked |
|---|---:|---:|
| Insertion (M0 - M1) | +0.000 [-0.007, +0.007]; 46/98; Holm p 0.986; equivalent within +-0.01: yes | +0.012 [+0.002, +0.026]; 13/29; Holm p 1.000; equivalent within +-0.01: no (insertion cannot act here: no scored blocked gauge has a visible gauge upstream) |

On blocked basins insertion cannot act, so M0 minus M1 there (+0.012, 90% interval [+0.002, +0.026]) is the difference between two trainings, not an insertion effect; it gives a scale for run-to-run differences with a single seed.

Counterfactual on the 40 held-out gauges that have a visible gauge upstream (the only gauges insertion can change): M0 with its insertion switched off at test time, same weights. Median NSE; gain = M0 minus M0 without insertion.

| Quantity | M0 | M0, insertion off | M1 | Gain from insertion (mean) |
|---|---:|---:|---:|---:|
| 6 h | 0.844 | 0.787 | 0.778 | +0.057 (p < 0.001) |
| 24 h | 0.652 | 0.702 | 0.695 | -0.008 (p 0.646) |
| per-gauge days 1-5 | 0.460 | 0.496 | 0.499 | +0.003 (p 0.581) |

Insertion helps at 6 h (+0.057, p < 0.001) and not beyond. Over all 98 held-out gauges the 6 h median NSE is M0 0.770 and M1 0.769, but the mean paired difference is +0.021: the median is set by the gauges insertion cannot reach.

## 5. KGE and its components (days 1-5, median over gauges)

Held-out (98 gauges):

| Model | KGE | r | alpha (variability) | beta (bias) |
|---|---:|---:|---:|---:|
| M0 | 0.429 | 0.661 | 0.617 | 0.933 |
| M1 | 0.428 | 0.663 | 0.618 | 0.944 |
| M2 | 0.421 | 0.657 | 0.628 | 0.954 |
| M4 | - | - | - | - |
| LSTM | 0.396 | 0.634 | 0.578 | 0.918 |
| NWM | 0.448 | 0.579 | 0.778 | 0.865 |

Blocked (29 gauges):

| Model | KGE | r | alpha (variability) | beta (bias) |
|---|---:|---:|---:|---:|
| M0 | 0.334 | 0.634 | 0.530 | 0.893 |
| M1 | 0.334 | 0.633 | 0.550 | 0.912 |
| M2 | 0.341 | 0.645 | 0.486 | 0.885 |
| M4 | - | - | - | - |
| LSTM | 0.381 | 0.575 | 0.626 | 0.880 |
| NWM | 0.385 | 0.536 | 0.744 | 0.855 |

KGE is the 2009 variant (alpha = std ratio, beta = mean ratio), validated against an independent implementation. On blocked basins the LSTM leads on KGE although M0, M1 and M2 lead on NSE: the M-models have the higher correlation but are more damped (lower alpha).

## 6. Pending

| Item | Status |
|---|---|
| M1 on the 3 blocked folds | done |
| M4: stage A, 5 held-out folds, 3 blocked folds | in progress (-) |
| Seed 2 on the blocked folds (M0, M1, M2, M4, LSTM) | after the job table is confirmed (-) |
| Seeds for held-out and gauged, seed 3, reach level for M1, M2, M4 | not planned |

The full report (inputs audit, leakage checks, high-flow measures, stratification, reach level) is `test_comparison_v3_full.md`; every table is a CSV in `reports/v3_tables/`.
