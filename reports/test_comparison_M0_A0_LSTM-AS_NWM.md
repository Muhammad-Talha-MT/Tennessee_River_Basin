# Test-set comparison: M0, M1, M2, A0, LSTM-AS (L4) and NWM

Test issues 2020-09-23 to 2025-12-22 (1,917 00Z issues), GEFSv12 operational rain to 240 h. Metric: median NSE over gauges unless stated.
Days 1-5 is the mean over leads 24, 48, 72, 96 and 120 h of the median over gauges. One seed for every model.

| Model | What it is |
|---|---|
| M0 | New model: Manning routing + gauge insertion + 3 graph layers |
| M1 | M0 without insertion (Manning routing + 3 graph layers) |
| M2 | M0 without graph layers (Manning routing + insertion) |
| A0 | Previous GNN: kernel routing + 3 graph layers |
| LSTM-AS (L4) | Lumped LSTM baseline with area-scaled output, NSE* loss and antecedent inputs |
| NWM | National Water Model v3.0 forecast |

Protocols: **gauged** = model trained on all gauges, scored at them (101 gauges); **held-out** = 5 random folds, each scored at gauges it never saw (98);
**blocked** = 3 whole tributary basins withheld (29 gauges). M1 has no blocked runs.

## Summary

- **Gauged:** M0 0.430, M1 0.428, LSTM-AS 0.428, M2 0.426, A0 0.416, NWM 0.255. M0, M1 and M2 are ahead of A0 (+0.011 to +0.016, p < 0.01). The three differ from each other by at most 0.005 (M0 - M1 is small but detectable, p = 0.001). M0 is ahead of LSTM-AS by +0.020 (p = 0.018).
- **Held-out:** M2 0.386, M0 0.385, M1 0.383, LSTM-AS 0.367, A0 0.356, NWM 0.255. M0 and M1 are ahead of A0 by +0.051 (p < 0.001) and of LSTM-AS by +0.015 (p = 0.002).
- **Blocked:** M2 0.322, M0 0.305, A0 0.296, LSTM-AS 0.280, NWM 0.183 by NSE. M0 and M2 are ahead of LSTM-AS (p = 0.018 and 0.001); neither is significantly ahead of A0. By KGE the order reverses (LSTM-AS 0.381, A0 0.367, M2 0.341, M0 0.334).
- **What drives the gain over A0:** insertion adds about nothing (M0 - M1: +0.002 gauged, +0.000 held-out, p = 0.91). The graph layers add little (M0 - M2: +0.005 gauged, +0.023 held-out but not significant, -0.013 blocked). The Manning routing is the common part of every M-model, so most of the gain comes from it.
- **Reach level (vs NWM retrospective):** M0 is ahead of A0, LSTM-AS and NWM on all reaches and small basins; A0 is clearly ahead on mid-size and large rivers at 24 h.
- **Caveats:** single seed; paired p-values are not corrected for multiple comparisons; KGE comes from my own code.

## Days 1-5 NSE

| Protocol (gauges) | M0 | M1 | M2 | A0 | LSTM-AS | NWM |
|---|---:|---:|---:|---:|---:|---:|
| Gauged (101) | **0.430** | 0.428 | 0.426 | 0.416 | 0.428 | 0.255 |
| Held-out, random folds (98) | 0.385 | 0.383 | **0.386** | 0.356 | 0.367 | 0.255 |
| Blocked basins (29) | 0.305 | not run | **0.322** | 0.296 | 0.280 | 0.183 |

## NSE by lead

Gauged (101 gauges):

| Model | 6 h | 24 h | 72 h | 120 h | 240 h | Days 1-5 |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.919 | 0.614 | 0.401 | 0.307 | 0.153 | **0.430** |
| M1 | 0.916 | 0.612 | 0.402 | 0.310 | 0.151 | 0.428 |
| M2 | 0.918 | 0.608 | 0.400 | 0.306 | 0.149 | 0.426 |
| A0 | 0.918 | 0.572 | 0.390 | 0.310 | 0.148 | 0.416 |
| LSTM-AS | 0.804 | 0.595 | 0.403 | 0.308 | 0.145 | 0.428 |
| NWM | 0.837 | 0.461 | 0.244 | 0.079 | -0.331 | 0.255 |

Held-out, random folds (98 gauges):

| Model | 6 h | 24 h | 72 h | 120 h | 240 h | Days 1-5 |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.770 | 0.567 | 0.372 | 0.248 | 0.115 | 0.385 |
| M1 | 0.769 | 0.569 | 0.369 | 0.235 | 0.115 | 0.383 |
| M2 | 0.754 | 0.567 | 0.366 | 0.253 | 0.111 | **0.386** |
| A0 | 0.718 | 0.532 | 0.330 | 0.219 | 0.100 | 0.356 |
| LSTM-AS | 0.690 | 0.518 | 0.361 | 0.236 | 0.099 | 0.367 |
| NWM | 0.842 | 0.463 | 0.245 | 0.070 | -0.331 | 0.255 |

Blocked basins (29 gauges):

| Model | 6 h | 24 h | 72 h | 120 h | 240 h | Days 1-5 |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.710 | 0.457 | 0.275 | 0.169 | 0.045 | 0.305 |
| M2 | 0.688 | 0.463 | 0.305 | 0.201 | 0.041 | **0.322** |
| A0 | 0.711 | 0.484 | 0.262 | 0.149 | 0.034 | 0.296 |
| LSTM-AS | 0.719 | 0.482 | 0.242 | 0.126 | 0.011 | 0.280 |
| NWM | 0.773 | 0.401 | 0.083 | 0.031 | -0.187 | 0.183 |

## NSE by dam category, days 1-5

| Protocol | Subset (gauges) | M0 | M1 | M2 | A0 | LSTM-AS | NWM |
|---|---|---:|---:|---:|---:|---:|---:|
| Gauged | Headwater, no dam (54) | **0.405** | 0.401 | 0.392 | 0.397 | 0.399 | 0.170 |
| Gauged | Below a gauge, no dam (29) | 0.510 | 0.505 | **0.512** | 0.505 | not computed | 0.406 |
| Gauged | Dam in basin (18) | 0.421 | 0.418 | **0.425** | 0.403 | not computed | 0.209 |
| Held-out | Headwater, no dam (52) | 0.352 | 0.348 | **0.365** | 0.346 | 0.326 | 0.170 |
| Held-out | Below a gauge, no dam (29) | 0.486 | **0.498** | 0.485 | 0.460 | not computed | 0.406 |
| Held-out | Dam in basin (17) | 0.371 | **0.383** | 0.343 | 0.327 | not computed | 0.221 |
| Blocked | Headwater, no dam (18) | **0.295** | not run | 0.277 | 0.279 | 0.257 | 0.043 |
| Blocked | Below a gauge, no dam (9) | **0.461** | not run | 0.460 | 0.455 | 0.332 | 0.391 |
| Blocked | Dam in basin (2) | 0.004 | not run | **0.115** | -0.065 | -0.125 | -0.014 |

## KGE

KGE is not in the evaluation reports. It is computed here from the saved forecasts, with the same gauges and issues as the NSE tables: median over gauges per
lead, Days 1-5 as the mean over leads 24-120 h. NWM KGE was not computed. The code is `scripts/test_comparison_gauges.py` and has not been cross-checked
against a reference implementation.

Gauged (101 gauges):

| Model | 6 h | 24 h | 72 h | 120 h | 240 h | Days 1-5 |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.926 | 0.702 | 0.512 | 0.396 | 0.150 | **0.526** |
| M1 | 0.919 | 0.696 | 0.521 | 0.398 | 0.156 | **0.526** |
| M2 | 0.923 | 0.703 | 0.505 | 0.382 | 0.150 | 0.517 |
| A0 | 0.926 | 0.695 | 0.508 | 0.398 | 0.161 | 0.521 |
| LSTM-AS | 0.695 | 0.551 | 0.505 | 0.379 | 0.157 | 0.481 |

Held-out, random folds (98 gauges):

| Model | 6 h | 24 h | 72 h | 120 h | 240 h | Days 1-5 |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.718 | 0.591 | 0.413 | 0.317 | 0.086 | **0.429** |
| M1 | 0.681 | 0.592 | 0.410 | 0.308 | 0.094 | 0.428 |
| M2 | 0.731 | 0.555 | 0.437 | 0.295 | 0.107 | 0.421 |
| A0 | 0.627 | 0.545 | 0.418 | 0.319 | 0.114 | 0.419 |
| LSTM-AS | 0.602 | 0.482 | 0.414 | 0.308 | 0.099 | 0.396 |

Blocked basins (29 gauges):

| Model | 6 h | 24 h | 72 h | 120 h | 240 h | Days 1-5 |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.566 | 0.504 | 0.311 | 0.190 | 0.025 | 0.334 |
| M2 | 0.552 | 0.488 | 0.347 | 0.196 | 0.018 | 0.341 |
| A0 | 0.534 | 0.532 | 0.365 | 0.221 | 0.036 | 0.367 |
| LSTM-AS | 0.649 | 0.494 | 0.380 | 0.273 | 0.057 | **0.381** |

On KGE the ranking on blocked basins reverses relative to NSE: LSTM-AS and A0 are ahead of M0 and M2. On gauged and held-out, M0 and M1 are ahead of A0 by 0.005 to 0.010 and of LSTM-AS by 0.033 to 0.045; M2 is level with or slightly behind A0.
The A0 held-out 6 h KGE (0.627) looks low next to its NSE and has not been checked.

## Paired tests

Days 1-5 NSE per gauge (mean over leads 24-120 h), difference of the first model minus the second, number of gauges where the first is better, and a two-sided Wilcoxon signed-rank p.
The mean can disagree with the win count when a few gauges have very large losses, as for M2 - LSTM-AS on held-out gauges.

| Comparison | Gauged (101): diff, wins, p | Held-out (98): diff, wins, p | Blocked (29): diff, wins, p |
|---|---|---|---|
| M0 - A0 | +0.016, 75/101, < 0.001 | +0.051, 72/98, < 0.001 | +0.028, 17/29, 0.169 |
| M1 - A0 | +0.014, 72/101, < 0.001 | +0.051, 78/98, < 0.001 | not run |
| M2 - A0 | +0.011, 59/101, 0.006 | +0.027, 63/98, 0.006 | +0.041, 18/29, 0.275 |
| M0 - M1 (insertion) | +0.002, 67/101, 0.001 | +0.000, 46/98, 0.906 | not run |
| M0 - M2 (graph layers) | +0.005, 59/101, 0.013 | +0.023, 50/98, 0.493 | -0.013, 14/29, 0.579 |
| M1 - M2 | +0.003, 62/101, 0.119 | +0.023, 50/98, 0.125 | not run |
| M0 - LSTM-AS | +0.020, 60/101, 0.018 | +0.015, 65/98, 0.002 | +0.066, 20/29, 0.018 |
| M1 - LSTM-AS | +0.017, 59/101, 0.041 | +0.015, 68/98, 0.002 | not run |
| M2 - LSTM-AS | +0.015, 57/101, 0.083 | -0.009, 67/98, 0.004 | +0.079, 22/29, 0.001 |
| A0 - LSTM-AS | +0.004, 50/101, 0.904 | -0.036, 55/98, 0.616 | +0.038, 19/29, 0.069 |

## Reach level: every reach against the NWM v3.0 retrospective (58,834 reaches, 425 issues to 2023-01-20)

Median NSE over reaches, GEFS rain. The target is the NWM retrospective flow, not observed flow (most reaches have no gauge), and
the NWM forecast row is scored against the same retrospective. "Held-out basins" are reaches inside the basin of a held-out random-fold gauge,
scored by the model that never saw that gauge. Best of the four models in bold. M1 and M2 were not evaluated at the reach level.

| Lead | Scope | Reaches | M0 | A0 | LSTM-AS | NWM |
|---|---|---:|---:|---:|---:|---:|
| 24 h | all reaches | 58,834 | **0.062** | 0.058 | -0.017 | -0.059 |
| 24 h | held-out basins | 31,684 | **0.081** | 0.068 | 0.002 | -0.087 |
| 24 h | outside every gauge basin | 25,588 | 0.042 | **0.048** | -0.054 | -0.032 |
| 24 h | held-out basins, < 50 km2 | 24,605 | **0.054** | 0.043 | -0.086 | -0.174 |
| 24 h | held-out basins, 50-500 | 4,223 | **0.320** | 0.282 | 0.264 | 0.238 |
| 24 h | held-out basins, 500-5,000 | 2,575 | 0.435 | **0.599** | 0.428 | 0.491 |
| 24 h | held-out basins, > 5,000 | 281 | -0.315 | **-0.313** | -3.282 | -0.869 |
| 24 h | behind dams (> 50% of basin) | 2,405 | -0.123 | -0.178 | -0.508 | **-0.054** |
| 24 h | free-flowing, > 500 km2 | 2,286 | 0.481 | 0.614 | 0.435 | **0.628** |
| 72 h | all reaches | 58,834 | **0.024** | 0.014 | -0.118 | -0.192 |
| 72 h | held-out basins | 31,684 | **0.029** | 0.013 | -0.153 | -0.264 |
| 72 h | outside every gauge basin | 25,588 | **0.018** | 0.015 | -0.078 | -0.112 |
| 72 h | held-out basins, < 50 km2 | 24,605 | **0.014** | 0.001 | -0.296 | -0.409 |
| 72 h | held-out basins, 50-500 | 4,223 | **0.171** | 0.121 | 0.116 | 0.040 |
| 72 h | held-out basins, 500-5,000 | 2,575 | 0.270 | **0.337** | 0.214 | 0.331 |
| 72 h | held-out basins, > 5,000 | 281 | -0.324 | **-0.236** | -2.027 | -1.364 |
| 72 h | behind dams (> 50% of basin) | 2,405 | -0.106 | -0.111 | -0.593 | **-0.102** |
| 72 h | free-flowing, > 500 km2 | 2,286 | 0.259 | **0.331** | 0.217 | 0.331 |
| 120 h | all reaches | 58,834 | **0.026** | 0.022 | -0.087 | -1.231 |
| 120 h | held-out basins | 31,684 | **0.031** | 0.024 | -0.107 | -1.180 |
| 120 h | outside every gauge basin | 25,588 | **0.020** | 0.017 | -0.063 | -1.349 |
| 120 h | held-out basins, < 50 km2 | 24,605 | **0.015** | 0.011 | -0.217 | -1.648 |
| 120 h | held-out basins, 50-500 | 4,223 | **0.169** | 0.135 | 0.111 | -0.446 |
| 120 h | held-out basins, 500-5,000 | 2,575 | **0.292** | 0.273 | 0.207 | 0.170 |
| 120 h | held-out basins, > 5,000 | 281 | -0.340 | **-0.088** | -1.792 | -1.781 |
| 120 h | behind dams (> 50% of basin) | 2,405 | **-0.050** | -0.083 | -0.399 | -0.177 |
| 120 h | free-flowing, > 500 km2 | 2,286 | 0.274 | **0.277** | 0.221 | 0.143 |

## Not included

- LSTM-AS values for the gauged and held-out "below a gauge" and "dam in basin" groups (not saved in the earlier attribution run).
- NWM KGE; CSI for all models; the realized-rain (AORC) diagnostic for M0 and LSTM-AS at the reach level (GEFS rain only).
- M1 and M2 at the reach level, and M1 on the blocked folds (no blocked M1 runs were trained).
- Multiple seeds; no correction for multiple comparisons in the paired tests.

## Sources

- M0: `models/attr/eval_M0_s1` (gauged, held-out), `models/attr/eval_S1_M0_s1` (blocked), `models/attr/eval_M0_reach` (reach level)
- M1: `models/attr/eval_M1_s1` (gauged, held-out)
- M2: `models/attr/eval_M2_s1` (gauged, held-out), `models/attr/eval_S1_M2_s1` (blocked)
- A0: `models/attr/eval_A0_s1` (gauged, held-out, reach level), `models/attr/eval_S1_A0_s1` (blocked)
- LSTM-AS: `models/attr/ladder_L4_s1_bench` (gauged), `ladder_L4_s1_bench_f{1..5}` (held-out), `ladder_L4_s1_bench_b{1,2,3}` (blocked), `ladder_L4_s1_reaches` (reach level; `scripts/lstm_every_reach.py`, adapted for area-scaled output and antecedent inputs)
- NWM: the NWM rows of the M0 and A0 reports
- NSE, KGE and paired tests at the gauges: `scripts/test_comparison_gauges.py` (computes them from the saved forecasts). Its gauged and held-out LSTM-AS days 1-5 NSE (0.428, 0.367) match the earlier attribution scores, and its blocked A0, M0 and M2 values match their evaluation reports; there is no report to check the blocked LSTM-AS values against
