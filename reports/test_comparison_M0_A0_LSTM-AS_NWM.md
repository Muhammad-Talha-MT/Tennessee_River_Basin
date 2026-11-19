# Test-set comparison: M0, A0, LSTM-AS (L4) and NWM

Test issues 2020-09-23 to 2025-12-22 (1,917 00Z issues), GEFSv12 operational rain to 240 h. Metric: median NSE over gauges.
One seed for every model. M0 = Manning routing + gauge insertion + 3 graph layers. A0 = previous GNN (kernel routing, 3 graph layers).
LSTM-AS = L4 (area-scaled output, NSE* loss, antecedent inputs).

## Days 1-5 (mean over leads 24-120 h)

| Protocol (gauges) | M0 | A0 | LSTM-AS | NWM |
|---|---:|---:|---:|---:|
| Gauged (101) | **0.430** | 0.416 | 0.428 | 0.255 |
| Held-out, random folds (98) | **0.385** | 0.356 | 0.367 | 0.255 |
| Blocked basins (29) | **0.305** | 0.296 | 0.280 | 0.183 |

## By lead, all gauges in each protocol

| Protocol | Lead | M0 | A0 | LSTM-AS | NWM |
|---|---|---:|---:|---:|---:|
| Gauged | 24 h | 0.614 | 0.572 | 0.595 | 0.461 |
| Gauged | 120 h | 0.307 | 0.310 | 0.308 | 0.079 |
| Gauged | 240 h | 0.153 | 0.148 | 0.145 | -0.331 |
| Held-out | 24 h | 0.567 | 0.532 | 0.518 | 0.463 |
| Held-out | 120 h | 0.248 | 0.219 | 0.236 | 0.070 |
| Held-out | 240 h | 0.115 | 0.100 | 0.099 | -0.331 |
| Blocked | 24 h | 0.457 | 0.484 | 0.482 | 0.401 |
| Blocked | 72 h | 0.275 | 0.262 | 0.242 | 0.083 |
| Blocked | 120 h | 0.169 | 0.149 | 0.126 | 0.031 |
| Blocked | 240 h | 0.045 | 0.034 | 0.011 | -0.187 |

## By dam category, days 1-5

| Protocol | Subset (gauges) | M0 | A0 | LSTM-AS | NWM |
|---|---|---:|---:|---:|---:|
| Gauged | Headwater, no dam (54) | 0.405 | 0.397 | 0.399 | 0.170 |
| Gauged | Below a gauge, no dam (29) | 0.510 | 0.505 | not computed | 0.406 |
| Gauged | Dam in basin (18) | 0.421 | 0.403 | not computed | 0.209 |
| Held-out | Headwater, no dam (52) | 0.352 | 0.346 | 0.326 | 0.170 |
| Held-out | Below a gauge, no dam (29) | 0.486 | 0.460 | not computed | 0.406 |
| Held-out | Dam in basin (17) | 0.371 | 0.327 | not computed | 0.221 |
| Blocked | Headwater, no dam (18) | 0.295 | 0.279 | 0.257 | 0.043 |
| Blocked | Below a gauge, no dam (9) | 0.461 | 0.455 | 0.332 | 0.391 |
| Blocked | Dam in basin (2) | 0.004 | -0.065 | -0.125 | -0.014 |

## Paired significance, blocked protocol only (29 gauges, days 1-5 NSE per gauge, Wilcoxon signed-rank)

| Comparison | Mean difference | Gauges better | p |
|---|---:|---:|---:|
| M0 - LSTM-AS | +0.066 | 20 / 29 | 0.018 |
| A0 - LSTM-AS | +0.038 | 19 / 29 | 0.069 |
| M0 - A0 | +0.028 | 17 / 29 | 0.169 |

## Reach level: every reach against the NWM v3.0 retrospective (58,834 reaches, 425 issues to 2023-01-20)

Median NSE over reaches, GEFS rain. The target is the NWM retrospective flow, not observed flow (most reaches have no gauge), and
the NWM forecast row is scored against the same retrospective. "Held-out basins" are reaches inside the basin of a held-out random-fold gauge,
scored by the model that never saw that gauge. Best of the four models in bold.

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
- Paired tests for the gauged and held-out protocols.
- KGE and CSI for M0; the realized-rain (AORC) diagnostic for M0 and LSTM-AS at the reach level (GEFS rain only).
- M1 and M2 (still training).

## Sources

- M0: `models/attr/eval_M0_s1/report.md` (gauged, held-out), `models/attr/eval_S1_M0_s1/report.md` (blocked)
- A0: `models/attr/eval_A0_s1/report.md` (gauged, held-out), `models/attr/eval_S1_A0_s1/report.md` (blocked)
- LSTM-AS gauged and held-out: `figures/attribution/phase1/scores.csv` (model L4)
- LSTM-AS blocked: `models/attr/ladder_L4_s1_bench_b{1,2,3}`, scored with the blocked comparison script
- NWM: the NWM rows of the M0 and A0 reports
- Reach level: M0 `models/attr/eval_M0_reach` (4-GPU run), A0 `models/attr/eval_A0_s1`, LSTM-AS `models/attr/ladder_L4_s1_reaches` (`scripts/lstm_every_reach.py`, adapted for area-scaled output and antecedent inputs), NWM from the same reports
