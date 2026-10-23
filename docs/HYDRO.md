# Graph-routing forecast model (`tva_flood.hydro`)

One model, one training recipe, three tests: forecasts at gauges with their recent flow,
at gauges whose flow the model never saw, and at all 58,834 reaches.

## The forecast rule (every model, training and test)

At a 00Z issue a model sees what a forecaster has: observed AORC weather and observed USGS
flow before the issue, and after it only the GEFSv12 control rain forecast to 240 h; every
other future variable is unknown. Training uses the GEFSv12 reforecast (issues 2000-2018,
targets ending before 2019), selection 2019 the same way, and the test the operational
GEFSv12 (from 2020-09-23, the first day it exists). The new model, V2 (from scratch) and
the tuned LSTM all follow it (`scripts/gefs_pipeline.sh`). Realized rain appears only in
`+aorc` rows of the evaluation, as a perfect-rain diagnostic.

## Model

| Part | What it does | Why |
|---|---|---|
| Inputs | Per reach and step: weather (8 AORC variables + validity; in the future only forecast rain), season, 30 days of observed flow at a visible gauge's reach, and the owning gauge's specific discharge at every reach of its basin. Per reach: 48 static inputs, 7 antecedent-storage features, area ratio to the owning gauge | Gauge state enters directly (short leads); a whole basin sees what its gauge is doing (ungauged reaches) |
| LSTM, per reach | 128 units over 36 days of history and the 10-day forecast window | Runoff memory and timing |
| Graph layers | 3 blocks mixing each reach's state with the mean state of its upstream and downstream neighbours at every step (32 channels) | Neighbours and nearby gauges share information |
| Simulation head | Lateral inflow q >= 0 = runoff depth x local catchment area, routed | Trained to the NWM v3.0 retrospective at every reach (the prior) |
| Observed head | q x exp(delta), delta time-varying from the graph states, kept small by a penalty | Trained on gauges: corrects NWM-like flow where observations disagree, stays close to it elsewhere |
| Routing | Q_i(t) = sum over upstream j of f_ij (h_g * q_j)(t); 27 kernels (river distance x lake reaches on the path), each summing to one, 12 days long | Mass is conserved: headwaters cannot add up to more than the river below them; main stems are trained like any reach |
| Gauge head | At visible gauges: an LSTM over 30 days of observed flow and 8 days of routed flow, a decoder over the 40 leads (routed forecasts, basin-mean forecast rain, lead, gauge descriptors); output = routed flow x exp(carried misfit + learned correction) | Uses the gauge's own history at every lead |
| Spread | Log-flow standard deviation at gauges | Probabilistic flood scores |

Antecedent storage (`processed/hydro/antecedent.npy`): exponentially weighted rain over
7, 30, 90, 180 and 365 days and temperature over 30 and 90 days, per reach and day.

## Training

| Stage | Inputs | Loss | Purpose |
|---|---|---|---|
| A, pre-training | Forecast samples (the rule above); no gauges | Simulation head vs NWM retrospective at every reach over 8 days before and 10 days after the issue: bounded per-reach NSE, log(1 + MSE/variance), on flow and log flow | Runoff, graph mixing and routing everywhere, main stems included |
| B, forecast | Trunk, simulation head and routing from A; other heads fresh. A random half of the eligible gauges visible (none in 15% of samples) | Gauge head and routed observed head at every training gauge, shown or not (bounded per-gauge NSE, x3 above the gauge's 90th percentile) + 0.3 x NWM prior (simulation head everywhere; observed head at a tenth inside visible gauges' basins) + 0.1 x spread + 0.01 x mean delta squared | One model for gauged and ungauged use; NWM keeps ungauged reaches in place, gauges win in their basins |

A fold (`--fold k`, five folds of CAMELS-H gauges, `data.folds`) withholds its gauges
from targets, inputs and selection. Stage B selects on 2019: the mean of median NSE at
training gauges with history shown, with history hidden, and against the NWM
retrospective at every reach, over leads 24-120 h.

## Commands

```bash
python -m tva_flood.hydro.prepare --workers 24   # processed/hydro (CPU, ~5 min; once)
setsid -f nohup bash scripts/gefs_pipeline.sh </dev/null >>/data/tva_data/logs/gefs_pipeline.log 2>&1
```

The pipeline trains the new model (`models/gefs_hydro_A`, `gefs_hydro_B`, folds
`gefs_hydro_B_f1..5`), V2 (`gefs_v2*`) and the LSTM (`gefs_lstm*`), scores V2 and the
LSTM at the gauges (`gefs_benchmark*`) and V2 and NWM at every reach
(`retrospective_comparison_gefs_*`), then `tva_flood.hydro.evaluate` writes the three
protocols for all of them on the same issues, gauges and masks
(`models/gefs_evaluation/report.md`).

The evaluation also splits results by the TVA dams (`python -m tva_flood hydro-dams`,
`processed/hydro/dams.npz`), which the model never reads: gauges in headwater basins
without a dam, below another gauge without a dam, and with a dam in the basin; reaches
with more than half their basin behind a dam.

## Tried and set aside

V3 (git tag `v3-snapshot`; `models/v3_*`, `models/v3_evaluation/report.md`) added a
reservoir module trained on TVA's releases, routing split at the dams, basin pooling and
gauge corrections carried downstream. It tied this model at the gauges and on ungauged
gauges and lost on larger rivers against the NWM retrospective; the dam split helped only
at regulated gauges at 24 h.

## Not yet included

Gauges in neighbouring regions, karst and lithology attributes, GEFS ensemble members
(only the control member is downloaded) and HRRR short-range rain need new downloads.
