# Regulated graph-routing forecast model (`tva_flood.hydro`, V3)

One model, one training recipe, four tests: forecasts at gauges with their recent flow,
at gauges whose flow the model never saw, at all 58,834 reaches, and at the TVA dams.

## The forecast rule (every model, training and test)

At a 00Z issue a model sees what a forecaster has: observed AORC weather, observed USGS
flow and the TVA dams' observed releases and inflows before the issue, and after it only
the GEFSv12 control rain forecast to 240 h; every other future variable is unknown.
Training uses the GEFSv12 reforecast (issues 2000-2018, targets ending before 2019),
selection 2019 the same way, and the test the operational GEFSv12 (from 2020-09-23).
Realized rain appears only in `+aorc` rows of the evaluation, as a perfect-rain diagnostic.
On the TVA-plan track the reservoir module also reads TVA's forecast release in force at
the issue (issued at or before it).

## The dams (`hydro.dams`, `processed/hydro/dams.npz`)

TVA's archive of 44 dams: 42 are on the graph (Barkley and Great Falls are on the
Cumberland). Six-hour observed releases (1976-2026) and inflows (computed by TVA from
releases and storage changes), and TVA's forecasts of both (32 dams, from 2016-10).
Timestamps are UTC (release changes match USGS gauges just below two dams at lag 0). Each
dam sits at the reach that ends at the dam and carries its water (mean NWM flow within a
factor 2 of the mean release); four reaches are fixed by hand, with the reasons in the
module. For every reach the file holds the first dam on its way down and the share of its
basin whose water passes a dam first (2,405 reaches have more than half).

## Model (`model.RegulatedModel`)

| Part | What it does | Why |
|---|---|---|
| Inputs | Per reach and step: weather (in the future only forecast rain), season, 30 days of observed flow at a visible gauge's reach, the owning gauge's specific discharge across its basin. Per reach: 48 static inputs, 7 antecedent-storage features, area ratio to the owning gauge | Gauge state enters directly; a basin sees what its gauge is doing |
| LSTM, per reach | 128 units over 36 days of history and the 10-day forecast window | Runoff memory and timing |
| Basin pooling | Area-weighted means of each reach's state over its whole upstream basin in near, middle and far river-distance bands, then 2 local graph layers (upstream and downstream neighbours) | What the basin above is doing, beyond a few hops |
| Simulation head | Lateral inflow q >= 0, routed along the natural river | The NWM prior (off the dams' reach) |
| Observed head | q x exp(delta), delta bounded and kept small | Corrects runoff where gauges disagree |
| Routing, split at dams | Q_i(t) = sum over upstream j not behind a dam above i of f_ij (h_g * q_j)(t) + releases of the dams above i (to the next dam), with one release kernel per distance bin; 27 runoff kernels (distance x lakes), each summing to one | Mass is conserved; below a dam the river carries what TVA releases |
| Reservoir module | Per dam: a GRU over 20 days of release, observed and routed inflow, with the storage change over 30 and 90 days; a decoder over the leads reading the routed inflow forecast, season and (plan track) TVA's planned release. Dams above first | Releases are operators' decisions: learned from 40 years of them |
| Gauge head | At visible gauges: the LSTM's inputs (30 days of flow, basin-mean weather and attributes, basin-mean forecast rain) plus the routed and simulated flow; output = routed flow x exp(correction) | It sees everything the LSTM sees, and the river network besides |
| Corrections | A visible gauge's correction (observed or forecast minus routed) carried down its main path to the next visible gauge or dam | Downstream reaches benefit from a gauge |
| Spread | Log-flow standard deviation at gauges | Probabilistic scores |

## Training (`hydro.train`)

| Stage | Inputs | Loss |
|---|---|---|
| A | Forecast samples, no gauges | Simulation head vs NWM retrospective, bounded NSE per reach, weighted by the share of the basin not behind a dam; forecast releases vs observed; reservoir inflow vs TVA's (daily means) |
| B | Trunk, simulation head, routing and reservoirs from A. A random half of the eligible gauges visible (none in 15%) | Gauge head and routed head (with corrections) at every training gauge, flows above the 90th percentile x3 + 0.3 NWM prior (a tenth inside visible gauges' basins) + 0.5 dams + 0.1 spread + 0.01 delta squared |

A fold (`--fold k`) withholds its gauges from targets, inputs and selection (dams never).
Stage B selects on 2019: the mean of median NSE at gauges with history shown, hidden,
against NWM at every reach, and of the forecast releases, over leads 24-120 h.

## Commands

```bash
python -m tva_flood hydro-prepare     # processed/hydro: routing, reach statistics, antecedent storage (once)
python -m tva_flood hydro-dams        # processed/hydro/dams.npz and dams.csv (once)
setsid -f nohup bash scripts/v3_pipeline.sh </dev/null >>/data/tva_data/logs/v3_pipeline.log 2>&1
```

The pipeline trains `models/v3_A`, `v3_B` and `v3_B_f1..5` and writes `models/v3_evaluation/report.md`: gauged, ungauged, every reach
(with the NWM forecast and the LSTM run at every reach, `scripts/lstm_every_reach.py`) and
the dams (against TVA's forecasts and persistence), split by regulation.

A third stage, training the gauge head further on a frozen stage-B model with cached
inputs (`hydro.head`), was tried and dropped: on unseen years (2017-2018 and 2019) the head
got worse by its own loss and by NSE, at every learning rate tried. Stage B already shows
the head about a million (issue, gauge) examples; more passes only overfit.

Earlier models stay loadable for evaluation: `HydroModel` (GEFS-rule run, `models/gefs_*`)
and V1 (`legacy.HydroModelV1`, `--legacy --rain-steps 16`).
