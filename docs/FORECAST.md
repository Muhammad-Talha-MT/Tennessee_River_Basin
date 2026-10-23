# Forecast-mode training and evaluation

At issue time t a forecaster predicts flow at t+1 ... t+40 (six-hour steps, 10 days)
using everything observed up to t. This workflow trains and scores exactly that.

## What the model sees at issue time

| Input | Source |
|---|---|
| 30 days of weather on every reach | AORC (8 variables plus validity flags) |
| Flow history | **Observed flow at gauges only**: TVA headwater gauges and every CAMELS-H gauge in Tennessee (the catalog). Every other reach has no flow history |
| Future precipitation | A forecast, known only up to `--rain-hours` (default 96) and only where a forecast exists. Nothing else about the future |
| Static attributes | 48 reach attributes |

TVA's forecast rain is real to about 72 h, carries a fifth of the observed rain from
78 to 96 h, and is zero filled after that, so 96 h is the default window. The forecast still runs 10
days. Rain from GFS or GEFS can extend the window later. NWM flow is never an input
at test time: NWM v3.0 ends in January 2023 and is not observed.

## Stages

| Stage | Command | Targets | What it teaches |
|---|---|---|---|
| 1 | `train-forecast --history-gauges` | NWM flow at all 58,834 reaches | Spatial dynamics, ungauged reaches, rain use. Flow history is visible only at gauge reaches (NWM flow stands in for the observation) |
| 2 | `finetune-observed` | Observed flow at the catalog gauges | Real gauge behaviour. History is the observed flow at every gauge in the neighbourhood |
| 3 | `evaluate-observed` | Observed flow at TVA gauges | Skill against TVA's own forecasts, 2020 on (never used in training) |

`ssl-pretrain` (masked flow reconstruction) is optional: pass its checkpoint to
`train-forecast --init`. In our test it matched training from scratch (0.081 vs
0.083 selection), so the default workflow starts from scratch.

Design points:

- **Encoder and decoder.** Causal dilated temporal blocks then directional sparse
  graph blocks (LayerNorm only, weights independent of graph size), and a decoder
  that reads the encoder state, a lead embedding and future rain. `--hidden 128`
  gives 502k parameters (the first version was hidden 32, 36k).
- **Whole basins (`--direction upstream`, from h128_v3).** With 4 + 2 graph blocks a
  reach sees 6 reaches; a TVA gauge's basin has a median of 181 reaches up to 37 hops
  away, so only about 10% of it was visible. `build-routing` stores every reach's
  whole upstream basin (area weights and river distances, `data/river.py`).
  `--basin-inputs` adds basin-mean weather history, `--basin-rain` basin-mean future
  rain, and `--routing` sends every upstream reach's decoder state to each reach
  through a learned travel-time kernel per distance bin. Graph blocks then pass
  messages from upstream only (`--aggregation attention` learns neighbour weights),
  and patches are whole basins: their outputs equal full-network outputs
  (`tests/test_river.py`). The 1.4% of reaches with more than 6,000 upstream reaches
  (the main stems) are not trained in stage 1. Stage 2 caps an update at
  `--max-nodes` reaches.
- **Exact subgraphs.** A batch is some reaches plus a halo as many hops as the model
  has graph blocks, so those reaches' outputs equal full-graph outputs. Training
  and evaluation use this; losses are at the chosen reaches only.
- **Rain during training.** Realized AORC rain, shown over a random share of the
  area, up to a random lead, or not at all, plus TVA's own forecast rain at TVA
  gauges in its forecast era (October 2016 on). With `--rain-error` (both stages)
  the AORC rain first gets the errors of TVA's rain forecast, measured against TVA's
  observed basin rain in 2016-2018 (`data/rain_error.py`): per lead day a miss or a
  log-normal factor (days 1-3: bias 0.90-0.95, 12-18% of storms missed; day 4: bias
  0.19, 85% missed), and a one-step timing shift. Without it the model trains on
  perfect rain to 96 h and meets TVA's much weaker rain at test time.
- **Loss in stage 2.** Normalized log-flow error plus a raw-flow error scaled by each
  gauge's training spread. Above its training 90th percentile the raw error weighs
  `1 + --high-flow-weight` times more, and `1 + --under-weight` times more again
  where the forecast is too low: a log loss alone learns to shrink flood peaks.
- **Selection in stage 2.** `selection` (higher is better) = median NSE over TVA
  gauges in 2019 averaged over the 6 to 120 h leads (`nse_selection`) plus
  `--csi-weight` x the flood CSI (flow above each gauge's training `--flood-quantile`,
  pooled over TVA gauges, same leads; `csi_selection`). TVA's forecast rain is the
  only rain. `tva_*` columns are TVA's own scores on the same issues.
  `--validation-issues 0` scores every 2019 issue; `--ema-decay` validates and saves
  a moving average of the weights.
- **Resume.** Rerun the identical command. Stage 1 also saves every evaluated
  checkpoint and the source snapshot.

## Splits

| Period | Use |
|---|---|
| 1979 to 2018 | Training (both stages) |
| 2019 | Validation and checkpoint selection |
| 2020 to 2025 | Test (`prepare-test`); CAMELS-H flow ends 2024-12-31, NWM flow ends 2023-01 |

## Commands

```bash
cd /home/talhamuh/tva_flood/tva_flood_clean
PY=/data/tva_data/conda_envs/tva_gnn/bin/python
M=/data/tva_data/models
CACHE=/data/tva_data/processed/pretraining_full_arrays
TEST=/data/tva_data/processed/test_preparation_2020_2025
G=/data/tva_data/processed/gauges/gauges.nc
RUN="env OMP_NUM_THREADS=1 PYTHONPATH=src"
```

### 0. Data (once)

```bash
$RUN $PY -m tva_flood build-array-cache --workers 8          # training months
$RUN $PY -m tva_flood build-gauges                           # observed-flow gauge catalog
# Test period: AORC for 2023-02 to 2025-12 needs the download env (s3fs)
PYTHONPATH=src /data/tva_data/conda_envs/tva_download/bin/python -m tva_flood \
  download-forcing --source aorc --start 2023-02-01 --stop 2026-01-01 --workers 6
$RUN $PY -m tva_flood aggregate --start 2023-02-01 --stop 2026-01-01 --workers 4 --no-streamflow
$RUN $PY -m tva_flood prepare-test --stop 2026-01-01 --out $TEST
$RUN $PY -m tva_flood build-array-cache --preparation $TEST --workers 8
```

`build-gauges` matches each CAMELS-H gauge to the graph reach nearest its
coordinates whose drainage area is within 20 to 25% of the gauge's, keeps one gauge
per reach (TVA's own series first), and writes `gauges.nc`, `gauges.csv` and
`excluded.csv` next to each other.

### Everything at once

`scripts/forecast_pipeline.sh` runs stages 1 and 2, the six test years and the event
scores (`DRY_RUN=1` prints the commands). Its header lists the current settings
(`NAME=h128_v2`): `--rain-error` in both stages, 15,000 stage-1 updates, and four
stage-2 loss variants trained side by side (one GPU each); the one with the best 2019
`selection` is written to `finetune_h128_v2.json` and is the one tested. The commands
below are the single steps.

### 1. Stage 1: forecast training on NWM targets (4 GPUs, about 3 hours)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 PYTHONPATH=src \
setsid -f nohup $PY -u -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m tva_flood train-forecast --array-cache $CACHE --history-gauges $G \
  --hidden 128 --batch-size 2 --lr 5e-4 --rain-hours 96 \
  --out $M/forecast_h128_v1 --updates 10000 --shuffle global \
  </dev/null >>/data/tva_data/logs/forecast_h128_v1.log 2>&1
```

Each update uses 8 windows (2 per GPU) of 6,000-reach patches and about 30 GiB per
GPU. Watch `validation_progress.csv`: `selection` is the band-balanced log MSE with
gauge-only flow history (lower is better).

### 2. Stage 2: fine-tune on observed flow (1 GPU, about 1 hour)

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=src \
setsid -f nohup $PY -u -m tva_flood finetune-observed --array-cache $CACHE \
  --init $M/forecast_h128_v1/best.pt --out $M/finetune_h128_v1 --updates 4000 \
  </dev/null >>/data/tva_data/logs/finetune_h128_v1.log 2>&1
```

Each update draws 32 gauges and 8 issue times (about 22 GiB). `validation_progress.csv`
has `selection` (higher is better), `tva_selection` (TVA's score on the same issues),
and CAMELS-H gauge scores (`camelsh_*`, with no rain unless a TVA basin covers them).

### 3. Evaluate on the test years (4 GPUs, about 15 minutes per year)

```bash
for YEAR in 2020 2021 2022 2023 2024 2025; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 PYTHONPATH=src \
  $PY -m torch.distributed.run --standalone --nproc_per_node=4 -m tva_flood \
    evaluate-observed --split test --preparation $TEST --array-cache ${TEST}_arrays \
    --issues-from $YEAR-01-01 --issues-until $((YEAR+1))-01-01 \
    --checkpoint tuned=$M/finetune_h128_v1/best.pt --out $M/test_$YEAR
done
```

Do not abbreviate to `--start`/`--stop`: `torchrun` reads them as `--start-method`.

Rows to compare, per lead (`report.md`):

- `tuned+qpf`: TVA's forecast rain over TVA gauge basins. The fair comparison with `tva_forecast`.
- `tuned+aorc`: realized rain over the same basins: a perfect rain forecast (upper bound).
- `tuned+unknown`: no future rain.
- `persistence`, `tva_forecast`, and `nwm_retrospective` (NWM driven by realized AORC,
  scored where NWM flow exists).

Per gauge and lead: NSE, KGE (with r, alpha, beta), RMSE, peak error. The
comparison to day 5 (120 h) is the one that carries information, because TVA has no
rain beyond about 4 days and no skill beyond day 5.

### 4. Flood-event scores

Median NSE is dominated by ordinary flow. `evaluate-observed` also saves every forecast
(`forecasts.npz`), and `evaluate-events` scores them on floods:

```bash
$RUN $PY -m tva_flood evaluate-events --runs $M/test_h128_v1_20{20,21,22,23,24,25} \
  --out $M/event_scores_h128_v1
```

- **Exceedance:** per gauge, was flow above its p95, p99 or median-annual-maximum
  threshold (from 1979 to 2018 only), and did the forecast say so? POD, FAR, CSI and
  frequency bias per lead, with a bootstrap interval on the CSI difference to TVA over
  (year, gauge) cells.
- **Observed peaks:** for each independent observed peak above a threshold, the
  forecast issued k leads before it: peak size error, timing error (maximum within
  +/- `--window` steps of the observed peak) and whether it crossed the threshold.

Outputs: `report.md`, `exceedance_by_lead.csv`, `csi_difference.csv`,
`events_by_lead.csv`, `events.csv.gz` (one row per event, lead and method) and
`thresholds.csv`.

### 5. Public benchmark for the paper (no TVA data)

TVA's forecasts cannot be published, so the paper compares against public baselines,
scored against USGS observations at the catalog's CAMELS-H sites, 00Z issues 2020-2025
(`scripts/benchmark.sh` runs every step):

- `download-usgs`: USGS instantaneous discharge from 2019-11 for every catalog gauge
  (TVA's stations are USGS gauges), six-hour means, as `benchmark_gauges.nc`.
- `download-nwm-forecasts`: NWM medium-range streamflow (deterministic member, 00Z) at
  the gauges' reaches (NWM feature_id = NHDPlus COMID) and the rain NWM was forced with,
  per reach. Our models and the LSTM use that rain, so the comparison with NWM holds the
  rain forecast fixed.
- `build-basin-series` and `train-lstm`: a lumped forecast LSTM (NeuralHydrology-style
  handoff encoder-decoder) on the same gauges, weather, observed-flow history, periods
  and loss.
- `evaluate-benchmark`: every method on the same issues, gauges and observations, with
  NWM rain (`+nwm`) and realized rain (`+aorc`, upper bound), plus persistence and
  climatology; `evaluate-events --reference nwm` for flood scores.

### 6. Like-for-like graph model vs LSTM (public data only)

`scripts/fair_comparison.sh` retrains stage 2 and the LSTM under one protocol:

- data: `build-public-gauges` (CAMELS-H before 2019-11, USGS after; no TVA series);
  `finetune-observed --public` reads no TVA archive;
- forecast rain in training: GEFSv12 reforecast (`download-gefs --source reforecast`)
  for 25% of samples, realized AORC otherwise; rain unknown for 10%;
- budget: the same number of usable (issue, gauge) training pairs (`--sample-budget`;
  stage 2 scores every gauge inside its subgraph, `--score-inside`);
- selection: 10 evaluations, 2019 00Z issues with GEFSv12 reforecast rain at every
  CAMELS-H gauge, one shared score (`training.selection`);
- three seeds each; then `evaluate-benchmark` on the public catalog, with NWM,
  GEFSv12 operational and realized rain.

### Other evaluations

```bash
# Per-reach skill against NWM on every reach (2019), with gauge-only flow history
CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN $PY -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m tva_flood evaluate-hydrology --array-cache $CACHE --split validation \
  --checkpoint $M/forecast_h128_v1/best.pt --out $M/forecast_h128_v1/hydrology_2019
# Rescore any checkpoint on the validation windows without training
... -m tva_flood train-forecast --array-cache $CACHE --evaluate-checkpoint CKPT --validation-windows 0 --out NEW
```

## Settings worth varying

- `--hidden`, `--temporal-layers`, `--graph-layers`, `--decoder-layers`,
  `--decoder-graph-layers`: capacity and spatial reach (the halo grows with graph layers).
- `--rain-hours`: the known-rain window (stage 1 and 2; evaluation takes each checkpoint's own).
- `--history-hide-fraction`, `--no-history-prob`: how often gauge history is missing.
- `--rain-error` (both stages): realistic forecast-rain errors on the training rain.
- Stage 2: `--under-weight`, `--csi-weight`, `--flood-quantile`, `--ema-decay`,
  `--qpf-fraction`, `--rain-known-fraction`, `--gauge-dropout`, `--raw-weight`,
  `--high-flow-weight`, `--encoder-lr-scale`, `--gauges-per-update`.
- `--precision bf16` lowers memory, if a larger model does not fit.
