# TVA Flood

Ten-day river forecasting for every reach of the Tennessee River network
(58,834 NHDPlus/NWM reaches, six-hour steps, 30-day history, 40-step horizon),
scored against observed flow and TVA's own forecasts. The package covers downloads,
data preparation, forecast-mode training and evaluation. Data and experiment outputs
live outside the repository under `/data/tva_data`.

Forecast mode: at issue time the model sees weather up to then, observed flow at
gauges only, and future rain only where a forecast exists (TVA's, to about 4 days).
Details, commands and settings are in [docs/FORECAST.md](docs/FORECAST.md).

## Install

```bash
conda activate /data/tva_data/conda_envs/tva_gnn
cd /home/talhamuh/tva_flood/tva_flood_clean
python -m pip install --no-deps -e .
python -m tva_flood --help
```

`--no-deps` keeps the existing CUDA PyTorch. For a fresh environment install a
CUDA PyTorch build first, then `python -m pip install -e '.[download,dev]'`.

## Workflow

| Stage | Command | Details |
|---|---|---|
| Downloads, graph, aggregation, statics, normalization | `download-*`, `inventory`, `build-graph`, `build-weights`, `aggregate`, `prepare-statics`, `prepare` | [PIPELINE.md](docs/PIPELINE.md) |
| Decode months once into memory-mapped arrays | `build-array-cache` | [FORECAST.md](docs/FORECAST.md) |
| Observed-flow gauge catalog (TVA and all CAMELS-H gauges) | `build-gauges` | [FORECAST.md](docs/FORECAST.md) |
| Stage 1: forecast training on NWM targets, gauge-only flow history | `train-forecast --history-gauges` | [FORECAST.md](docs/FORECAST.md) |
| Stage 2: fine-tune on observed flow at every gauge | `finetune-observed` | [FORECAST.md](docs/FORECAST.md) |
| Held-out test period (2020 to 2025) | `prepare-test`, then `evaluate-observed --split test` | [FORECAST.md](docs/FORECAST.md) |
| Per-reach NSE, KGE, RMSE, peak error against NWM | `evaluate-hydrology` | [FORECAST.md](docs/FORECAST.md) |
| Optional masked-reconstruction pretraining | `ssl-pretrain` | [FORECAST.md](docs/FORECAST.md) |

## Runs kept in `/data/tva_data/models`

| Folder | What it is |
|---|---|
| `encoder_ssl_v1`, `encoder_forecast_ssl_v1`, `encoder_forecast_scratch_v1` | The SSL experiment: pretrained encoder, and forecast models initialized from it or from scratch (equal within 2%) |
| `encoder_forecast_precip_v1`, `encoder_finetune_observed_v1` | The first forecast-mode baseline (hidden 32, 36k parameters, NWM flow history) |
| `observed_eval_test_gauge_only_2020` ... `2025` | That baseline against TVA, per test year |

## Layout

```text
src/tva_flood/
  cli.py          command dispatch
  common.py       atomic JSON and hashes
  downloads/      boundaries, NHDPlus, AORC/NWM, CAMELS-H
  data/           inventory, graph, weights, aggregation, statics, preparation, test preparation,
                  monthly loader and array cache, encoder windows and masks, river patches,
                  gauge catalog, TVA archive and basin rain, forecast-mode inputs
  models/         sparse graph supports; encoder, reconstruction and forecast heads
  training/       shared distributed loop, ssl-pretrain, train-forecast, finetune-observed
  evaluation/     metrics, hydrology (NWM targets), observed (TVA gauges)
tests/            offline tests on small synthetic data
docs/             PIPELINE.md, FORECAST.md, DEVELOPMENT.md
```
