# Development

```bash
python -m pip install -e '.[dev]'
OMP_NUM_THREADS=1 python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

Use normal package imports. Do not add numbered executable scripts, dynamic
file imports, duplicate model definitions, or hardcoded repository paths.

Ownership:

- `data/schema.py`: temporal dimensions and weather ordering.
- `data/transforms.py`: precipitation transform, shared by fitting and loading.
- `data/dataset.py`: prepared-data checks, split indexing, COMID-aligned monthly cache, `lineage`.
- `data/prepare_test.py`: test-period preparation sharing the source's normalization.
- `data/arrays.py`: memory-mapped monthly array cache (`build-array-cache`).
- `data/ssl.py`: 18-channel encoder windows, future rain, history and future masks.
- `data/subgraph.py`: connected river patches with an exact halo.
- `data/tva.py`: TVA archive (gauges, observed flow, issued forecasts), basin ownership and rain.
- `data/gauges.py`: observed-flow gauge catalog (TVA and CAMELS-H) and its loader.
- `data/inputs.py`: forecast-mode inputs on gauge subgraphs (history, rain specs, targets).
- `models/sparse.py`: directed sparse graph supports.
- `models/ssl.py`: graph-size-independent encoder, reconstruction and forecast heads.
- `training/encoder_loop.py`: shared distributed loop for `ssl-pretrain`/`train-forecast`.
- `training/forecast.py`: forecast task (gauge-only history, rain window, future-rain masks).
- `training/observed_finetune.py`: fine-tuning on observed flow at every catalog gauge.
- `training/runtime.py`: RNG, atomic checkpoint, window sampling and reduction helpers.
- `evaluation/metrics.py`: additive masked statistics.
- `evaluation/hydrology.py`: per-reach hydrologic skill against NWM targets.
- `evaluation/observed.py`: skill at TVA gauges against observed flow and TVA's forecasts.

Download providers remain separate because their formats and resume semantics differ.

## Possible next steps

- Rain history from TVA's observed basin precipitation instead of AORC (AORC is
  published with a delay, so TVA's observations are what operation would use).
- Future rain for days 4 to 10 from GFS or GEFS, over every gauge basin.
- Real-time USGS flow beyond 2024 for the CAMELS-H gauges (their archive ends 2024-12-31).
- Rain-forecast error augmentation for the training proxy (AORC) so it resembles TVA's forecast.
- A calibrated spread (ensemble or quantile output), not only a point forecast.
