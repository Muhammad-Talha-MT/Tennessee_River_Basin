# Complete pipeline

Every command accepts `--help`. Run only missing stages when using an existing
data root. Dates below are exclusive stop dates. Download one extra month of
rainfall for the six-hour interval boundary, then aggregate only the desired
period. The forcing command now downloads **both AORC and NWM** by default. The former
`04_download_aorc_parallel.py` contained an NWM function but did not call it.
Use `--source aorc` or `--source nwm` to select only one provider.

```bash
# 1. Source geometry and NWM reach coordinate inventory
python -m tva_flood download-boundaries --root /data/tva_data
python -m tva_flood download-nhdplus --root /data/tva_data
python -m tva_flood inventory --root /data/tva_data

# 2. New datasets only: existing graph directories must be reused
python -m tva_flood build-graph --root /data/tva_data

# 3. Hourly meteorology and simulated streamflow
python -m tva_flood download-forcing --root /data/tva_data \
  --start 1979-02-01 --stop 2020-02-01 --workers 6

# 4. Catchment-area-weighted mapping and six-hour aggregation
python -m tva_flood build-weights --root /data/tva_data
python -m tva_flood aggregate --root /data/tva_data \
  --start 1979-02-01 --stop 2020-01-01 --workers 4

# 5. Static descriptors and training-only dynamic statistics
python -m tva_flood prepare-statics --root /data/tva_data --workers 2
python -m tva_flood prepare --root /data/tva_data \
  --train-start 1979-02-01 --train-stop 2019-01-01 \
  --validation-stop 2020-01-01 --precip-transform log1p

# 6. Array cache, SSL pretraining, forecasting and evaluation: docs/FORECAST.md
```

Optional CAMELS-H reference collection:

```bash
python -m tva_flood download-camelsh --root /data/tva_data
```

CAMELS-H basin attributes are not automatically assigned to every river reach.
The training core uses NHDPlus and StreamCat attributes. Dated NLCD snapshots
are downloaded separately and are not backfilled into historical inputs.

## Data contract

| Quantity | Contract |
|---|---|
| Time step | Six hours |
| History | 120 steps / 30 days |
| Forecast | 40 steps / 10 days |
| Dynamic input | `[batch, 120, nodes, 18]` |
| Static input | `[nodes, 48]`, separate from the time axis |
| Forecast and target | `[batch, 40, nodes]` |
| Weather channels 0–7 | APCP, temperature, specific humidity, surface pressure, downward shortwave, downward longwave, U wind, V wind |
| Channels 8–15 | Corresponding weather validity masks |
| Channel 16 | Training-standardized log1p historical streamflow |
| Channel 17 | Historical flow availability mask |
| Static 0–23 / 24–47 | 24 normalized static attributes in manifest order / their validity masks |

Precipitation is a six-hour sum in mm, log1p-transformed before standardization.
Other weather variables and NWM flow are six-hour means. A valid interval
requires six valid hourly values. Dynamic scalers use only the training split.
Static scalers use training network nodes; geographic holdout would require a
separate normalization policy. Missing values become zero in normalized space
with a separate mask; real zero observations remain valid.

The prepared manifest's `input_features=64` counts weather and static base
features with their masks; the historical-flow pair is added by the loader.

Default training targets end in 2018; validation is 2019. Forecast windows may
not cross target split boundaries. Historical context can precede the target
split. Later test years are not opened by pretraining or validation.

## Outputs

Existing directory conventions are preserved:

- `raw/`: remote source files.
- `metadata/`: inventory, download provenance and coverage reports.
- `processed/graph_candidate/`: node order, indexed edges and catchment polygons.
- `processed/`: mapping and six-hour products using their original paths.
- `processed/static/`: 24 core attributes and separate land-cover snapshots.
- `processed/pretraining_full/`: graph, manifest and normalization JSON.
- `models/<experiment>/`: configurations, best/last checkpoints, epoch reports.

The graph command is new because the archive omitted its producer. Its node
order follows `nwm_feature_index`; it retains internal NHDPlus connections,
collapses duplicate edges, repairs polygon geometries and reports missing
catchments. This is not evidence that NHDPlus routing equals NWM routing.
It refuses to replace an existing graph. Rebuilding into another directory
requires deliberate downstream configuration/data rebuilding; no automatic
migration of old graph indices is attempted.
