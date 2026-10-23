#!/usr/bin/env bash
# Public forecast data that replaces TVA's: rain forecasts and NWM streamflow forecasts.
#
#   setsid -f nohup bash scripts/download_forecasts.sh </dev/null \
#     >>/data/tva_data/logs/download_forecasts.log 2>&1
#
#   1. GEFSv12 reforecast rain, 00Z control, 2000-2019: training and validation  (~1 h)
#   2. GEFSv12 operational rain, 00Z control, 2020-09-23 to 2025: test             (~30 min)
#   3. NWM medium-range streamflow at the gauges and its GFS rain forcing, 2020-2025 (~4-10 h)
# All three resume month by month; rerun the same command after an interruption.
# Rain is six-hour totals per reach (mm) to 96 h. Outputs:
#   /data/tva_data/processed/rain_forecasts/gefs_reforecast/YYYY-MM.npz
#   /data/tva_data/processed/rain_forecasts/gefs_operational/YYYY-MM.npz
#   /data/tva_data/processed/benchmarks/nwm_medium_range/YYYY-MM.npz
set -euo pipefail

ROOT=/data/tva_data
PY=$ROOT/conda_envs/tva_download/bin/python
GNN=$ROOT/conda_envs/tva_gnn/bin/python
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export OMP_NUM_THREADS=1 PYTHONPATH=$REPO/src
say() { echo "[$(date '+%F %T')] $*"; }

if [ ! -f "$ROOT/processed/benchmarks/benchmark_gauges.nc" ]; then
  say "USGS observed flow (needed for the NWM gauge list)"
  "$GNN" -W ignore -m tva_flood download-usgs --workers 8
fi
say "1. GEFSv12 reforecast rain 2000-2019"
"$PY" -W ignore -m tva_flood download-gefs --source reforecast --workers 32
say "2. GEFSv12 operational rain 2020-09-23 to 2025"
"$PY" -W ignore -m tva_flood download-gefs --source operational --workers 32
say "3. NWM medium-range forecasts 2020-2025"
"$PY" -W ignore -m tva_flood download-nwm-forecasts --workers 32
say "All forecast data downloaded"
