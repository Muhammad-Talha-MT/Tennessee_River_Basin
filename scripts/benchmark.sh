#!/usr/bin/env bash
# Public benchmark for the paper (no TVA data in scoring): our graph models, a lumped
# forecast LSTM and NOAA NWM medium-range forecasts against USGS observations at the
# catalog's CAMELS-H sites, 00Z issues 2020-2025.
#
#   setsid -f nohup bash scripts/benchmark.sh </dev/null \
#     >>/data/tva_data/logs/benchmark.log 2>&1
#
# Steps (each skipped when its output exists; downloads resume by month):
#   1. USGS observed flow 2019-11 to 2025 for every catalog gauge   (~5 min)
#   2. basin-mean weather per gauge, training and test periods      (~5 min)
#   3. NWM medium-range streamflow at the gauges + its rain forcing (~4-10 h, network)
#   4. LSTM baseline, three seeds, one GPU each                     (~15 min)
#   5. benchmark of the whole-basin graph models (V2 = h128_v3, plus V4 if given) and
#      the LSTMs, then of V1 (needs its own run), 4 GPUs each       (~1 h each)
#   6. flood-event scores against NWM for both runs
# DRY_RUN=1 only prints the commands. V4=/path/to/best.pt adds a V4 checkpoint.
set -euo pipefail

ROOT=/data/tva_data
M=$ROOT/models
B=$ROOT/processed/benchmarks
LOGS=$ROOT/logs
TRAIN=$ROOT/processed/pretraining_full
TEST=$ROOT/processed/test_preparation_2020_2025
PY=$ROOT/conda_envs/tva_gnn/bin/python
PYDL=$ROOT/conda_envs/tva_download/bin/python
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export OMP_NUM_THREADS=1 PYTHONPATH=$REPO/src

say() { echo "[$(date '+%F %T')] $*"; }
run() {
  local log=$1; shift
  if [ -n "${DRY_RUN:-}" ]; then echo "  $*  >> $log"; else "$@" >>"$log" 2>&1; fi
}
mkdir -p "$LOGS" "$B"

if [ -f "$B/benchmark_gauges.nc" ]; then say "USGS observations already downloaded"; else
  say "1. USGS observed flow"
  run "$LOGS/benchmark_usgs.log" "$PY" -W ignore -m tva_flood download-usgs --workers 8
fi

for P in "$TRAIN" "$TEST"; do
  if [ -f "$P/basin_series.npz" ]; then continue; fi
  say "2. basin series for $(basename "$P")"
  run "$LOGS/benchmark_basin_series.log" env OMP_NUM_THREADS=8 "$PY" -W ignore -m tva_flood \
    build-basin-series --preparation "$P" --array-cache "${P}_arrays"
done

say "3. NWM medium-range forecasts (resumes by month)"
# A stalled connection fails its month; rerun until every month exists (up to 6 tries).
for TRY in 1 2 3 4 5 6; do
  run "$LOGS/benchmark_nwm.log" "$PYDL" -W ignore -m tva_flood download-nwm-forecasts --workers 32 && break
  say "NWM download attempt $TRY failed; retrying"
  [ "$TRY" = 6 ] && exit 1
done

say "4. LSTM baseline, seeds 1-3"
pids=()
for SEED in 1 2 3; do
  OUT=$M/lstm_benchmark_s$SEED
  if [ -f "$OUT/last.pt" ]; then continue; fi
  [ -d "$OUT" ] && { [ -n "${DRY_RUN:-}" ] || rm -rf "$OUT"; }
  run "$LOGS/lstm_benchmark_s$SEED.log" env CUDA_VISIBLE_DEVICES=$((SEED - 1)) \
    "$PY" -u -m tva_flood train-lstm --out "$OUT" --seed "$SEED" &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

bench() {
  local out=$1; shift
  if [ -f "$out/report.md" ]; then say "$(basename "$out") already done"; return; fi
  [ -d "$out" ] && { [ -n "${DRY_RUN:-}" ] || rm -rf "$out"; }
  say "5. $(basename "$out")"
  run "$LOGS/$(basename "$out").log" env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 -m tva_flood \
    evaluate-benchmark --preparation "$TEST" --array-cache "${TEST}_arrays" --out "$out" "$@"
}
WHOLE=(--checkpoint "v2=$M/finetune_h128_v3/best.pt")
[ -n "${V4:-}" ] && WHOLE+=(--checkpoint "v4=$V4")
bench "$M/benchmark_main" "${WHOLE[@]}" \
  --lstm "lstm_s1=$M/lstm_benchmark_s1/best.pt" --lstm "lstm_s2=$M/lstm_benchmark_s2/best.pt" \
  --lstm "lstm_s3=$M/lstm_benchmark_s3/best.pt"
bench "$M/benchmark_v1" --checkpoint "v1=$M/finetune_h128_v1/best.pt"
# GEFSv12 operational rain (download-gefs) as a second rain source, from 2020-09-23.
GEFS=$ROOT/processed/rain_forecasts/gefs_operational
if ls "$GEFS"/*.npz >/dev/null 2>&1; then
  bench "$M/benchmark_gefs" "${WHOLE[@]}" --gefs "$GEFS" \
    --lstm "lstm_s1=$M/lstm_benchmark_s1/best.pt" --lstm "lstm_s2=$M/lstm_benchmark_s2/best.pt" \
    --lstm "lstm_s3=$M/lstm_benchmark_s3/best.pt"
fi

for RUN in benchmark_main:v2 benchmark_v1:v1; do
  NAME=${RUN%%:*}; MODEL=${RUN#*:}
  if [ -f "$M/events_$NAME/report.md" ]; then continue; fi
  [ -d "$M/events_$NAME" ] && { [ -n "${DRY_RUN:-}" ] || rm -rf "$M/events_$NAME"; }
  say "6. flood events for $NAME"
  run "$LOGS/events_$NAME.log" "$PY" -m tva_flood evaluate-events --runs "$M/$NAME" \
    --gauges "$B/benchmark_gauges.nc" --reference nwm --candidate "$MODEL+nwm" \
    --out "$M/events_$NAME"
done
say "All done: $M/benchmark_main/report.md, $M/benchmark_v1/report.md, $M/events_benchmark_*/report.md"
