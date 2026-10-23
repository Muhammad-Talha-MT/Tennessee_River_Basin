#!/usr/bin/env bash
# Shared settings and commands of the public-data experiments (sourced, not run).
# Every model trains under one protocol: public catalog (CAMELS-H, then USGS), GEFSv12
# reforecast rain for 25% of samples, rain unknown for 10%, BUDGET usable (issue, gauge)
# pairs, 10 evaluations on 2019 00Z issues with GEFSv12 rain at the CAMELS-H gauges.

ROOT=/data/tva_data
M=$ROOT/models
LOGS=$ROOT/logs
TRAIN=$ROOT/processed/pretraining_full
TEST=$ROOT/processed/test_preparation_2020_2025
GAUGES=$ROOT/processed/gauges_public/gauges.nc
REFORECAST=$ROOT/processed/rain_forecasts/gefs_reforecast
OPERATIONAL=$ROOT/processed/rain_forecasts/gefs_operational
INIT=$M/forecast_h128_v3/best.pt  # architecture and normalization only (--from-scratch)
BUDGET=${BUDGET:-3300807}
PY=$ROOT/conda_envs/tva_gnn/bin/python
export OMP_NUM_THREADS=4
mkdir -p "$LOGS"

say() { echo "[$(date '+%F %T')] $*"; }
run() {
  local log=$1; shift
  if [ -n "${DRY_RUN:-}" ]; then echo "  $*  >> $log"; else "$@" >>"$log" 2>&1; fi
}
fresh() {  # remove an output folder that lacks its finished-marker file
  if [ -d "$1" ] && [ ! -f "$1/$2" ]; then
    say "Removing unfinished $1"
    [ -n "${DRY_RUN:-}" ] || rm -rf "$1"
  fi
}
finish() {  # mark an output done, or failed (so jobs waiting on it give up)
  local out=$1 status=$2
  [ -n "${DRY_RUN:-}" ] && return "$status"
  mkdir -p "$out"
  if [ "$status" = 0 ]; then rm -f "$out/FAILED"; touch "$out/DONE"; else touch "$out/FAILED"; fi
  return "$status"
}
wait_for() {  # wait until every named output is DONE; fail if one FAILED
  [ -n "${DRY_RUN:-}" ] && return 0
  local out
  for out in "$@"; do
    until [ -f "$out/DONE" ] || [ -f "$out/checkpoints/done" ]; do
      [ -f "$out/FAILED" ] && { say "$out failed; giving up on what needs it"; return 1; }
      sleep 120
    done
  done
}

# Graph model from scratch: train_graph NAME GPU SEED [extra finetune-observed arguments]
train_graph() {
  local name=$1 gpu=$2 seed=$3; shift 3
  local out=$M/$name
  if [ -f "$out/DONE" ] || [ -f "$out/checkpoints/done" ]; then return 0; fi
  [ -f "$out/last.pt" ] || fresh "$out" last.pt
  rm -f "$out/FAILED"
  say "train $name (GPU $gpu)"
  run "$LOGS/$name.log" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" -u -m tva_flood finetune-observed --array-cache "${TRAIN}_arrays" \
    --init "$INIT" --from-scratch --out "$out" --public --gauges "$GAUGES" \
    --forecast-rain "$REFORECAST" --forecast-fraction 0.25 --unknown-rain-prob 0.1 \
    --sample-budget "$BUDGET" --evaluations 10 --min-count 100 \
    --rain-known-fraction 1 1 --score-inside --amp --max-nodes 4000 --batch-size 3 \
    --lr 5e-4 --encoder-lr-scale 1 --warmup-updates 1000 \
    --validation-issues 0 --seed "$seed" "$@"
  finish "$out" $?
}

# LSTM: train_lstm NAME GPU SEED [extra train-lstm arguments, e.g. a tuning variant]
train_lstm() {
  local name=$1 gpu=$2 seed=$3; shift 3
  local out=$M/$name
  if [ -f "$out/DONE" ]; then return 0; fi
  fresh "$out" DONE
  say "train $name (GPU $gpu)"
  run "$LOGS/$name.log" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" -u -m tva_flood train-lstm --out "$out" --seed "$seed" --gauges "$GAUGES" \
    --forecast-rain "$REFORECAST" --forecast-fraction 0.25 --unknown-rain-prob 0.1 \
    --rain-cut-prob 0 --validation-rain "$REFORECAST" --validation-gauges camelsh \
    --validation-split --sample-budget "$BUDGET" --evaluations 10 --min-count 100 "$@"
  finish "$out" $?
}

# Benchmark 2020-2025: bench OUT GPUS [evaluate-benchmark arguments]; GPUS like "0,1,2,3".
bench() {
  local out=$1 gpus=$2; shift 2
  if [ -f "$out/report.md" ]; then return 0; fi
  fresh "$out" report.md
  local count
  count=$(echo "$gpus" | tr ',' '\n' | wc -l)
  say "benchmark $(basename "$out") (GPUs $gpus)"
  run "$LOGS/$(basename "$out").log" env CUDA_VISIBLE_DEVICES="$gpus" \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node="$count" \
    -m tva_flood evaluate-benchmark \
    --preparation "$TEST" --array-cache "${TEST}_arrays" --gauges "$GAUGES" --out "$out" "$@"
}
