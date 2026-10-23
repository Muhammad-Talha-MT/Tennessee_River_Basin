#!/usr/bin/env bash
# One forecast rule for every model: observed weather and flow before the 00Z issue, and
# after it only GEFSv12 control rain to 240 h (reforecast 2000-2018 to train, 2019 to
# select; operational from 2020-09-23 to test). One seed per model.
#
#   setsid -f nohup bash scripts/gefs_pipeline.sh </dev/null >>/data/tva_data/logs/gefs_pipeline.log 2>&1
#
# Models (in /data/tva_data/models):
#   gefs_hydro_A, gefs_hydro_B, gefs_hydro_B_f1..5  new graph-routing model (tva_flood.hydro), 4 GPUs
#   gefs_lstm, gefs_lstm_f1..5                       LSTM, tuned settings (graph_like)
# Evaluation:
#   gefs_benchmark, gefs_benchmark_f1..5             the LSTM at the gauges (evaluate-benchmark, CPU)
#   retrospective_comparison_gefs_nwm                the NWM forecast at every reach (CPU)
#   gefs_evaluation                                  the three protocols (tva_flood.hydro.evaluate)
#
# A step whose output holds DONE is skipped, so a rerun resumes. DRY_RUN=1 prints the commands.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH=$REPO/src OMP_NUM_THREADS=4
ROOT=/data/tva_data
M=$ROOT/models
LOGS=$ROOT/logs
TEST=$ROOT/processed/test_preparation_2020_2025
GAUGES=$ROOT/processed/gauges_public/gauges.nc
REFORECAST=$ROOT/processed/rain_forecasts/gefs_reforecast_240
OPERATIONAL=$ROOT/processed/rain_forecasts/gefs_operational_240
BUDGET=3300807  # (issue, gauge) pairs, as every earlier LSTM run
PY=$ROOT/conda_envs/tva_gnn/bin/python
GPUS=0,1,2,3
FOLDS=5
mkdir -p "$LOGS"

say() { echo "[$(date '+%F %T')] $*"; }
run() {  # run LOG command...
  local log=$1; shift
  if [ -n "${DRY_RUN:-}" ]; then echo "  $*  >> $log"; else "$@" >>"$log" 2>&1; fi
}
step() {  # step OUT LOG command...: run unless OUT/DONE; mark DONE or FAILED
  local out=$1 log=$2; shift 2
  [ -f "$out/DONE" ] && return 0
  rm -f "$out/FAILED"
  say "start $(basename "$out")"
  if run "$log" "$@"; then
    [ -n "${DRY_RUN:-}" ] || { mkdir -p "$out"; touch "$out/DONE"; }
    say "done $(basename "$out")"
  else
    mkdir -p "$out"; touch "$out/FAILED"
    say "FAILED $(basename "$out") (see $log)"
    return 1
  fi
}
wait_for() {  # wait until every named output is DONE; fail if one FAILED
  [ -n "${DRY_RUN:-}" ] && return 0
  local out
  for out in "$@"; do
    until [ -f "$out/DONE" ]; do
      [ -f "$out/FAILED" ] && { say "$(basename "$out") failed; skipping what needs it"; return 1; }
      sleep 120
    done
  done
}
suffix() { [ "$1" -gt 0 ] && echo "_f$1"; }
fold_args() { [ "$1" -gt 0 ] && echo "--holdout-fold $1 --folds $FOLDS"; }

# ---- models --------------------------------------------------------------------------
hydro() {  # hydro STAGE FOLD
  local stage=$1 fold=$2 name=gefs_hydro_$1 extra n
  [ "$stage" = B ] && name=gefs_hydro_B$(suffix "$fold")
  # A run of this output started by an earlier pipeline: let it finish and keep it.
  if [ -z "${DRY_RUN:-}" ] && [ ! -f "$M/$name/DONE" ] && pgrep -f -- "--out $M/$name --" >/dev/null; then
    say "waiting for the running $name"
    while pgrep -f -- "--out $M/$name --" >/dev/null; do sleep 60; done
    if grep -q "TRAINING COMPLETE" "$LOGS/$name.log"; then touch "$M/$name/DONE"; say "done $name"; fi
  fi
  n=$(echo "$GPUS" | tr ',' '\n' | wc -l)
  # Eight samples per update whatever the GPU count.
  local accumulate=$((8 / n))
  if [ "$stage" = A ]; then
    extra=(--updates 5000 --lr 5e-4 --warmup 500 --validate-every 1000)
  else
    extra=(--init "$M/gefs_hydro_A/best.pt" --fold "$fold" --updates 3000 --lr 3e-4 --warmup 100 --validate-every 500)
  fi
  step "$M/$name" "$LOGS/$name.log" env CUDA_VISIBLE_DEVICES="$GPUS" \
    "$PY" -W ignore::UserWarning -m torch.distributed.run --standalone --nproc_per_node "$n" \
    -m tva_flood.hydro.train --stage "$stage" --out "$M/$name" --accumulate "$accumulate" "${extra[@]}"
}

lstm() {  # lstm FOLD GPU
  local fold=$1 gpu=$2 name=gefs_lstm$(suffix "$1")
  [ -f "$M/$name/DONE" ] || rm -rf "$M/$name"
  # shellcheck disable=SC2046
  step "$M/$name" "$LOGS/$name.log" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" -u -m tva_flood train-lstm --out "$M/$name" --seed 1 --gauges "$GAUGES" \
    --forecast-rain "$REFORECAST" --forecast-fraction 1 --unknown-rain-prob 0 --rain-cut-prob 0 \
    --rain-hours 240 --validation-rain "$REFORECAST" --validation-gauges camelsh --validation-split \
    --sample-budget "$BUDGET" --evaluations 10 --min-count 100 \
    --batch-size 48 --lr 5e-4 --dropout 0 --balanced $(fold_args "$fold")
}

# ---- evaluation of the LSTM and NWM (CPU) -----------------------------------------------
benchmark() {  # benchmark FOLD: the LSTM at the gauges (that fold's only, if any)
  local fold=$1 s
  s=$(suffix "$1")
  [ -f "$M/gefs_benchmark$s/DONE" ] || rm -rf "$M/gefs_benchmark$s"
  # shellcheck disable=SC2046
  step "$M/gefs_benchmark$s" "$LOGS/gefs_benchmark$s.log" env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node 12 -m tva_flood evaluate-benchmark \
    --device cpu --preparation "$TEST" --array-cache "${TEST}_arrays" --gauges "$GAUGES" \
    --gefs "$OPERATIONAL" --out "$M/gefs_benchmark$s" --lstm "lstm=$M/gefs_lstm$s/best.pt" \
    $(fold_args "$fold")
}

nwm_reaches() {  # the NWM forecast against the retrospective at every reach
  local out=$M/retrospective_comparison_gefs_nwm k
  [ -f "$out/DONE" ] && return 0
  say "start NWM every-reach reference (CPU, 12 processes)"
  for k in $(seq 0 11); do
    run "$LOGS/retrospective_gefs_nwm_part$k.log" env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 \
      "$PY" scripts/retrospective_comparison.py compute --device cpu --every 2 --reference-only \
      --rain gefs --gefs "$OPERATIONAL" --tag gefs_nwm --shard "$k" --shards 12 &
  done
  wait
  step "$out" "$LOGS/retrospective_gefs_nwm.log" "$PY" scripts/retrospective_comparison.py merge --tag gefs_nwm
}

# ---- order -----------------------------------------------------------------------------
for k in 0 1 2 3 4 5; do lstm "$k" 0; done
(nwm_reaches; for k in 0 1 2 3 4 5; do benchmark "$k"; done) &
hydro A 0 || exit 1
for k in 0 1 2 3 4 5; do hydro B "$k" || exit 1; done
wait

say "evaluation, all four GPUs"
need=("$M/retrospective_comparison_gefs_nwm")
for k in 0 1 2 3 4 5; do need+=("$M/gefs_benchmark$(suffix "$k")" "$M/gefs_hydro_B$(suffix "$k")"); done
wait_for "${need[@]}" &&
  step "$M/gefs_evaluation" "$LOGS/gefs_evaluation.log" env CUDA_VISIBLE_DEVICES="$GPUS" \
    "$PY" -W ignore::UserWarning -m torch.distributed.run --standalone --nproc_per_node 4 \
    -m tva_flood.hydro.evaluate --out "$M/gefs_evaluation"
say "pipeline finished: $M/gefs_evaluation/report.md"
