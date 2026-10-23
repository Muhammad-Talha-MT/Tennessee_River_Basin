#!/usr/bin/env bash
# Like-for-like comparison of the whole-basin graph model and the lumped LSTM, public data only
# (for the paper). Both models get the same:
#   - training data: public catalog (CAMELS-H before 2019-11, USGS after; no TVA series),
#     targets ending before 2019, AORC weather;
#   - forecast rain in training: GEFSv12 reforecast for 25% of samples (00Z issues), realized
#     AORC otherwise, rain unknown for 10%, known to 96 h;
#   - training budget: BUDGET usable (issue, gauge) pairs (default 3,300,807 = one pass over
#     every training pair);
#   - checkpoint selection: 10 evaluations, 2019 00Z issues with GEFSv12 reforecast rain, all
#     CAMELS-H gauges, mean over 6-120 h of median NSE (training.selection);
#   - seeds 1, 2, 3; the same loss.
# The graph model starts from the V2 stage-1 checkpoint (NWM retrospective, no TVA data).
#
#   setsid -f nohup bash scripts/fair_comparison.sh </dev/null \
#     >>/data/tva_data/logs/fair_comparison.log 2>&1
#
# Steps (each skipped when its output exists):
#   1. public gauge catalog                                        (~1 min)
#   2. graph stage 2, seeds 1-3 on GPUs 0-2, and the LSTM seeds in turn on GPU 3
#   3. benchmark 2020-2025 with NWM rain and realized rain; then 2020-09-23 to 2025 with GEFS
#      operational rain too, 4 GPUs                                 (~2-3 h each)
#   4. flood-event scores against NWM
# DRY_RUN=1 only prints the commands. BUDGET=<pairs> changes the training budget.
set -euo pipefail

ROOT=/data/tva_data
M=$ROOT/models
LOGS=$ROOT/logs
TRAIN=$ROOT/processed/pretraining_full
TEST=$ROOT/processed/test_preparation_2020_2025
GAUGES=$ROOT/processed/gauges_public/gauges.nc
REFORECAST=$ROOT/processed/rain_forecasts/gefs_reforecast
OPERATIONAL=$ROOT/processed/rain_forecasts/gefs_operational
INIT=$M/forecast_h128_v3/best.pt
BUDGET=${BUDGET:-3300807}
PY=$ROOT/conda_envs/tva_gnn/bin/python
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export OMP_NUM_THREADS=4 PYTHONPATH=$REPO/src

say() { echo "[$(date '+%F %T')] $*"; }
run() {
  local log=$1; shift
  if [ -n "${DRY_RUN:-}" ]; then echo "  $*  >> $log"; else "$@" >>"$log" 2>&1; fi
}
fresh() {  # remove an unfinished output folder
  if [ -d "$1" ] && [ ! -f "$1/$2" ]; then
    say "Removing unfinished $1"
    [ -n "${DRY_RUN:-}" ] || rm -rf "$1"
  fi
}
mkdir -p "$LOGS"

if [ ! -f "$GAUGES" ]; then
  say "1. public gauge catalog"
  run "$LOGS/fair_gauges.log" "$PY" -W ignore -m tva_flood build-public-gauges
fi

RAIN=(--forecast-rain "$REFORECAST" --forecast-fraction 0.25 --unknown-rain-prob 0.1)
SELECT=(--sample-budget "$BUDGET" --evaluations 10 --min-count 100)

say "2. graph stage 2 (seeds 1-3, GPUs 0-2) and LSTM (seeds 1-3, GPU 3); budget $BUDGET pairs"
pids=()
for SEED in 1 2 3; do
  OUT=$M/fair_graph_s$SEED
  if [ -f "$OUT/checkpoints/done" ]; then continue; fi
  # finetune-observed resumes from last.pt; a run without it starts over.
  [ -f "$OUT/last.pt" ] || fresh "$OUT" last.pt
  (
    run "$LOGS/fair_graph_s$SEED.log" env CUDA_VISIBLE_DEVICES=$((SEED - 1)) \
      "$PY" -u -m tva_flood finetune-observed --array-cache "${TRAIN}_arrays" \
      --init "$INIT" --out "$OUT" --public --gauges "$GAUGES" "${RAIN[@]}" "${SELECT[@]}" \
      --rain-known-fraction 1 1 --score-inside --amp --max-nodes 4000 --batch-size 3 \
      --validation-issues 0 --seed "$SEED"
    [ -n "${DRY_RUN:-}" ] || touch "$OUT/checkpoints/done"
  ) &
  pids+=($!)
done
(
  for SEED in 1 2 3; do
    OUT=$M/fair_lstm_s$SEED
    if [ -f "$OUT/last.pt" ]; then continue; fi
    fresh "$OUT" last.pt
    run "$LOGS/fair_lstm_s$SEED.log" env CUDA_VISIBLE_DEVICES=3 \
      "$PY" -u -m tva_flood train-lstm --out "$OUT" --seed "$SEED" --gauges "$GAUGES" \
      "${RAIN[@]}" "${SELECT[@]}" --rain-cut-prob 0 --validation-rain "$REFORECAST" \
      --validation-gauges camelsh
  done
) &
pids+=($!)
for pid in "${pids[@]}"; do wait "$pid"; done

# A from-scratch graph run (scratch_graph.sh) may still train on a GPU this needs.
if [ -z "${DRY_RUN:-}" ]; then
  while pgrep -f "[f]inetune-observed.*scratch_graph" >/dev/null; do sleep 300; done
fi
MODELS=()
for SEED in 1 2 3; do
  MODELS+=(--checkpoint "graph_s$SEED=$M/fair_graph_s$SEED/best.pt")
  MODELS+=(--lstm "lstm_s$SEED=$M/fair_lstm_s$SEED/best.pt")
done
bench() {
  local out=$1; shift
  if [ -f "$out/report.md" ]; then say "$(basename "$out") already done"; return; fi
  fresh "$out" report.md
  say "3. $(basename "$out")"
  run "$LOGS/$(basename "$out").log" env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 -m tva_flood \
    evaluate-benchmark --preparation "$TEST" --array-cache "${TEST}_arrays" --gauges "$GAUGES" \
    --out "$out" "${MODELS[@]}" "$@"
}
bench "$M/fair_benchmark"
bench "$M/fair_benchmark_gefs" --gefs "$OPERATIONAL"

for NAME in fair_benchmark fair_benchmark_gefs; do
  if [ -f "$M/events_$NAME/report.md" ]; then continue; fi
  fresh "$M/events_$NAME" report.md
  say "4. flood events for $NAME"
  run "$LOGS/events_$NAME.log" "$PY" -m tva_flood evaluate-events --runs "$M/$NAME" \
    --gauges "$GAUGES" --reference nwm --candidate "graph_s1+nwm" --out "$M/events_$NAME"
done
say "All done: $M/fair_benchmark/report.md, $M/fair_benchmark_gefs/report.md"
