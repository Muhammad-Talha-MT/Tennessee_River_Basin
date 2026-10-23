#!/usr/bin/env bash
# The whole-basin graph model trained on observed flow only (no NWM-trained stage 1), under
# the like-for-like protocol of fair_comparison.sh: same public catalog, GEFSv12 reforecast
# rain for 25% of samples, budget, selection, loss and model size. It sees every reach
# upstream of the drawn gauges and is scored at the gauges inside each update only. Only
# the optimizer differs from the fine-tuned runs, as nothing is pretrained: stage 1's
# learning rate (5e-4), the full rate on the encoder, and a longer warmup.
#
#   SEEDS="1" GPUS="3" setsid -f nohup bash scripts/scratch_graph.sh </dev/null \
#     >>/data/tva_data/logs/scratch_graph.log 2>&1
#
# Steps (each skipped when its output exists):
#   1. train each seed in SEEDS on the matching GPU in GPUS        (~19 h, in parallel)
#   2. after fair_comparison.sh has finished (GPUs free): benchmark 2020-2025 with NWM and
#      realized rain, then from 2020-09-23 with GEFSv12 operational rain too, 4 GPUs
# Rows are comparable with fair_benchmark*/report.md: same issues, gauges and observations.
# DRY_RUN=1 only prints the commands.
set -euo pipefail

ROOT=/data/tva_data
M=$ROOT/models
LOGS=$ROOT/logs
TRAIN=$ROOT/processed/pretraining_full
TEST=$ROOT/processed/test_preparation_2020_2025
GAUGES=$ROOT/processed/gauges_public/gauges.nc
REFORECAST=$ROOT/processed/rain_forecasts/gefs_reforecast
OPERATIONAL=$ROOT/processed/rain_forecasts/gefs_operational
INIT=$M/forecast_h128_v3/best.pt  # architecture and normalization only
BUDGET=${BUDGET:-3300807}
read -r -a SEEDS <<<"${SEEDS:-1}"
read -r -a GPUS <<<"${GPUS:-3}"
PY=$ROOT/conda_envs/tva_gnn/bin/python
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export OMP_NUM_THREADS=4 PYTHONPATH=$REPO/src

say() { echo "[$(date '+%F %T')] $*"; }
run() {
  local log=$1; shift
  if [ -n "${DRY_RUN:-}" ]; then echo "  $*  >> $log"; else "$@" >>"$log" 2>&1; fi
}
fresh() {
  if [ -d "$1" ] && [ ! -f "$1/$2" ]; then
    say "Removing unfinished $1"
    [ -n "${DRY_RUN:-}" ] || rm -rf "$1"
  fi
}
[ "${#SEEDS[@]}" = "${#GPUS[@]}" ] || { say "SEEDS and GPUS need the same length"; exit 1; }
mkdir -p "$LOGS"

say "1. graph from scratch, seeds ${SEEDS[*]} on GPUs ${GPUS[*]}; budget $BUDGET pairs"
pids=()
for i in "${!SEEDS[@]}"; do
  SEED=${SEEDS[$i]}
  OUT=$M/scratch_graph_s$SEED
  if [ -f "$OUT/checkpoints/done" ]; then continue; fi
  [ -f "$OUT/last.pt" ] || fresh "$OUT" last.pt
  (
    run "$LOGS/scratch_graph_s$SEED.log" env CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
      "$PY" -u -m tva_flood finetune-observed --array-cache "${TRAIN}_arrays" \
      --init "$INIT" --from-scratch --out "$OUT" --public --gauges "$GAUGES" \
      --forecast-rain "$REFORECAST" --forecast-fraction 0.25 --unknown-rain-prob 0.1 \
      --sample-budget "$BUDGET" --evaluations 10 --min-count 100 \
      --rain-known-fraction 1 1 --score-inside --amp --max-nodes 4000 --batch-size 3 \
      --lr 5e-4 --encoder-lr-scale 1 --warmup-updates 1000 \
      --validation-issues 0 --seed "$SEED"
    [ -n "${DRY_RUN:-}" ] || touch "$OUT/checkpoints/done"
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

if [ -z "${DRY_RUN:-}" ]; then
  while pgrep -f "[s]cripts/fair_comparison.sh" >/dev/null; do sleep 300; done
fi
MODELS=()
for SEED in "${SEEDS[@]}"; do
  MODELS+=(--checkpoint "scratch_s$SEED=$M/scratch_graph_s$SEED/best.pt")
done
TAG=$(IFS=_; echo "${SEEDS[*]}")
bench() {
  local out=$1; shift
  if [ -f "$out/report.md" ]; then say "$(basename "$out") already done"; return; fi
  fresh "$out" report.md
  say "2. $(basename "$out")"
  run "$LOGS/$(basename "$out").log" env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 -m tva_flood \
    evaluate-benchmark --preparation "$TEST" --array-cache "${TEST}_arrays" --gauges "$GAUGES" \
    --out "$out" "${MODELS[@]}" "$@"
}
bench "$M/scratch_benchmark_s$TAG"
bench "$M/scratch_benchmark_gefs_s$TAG" --gefs "$OPERATIONAL"
say "All done: $M/scratch_benchmark_s$TAG/report.md, $M/scratch_benchmark_gefs_s$TAG/report.md"
