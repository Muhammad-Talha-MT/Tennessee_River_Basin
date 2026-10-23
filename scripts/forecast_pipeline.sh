#!/usr/bin/env bash
# Whole-basin river model (h128_v3): routing table, stage 1 (NWM targets),
# stage 2 (observed flow), the 2020-2025 test years and their flood-event scores.
#
#   setsid -f nohup bash scripts/forecast_pipeline.sh </dev/null \
#     >>/data/tva_data/logs/forecast_pipeline_h128_v3.log 2>&1
#
# Safe to rerun: a stopped training run resumes exactly where it stopped, a finished
# one returns at once, and a test year with a report is skipped. Stops at the first
# failure. NAME changes every output folder; DRY_RUN=1 only prints the commands.
#
# v3 against h128_v1 (same data, rain and stage-2 loss; only the river network changes):
# - each reach sees its whole upstream basin, not 6 hops: basin-mean weather history
#   (--basin-inputs) and future rain (--basin-rain), and every upstream reach's decoder
#   state routed to it with a learned travel time per river-distance bin (--routing);
# - graph blocks pass messages from upstream only and attend to their neighbours
#   (--direction upstream --aggregation attention), so training uses whole basins;
# - stage 2 caps each update at 4,000 reaches (whole basins are large), runs 6,000
#   updates, and validates on TVA basins only (all-gauge whole-basin graphs are slow).
# Earlier runs: h128_v1 = this script's stage 1 without the five options above and
# stage 2 with 4,000 updates, 32 gauges x 8 issues; h128_v2 added rain errors and
# flood-weighted stage-2 losses (both scored worse). See their run_config.json files.
set -euo pipefail

NAME=${NAME:-h128_v3}
ROOT=/data/tva_data
M=$ROOT/models
LOGS=$ROOT/logs
CACHE=$ROOT/processed/pretraining_full_arrays
G=$ROOT/processed/gauges/gauges.nc
TEST=$ROOT/processed/test_preparation_2020_2025
PY=$ROOT/conda_envs/tva_gnn/bin/python
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export OMP_NUM_THREADS=1 PYTHONPATH=$REPO/src

RIVER="--direction upstream --aggregation attention --basin-inputs --routing --basin-rain"

say() { echo "[$(date '+%F %T')] $*"; }
# run LOG COMMAND...: appends the command's output to LOG (or just prints the command).
run() {
  local log=$1; shift
  if [ -n "${DRY_RUN:-}" ]; then echo "  $*  >> $log"; else "$@" >>"$log" 2>&1; fi
}

for need in "$CACHE/array_cache.json" "$G" "$TEST/dataset_manifest.json" "$TEST"_arrays/array_cache.json; do
  [ -e "$need" ] || { say "Missing $need"; exit 1; }
done
mkdir -p "$LOGS"

if ls "$ROOT"/processed/river_routing/*.npz >/dev/null 2>&1; then
  say "River routing table already built"
else
  say "River routing table (upstream basins, area weights, distances)"
  run "$LOGS/build_routing.log" "$PY" -m tva_flood build-routing
fi

say "Stage 1: forecast training on NWM targets, whole basins (4 GPUs)"
# shellcheck disable=SC2086  # RIVER is a word list
run "$LOGS/forecast_$NAME.log" env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  "$PY" -u -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m tva_flood train-forecast --array-cache "$CACHE" --history-gauges "$G" \
  --hidden 128 --batch-size 2 --lr 5e-4 --rain-hours 96 $RIVER \
  --out "$M/forecast_$NAME" --updates 10000 --shuffle global

say "Stage 2: fine-tuning on observed flow (1 GPU)"
run "$LOGS/finetune_$NAME.log" env CUDA_VISIBLE_DEVICES=0 \
  "$PY" -u -m tva_flood finetune-observed --array-cache "$CACHE" \
  --init "$M/forecast_$NAME/best.pt" --out "$M/finetune_$NAME" \
  --updates 6000 --eval-every 500 --max-nodes 4000 --batch-size 3 \
  --validation-gauges tva --validation-issues 250

for YEAR in 2020 2021 2022 2023 2024 2025; do
  OUT=$M/test_${NAME}_$YEAR
  if [ -f "$OUT/report.md" ]; then say "Test $YEAR already done"; continue; fi
  if [ -d "$OUT" ]; then
    say "Removing unfinished $OUT"
    [ -n "${DRY_RUN:-}" ] || rm -rf "$OUT"
  fi
  say "Test $YEAR"
  run "$LOGS/test_${NAME}_$YEAR.log" env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m tva_flood evaluate-observed --split test --preparation "$TEST" --array-cache "${TEST}_arrays" \
    --issues-from "$YEAR-01-01" --issues-until "$((YEAR + 1))-01-01" \
    --checkpoint "tuned=$M/finetune_$NAME/best.pt" --out "$OUT"
done
EVENTS=$M/event_scores_$NAME
if [ -f "$EVENTS/report.md" ]; then
  say "Event scores already done"
else
  [ -d "$EVENTS" ] && { say "Removing unfinished $EVENTS"; [ -n "${DRY_RUN:-}" ] || rm -rf "$EVENTS"; }
  say "Event scores (flood thresholds and observed peaks)"
  run "$LOGS/event_scores_$NAME.log" "$PY" -m tva_flood evaluate-events \
    --runs "$M"/test_${NAME}_20{20,21,22,23,24,25} --out "$EVENTS"
fi
say "All done. Results: $M/test_${NAME}_YYYY/report.md and $EVENTS/report.md"
