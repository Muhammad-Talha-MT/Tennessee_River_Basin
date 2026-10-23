#!/usr/bin/env bash
# V4: two stage-2 changes for floods, tested on V3's stage 1 (h128_v3), one GPU each.
#
#   setsid -f nohup bash scripts/v4_stage2.sh </dev/null \
#     >>/data/tva_data/logs/v4_stage2.log 2>&1
#
# Variants (all otherwise exactly V3's stage 2):
#   flood  - 50% of training samples from issue times whose next 10 days reach the gauge's
#            99th-percentile flow (about 10% of uniform draws), i.e. floods 5x oversampled
#   select - 2018 held out of training and joined to 2019 for checkpoint selection, which
#            scores median NSE plus flood CSI (99th percentile) over both years
#   both   - flood and select together
#   seed   - V3's stage 2 with another seed: how much results move by chance alone
# Each variant then runs the six test years and the flood-event scores on its own GPU.
# Safe to rerun: finished steps are skipped or resume. DRY_RUN=1 only prints commands.
set -euo pipefail

ROOT=/data/tva_data
M=$ROOT/models
LOGS=$ROOT/logs
CACHE=$ROOT/processed/pretraining_full_arrays
TEST=$ROOT/processed/test_preparation_2020_2025
PY=$ROOT/conda_envs/tva_gnn/bin/python
INIT=$M/forecast_h128_v3/best.pt
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export OMP_NUM_THREADS=1 PYTHONPATH=$REPO/src

V3="--updates 6000 --eval-every 500 --max-nodes 4000 --batch-size 3 --validation-gauges tva"
FLOOD="--flood-fraction 0.5"
SELECT="--select-from 2018-01-01 --csi-weight 1 --validation-issues 400"
VARIANTS=(
  "flood:$FLOOD --validation-issues 250"
  "select:$SELECT"
  "both:$FLOOD $SELECT"
  "seed:--seed 7 --validation-issues 250"
)

say() { echo "[$(date '+%F %T')] $*"; }
run() {
  local log=$1; shift
  if [ -n "${DRY_RUN:-}" ]; then echo "  $*  >> $log"; else "$@" >>"$log" 2>&1; fi
}

[ -f "$INIT" ] || { say "Missing $INIT (V3 stage 1)"; exit 1; }
mkdir -p "$LOGS"

variant() {
  local gpu=$1 name=$2 options=$3
  local tag=h128_v4_$name
  say "$tag: stage 2 on GPU $gpu"
  # shellcheck disable=SC2086  # option lists
  run "$LOGS/finetune_$tag.log" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" -u -m tva_flood finetune-observed --array-cache "$CACHE" \
    --init "$INIT" --out "$M/finetune_$tag" $V3 $options
  for YEAR in 2020 2021 2022 2023 2024 2025; do
    local out=$M/test_${tag}_$YEAR
    if [ -f "$out/report.md" ]; then continue; fi
    [ -d "$out" ] && { [ -n "${DRY_RUN:-}" ] || rm -rf "$out"; }
    say "$tag: test $YEAR"
    run "$LOGS/test_${tag}_$YEAR.log" env CUDA_VISIBLE_DEVICES="$gpu" \
      "$PY" -m tva_flood evaluate-observed --split test --preparation "$TEST" \
      --array-cache "${TEST}_arrays" --issues-from "$YEAR-01-01" \
      --issues-until "$((YEAR + 1))-01-01" \
      --checkpoint "tuned=$M/finetune_$tag/best.pt" --out "$out"
  done
  local events=$M/event_scores_$tag
  if [ ! -f "$events/report.md" ]; then
    [ -d "$events" ] && { [ -n "${DRY_RUN:-}" ] || rm -rf "$events"; }
    run "$LOGS/event_scores_$tag.log" "$PY" -m tva_flood evaluate-events \
      --runs "$M"/test_${tag}_20{20,21,22,23,24,25} --out "$events"
  fi
  say "$tag: done"
}

pids=()
for i in "${!VARIANTS[@]}"; do
  variant "$i" "${VARIANTS[$i]%%:*}" "${VARIANTS[$i]#*:}" &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[ "$status" = 0 ] || { say "A variant failed; see $LOGS/*h128_v4*"; exit 1; }
say "All done. Results: $M/test_h128_v4_*_YYYY and $M/event_scores_h128_v4_*"
