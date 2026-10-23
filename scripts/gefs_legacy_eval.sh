#!/usr/bin/env bash
# The earlier graph model V2 under the strict forecast rule (observed weather and flow before
# the 00Z issue, GEFSv12 operational rain after it, to the 96 h the model was trained for), on
# the same test issues as the new model. Its gauged scores with GEFS rain already exist
# (models/fair_benchmark_gefs, scratch_benchmark_gefs_s1). This adds what was missing:
#   gefs_heldout_v2_f1..5                       ungauged: the from-scratch fold models at their held-out gauges
#   retrospective_comparison_gefs_v2s[_f1..5]   every reach against the NWM retrospective (composite of the
#                                               all-gauge model and the five fold models)
# Three GPU lanes (0-2); a step whose output holds DONE is skipped. DRY_RUN=1 prints commands.
#
#   setsid -f nohup bash scripts/gefs_legacy_eval.sh </dev/null >>/data/tva_data/logs/gefs_legacy_eval.log 2>&1
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH=$REPO/src OMP_NUM_THREADS=4
ROOT=/data/tva_data
M=$ROOT/models
LOGS=$ROOT/logs
TEST=$ROOT/processed/test_preparation_2020_2025
GAUGES=$ROOT/processed/gauges_public/gauges.nc
OPERATIONAL=$ROOT/processed/rain_forecasts/gefs_operational  # to 96 h: the rain V2 was trained with
PY=$ROOT/conda_envs/tva_gnn/bin/python
FOLDS=5

say() { echo "[$(date '+%F %T')] $*"; }
run() { local log=$1; shift; if [ -n "${DRY_RUN:-}" ]; then echo "  $*  >> $log"; else "$@" >>"$log" 2>&1; fi; }
step() {  # step OUT LOG command...
  local out=$1 log=$2; shift 2
  [ -f "$out/DONE" ] && return 0
  say "start $(basename "$out")"
  [ -n "${DRY_RUN:-}" ] || rm -rf "$out"
  if run "$log" "$@"; then
    [ -n "${DRY_RUN:-}" ] || { mkdir -p "$out"; touch "$out/DONE"; }
    say "done $(basename "$out")"
  else
    say "FAILED $(basename "$out") (see $log)"; return 1
  fi
}

ungauged() {  # ungauged FOLD GPU
  local k=$1 gpu=$2
  step "$M/gefs_heldout_v2_f$k" "$LOGS/gefs_heldout_v2_f$k.log" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node 1 -m tva_flood evaluate-benchmark \
    --preparation "$TEST" --array-cache "${TEST}_arrays" --gauges "$GAUGES" --gefs "$OPERATIONAL" \
    --out "$M/gefs_heldout_v2_f$k" --checkpoint "graph=$M/scratch_graph_heldout_f$k/best.pt" \
    --holdout-fold "$k" --folds "$FOLDS"
}
reaches() {  # reaches FOLD GPU (0: the all-gauge model)
  local k=$1 gpu=$2 tag=gefs_v2s checkpoint=$M/scratch_graph_s1/best.pt
  if [ "$k" -gt 0 ]; then tag=gefs_v2s_f$k; checkpoint=$M/scratch_graph_heldout_f$k/best.pt; fi
  step "$M/retrospective_comparison_$tag" "$LOGS/retrospective_$tag.log" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" scripts/retrospective_comparison.py compute --every 2 --rain gefs --gefs "$OPERATIONAL" \
    --checkpoint "$checkpoint" --holdout-fold "$k" --tag "$tag"
}

lane() { local gpu=$1 job; shift; for job in "$@"; do ${job%%:*} "${job##*:}" "$gpu"; done; }

lane 0 ungauged:1 ungauged:4 reaches:0 reaches:3 &
lane 1 ungauged:2 ungauged:5 reaches:1 reaches:4 &
lane 2 ungauged:3 reaches:2 reaches:5 &
wait
say "V2 strict-rule evaluation finished"
