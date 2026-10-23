#!/usr/bin/env bash
# Was the river routing helpful? An ablation ladder of the whole-basin graph model, every
# variant trained from scratch under the protocol of train_lib.sh (as scratch_graph_s1).
# Training draws do not depend on the model, so all variants see the same (issue, gauge)
# samples. Each step removes one more way the model uses the river network:
#
#   full         scratch_graph_s1 and a second seed, scratch_graph_s2
#   no_routing   without the decoder's travel-time routing block
#   no_basin     also without basin-mean weather history and basin-mean forecast rain
#   no_network   also without message passing: each reach sees only its own catchment
#   knockout     the trained full model with its routing block's output removed at test time
#
#   setsid -f nohup bash scripts/routing_ablation.sh </dev/null \
#     >>/data/tva_data/logs/routing_ablation.log 2>&1
#
# Steps (each skipped when its output exists; experiments.sh trains the same runs in its
# GPU queue, after which this script only benchmarks and plots):
#   1. train no_routing, no_basin, no_network and full seed 2 on GPUs 0-3   (~20 h)
#   2. benchmark every variant, 2020-2025, NWM and realized rain, 4 GPUs    (GPU 2 only, so the other GPUs stay with ungauged_ablation.sh)
#   3. figures and a report: /data/tva_data/figures/routing_ablation
# DRY_RUN=1 only prints the commands.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH=$REPO/src
source scripts/train_lib.sh
FIGURES=$ROOT/figures/routing_ablation

# The ablation jobs (also used by experiments.sh): NAME SEED [--ablate parts]
ABLATIONS=(
  "scratch_graph_no_routing_s1 1 --ablate routing"
  "scratch_graph_no_basin_s1 1 --ablate routing basin"
  "scratch_graph_no_network_s1 1 --ablate routing basin graph"
  "scratch_graph_s2 2"
)

main() {
  [ -f "$M/scratch_graph_s1/best.pt" ] || { say "Run scratch_graph.sh first (full model, seed 1)"; exit 1; }
  say "1. ablations and full seed 2; budget $BUDGET pairs"
  local pids=() gpu=0 job
  for job in "${ABLATIONS[@]}"; do
    read -r -a parts <<<"$job"
    train_graph "${parts[0]}" "$gpu" "${parts[1]}" "${parts[@]:2}" &
    pids+=($!)
    gpu=$((gpu + 1))
  done
  for pid in "${pids[@]}"; do wait "$pid" || { say "a training run failed"; exit 1; }; done

  say "2. routing knockout checkpoints and benchmark"
  run "$LOGS/routing_benchmark.log" "$PY" scripts/routing_analysis.py knockout \
    --checkpoint "$M/scratch_graph_s1/best.pt" --out "$M/routing_knockout/scratch_graph_s1.pt"
  bench "$M/routing_benchmark" 2 \
    --checkpoint "full_s1=$M/scratch_graph_s1/best.pt" \
    --checkpoint "full_s2=$M/scratch_graph_s2/best.pt" \
    --checkpoint "knockout_s1=$M/routing_knockout/scratch_graph_s1.pt" \
    --checkpoint "no_routing=$M/scratch_graph_no_routing_s1/best.pt" \
    --checkpoint "no_basin=$M/scratch_graph_no_basin_s1/best.pt" \
    --checkpoint "no_network=$M/scratch_graph_no_network_s1/best.pt" || exit 1

  say "3. figures"
  run "$LOGS/routing_benchmark.log" "$PY" scripts/routing_analysis.py plots --out "$FIGURES"
  say "All done: $FIGURES/report.md"
}

# Sourced by experiments.sh for ABLATIONS only.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then main; fi
