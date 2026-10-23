#!/usr/bin/env bash
# Which parts of the river-network design carry the ungauged gain? The ablation ladder of
# routing_ablation.sh (no routing; also no basin means; also no message passing), trained
# with each held-out fold withheld (one seed), then scored on that fold's gauges with
# their flow history hidden, beside the full model of experiments.sh.
#
#   setsid -f nohup bash scripts/ungauged_ablation.sh </dev/null \
#     >>/data/tva_data/logs/ungauged_ablation.log 2>&1
#
# One job per GPU, ordered by fold so folds 1-3 finish first (a 20 h job each). A GPU
# takes a job once it is free; GPU 2 also waits for experiments.sh's routing benchmark.
# Rerunning resumes. Outputs: scratch_graph_heldout_{no_routing,no_basin,no_network}_f{k},
# heldout_ablation_benchmark_f{k}.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH=$REPO/src
source scripts/train_lib.sh
FOLDS=${FOLDS:-5}
GPUS=(${GPUS:-0 1 2 3})
VARIANTS=("no_routing routing" "no_basin routing basin" "no_network routing basin graph")

wait_gpu() {  # wait until nothing else uses the GPU
  until [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 3000 ]; do sleep 60; done
}

fold_trains() {
  local fold=$1 v parts
  for v in "${VARIANTS[@]}"; do
    read -r -a parts <<<"$v"
    echo "train_graph scratch_graph_heldout_${parts[0]}_f$fold \$GPU 1 --holdout-fold $fold --folds $FOLDS --ablate ${parts[*]:1}"
  done
}

ablation_bench() {  # ablation_bench FOLD GPU
  local fold=$1 gpu=$2 v parts args=()
  for v in "${VARIANTS[@]}"; do
    read -r -a parts <<<"$v"
    wait_for "$M/scratch_graph_heldout_${parts[0]}_f$fold" || return 1
    args+=(--checkpoint "${parts[0]}=$M/scratch_graph_heldout_${parts[0]}_f$fold/best.pt")
  done
  bench "$M/heldout_ablation_benchmark_f$fold" "$gpu" --holdout-fold "$fold" --folds "$FOLDS" "${args[@]}"
}

QUEUE=$LOGS/ungauged_ablation_queue
next_job() {
  (
    flock 9
    local n line
    n=$(cat "$QUEUE/counter")
    line=$(sed -n "$((n + 1))p" "$QUEUE/jobs")
    [ -n "$line" ] || exit 1
    echo $((n + 1)) >"$QUEUE/counter"
    echo "$line"
  ) 9>"$QUEUE/lock"
}
worker() {
  local GPU=$1 job
  while true; do
    [ "$GPU" = 2 ] && [ -z "${DRY_RUN:-}" ] && until [ -f "$M/routing_benchmark/report.md" ]; do sleep 300; done
    [ -n "${DRY_RUN:-}" ] || wait_gpu "$GPU"
    job=$(next_job) || break
    eval "$job" || say "job failed (GPU $GPU): $job"
  done
}

rm -rf "$QUEUE"
mkdir -p "$QUEUE"
echo 0 >"$QUEUE/counter"
{
  for fold in $(seq 1 "$FOLDS"); do
    fold_trains "$fold"
    [ "$fold" -gt 1 ] && echo "ablation_bench $((fold - 1)) \$GPU"
  done
  echo "ablation_bench $FOLDS \$GPU"
} >"$QUEUE/jobs"
say "Queue ($(wc -l <"$QUEUE/jobs") jobs on GPUs ${GPUS[*]}):"
sed 's/^/    /' "$QUEUE/jobs"
workers=()
for gpu in "${GPUS[@]}"; do
  worker "$gpu" &
  workers+=($!)
done
for pid in "${workers[@]}"; do wait "$pid"; done
say "All done: $M/heldout_ablation_benchmark_f*"
