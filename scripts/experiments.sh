#!/usr/bin/env bash
# Does the river network help? Three experiments, graph model from scratch only (the
# NWM-trained stage 1 added nothing), all under the protocol of train_lib.sh:
#
#   1. LSTM tuning check: the LSTM moved toward the graph model's training settings, one
#      at a time and together (3 seeds each); the variant with the best 2019 validation
#      score is the LSTM of the later experiments. Tested 2020-2025 for transparency.
#   2. Ungauged test: FOLDS folds of the CAMELS-H gauges (data.folds, ~25 gauges each).
#      For each fold the graph model (1 seed) and the selected LSTM (3 seeds) train with
#      that fold withheld entirely, and are scored on it with its flow history hidden.
#   3. Routing ablation (routing_ablation.sh): the graph model without its routing block,
#      basin means, and message passing; plus a second full seed.
#
#   setsid -f nohup bash scripts/experiments.sh </dev/null \
#     >>/data/tva_data/logs/experiments.log 2>&1
#
# Graph-model jobs run in one queue, one job per GPU (folds, then fold benchmarks, then
# ablations); LSTM runs share the GPUs alongside. Rerunning resumes: finished runs are
# skipped, graph runs resume from last.pt. FOLDS=4 (25% held out) saves one ~20 h round.
# DRY_RUN=1 only prints the commands. Outputs: /data/tva_data/figures/{lstm_tuning,
# ungauged,routing_ablation}.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH=$REPO/src
source scripts/train_lib.sh
source scripts/routing_ablation.sh  # ABLATIONS only
FOLDS=${FOLDS:-5}
GPUS=(0 1 2 3)
LSTM_PARALLEL=4

# name, then train-lstm arguments that differ from the protocol's LSTM
VARIANTS=(
  "base"
  "small_batch --batch-size 48"
  "lr_5e-4 --lr 5e-4"
  "no_dropout --dropout 0"
  "balanced --balanced"
  "graph_like --batch-size 48 --lr 5e-4 --dropout 0 --balanced"
)

throttle() {  # keep at most LSTM_PARALLEL background LSTM runs
  while [ "$(jobs -rp | wc -l)" -ge "$1" ]; do wait -n; done
}

lstm_stage() {
  say "1. LSTM tuning: ${#VARIANTS[@]} variants x 3 seeds"
  local i=0 variant parts seed args=()
  for variant in "${VARIANTS[@]}"; do
    read -r -a parts <<<"$variant"
    for seed in 1 2 3; do
      throttle "$LSTM_PARALLEL"
      train_lstm "lstm_tune_${parts[0]}_s$seed" "${GPUS[$((i % 4))]}" "$seed" "${parts[@]:1}" &
      i=$((i + 1))
    done
  done
  wait
  for variant in "${VARIANTS[@]}"; do
    read -r -a parts <<<"$variant"
    for seed in 1 2 3; do args+=(--lstm "${parts[0]}_s$seed=$M/lstm_tune_${parts[0]}_s$seed/best.pt"); done
  done
  bench "$M/lstm_tuning_benchmark" 3 "${args[@]}" &
  local best
  if [ -n "${DRY_RUN:-}" ]; then
    best=base
  else
    best=$("$PY" scripts/ungauged_analysis.py best-lstm) || { say "LSTM selection failed"; return 1; }
  fi
  say "Selected LSTM variant (2019 validation): $best"
  echo "$best" >"$M/lstm_selected_variant.txt"
  local extra=()
  for variant in "${VARIANTS[@]}"; do
    read -r -a parts <<<"$variant"
    [ "${parts[0]}" = "$best" ] && extra=("${parts[@]:1}")
  done
  say "2a. LSTM for each held-out fold (variant $best)"
  local fold
  for fold in $(seq 1 "$FOLDS"); do
    for seed in 1 2 3; do
      throttle "$LSTM_PARALLEL"
      train_lstm "lstm_heldout_f${fold}_s$seed" "${GPUS[$((i % 4))]}" "$seed" \
        --holdout-fold "$fold" --folds "$FOLDS" "${extra[@]}" &
      i=$((i + 1))
    done
  done
  wait
}

fold_bench() {  # fold_bench FOLD GPU
  local fold=$1 gpu=$2 lstms=() seed
  for seed in 1 2 3; do lstms+=("$M/lstm_heldout_f${fold}_s$seed"); done
  wait_for "$M/scratch_graph_heldout_f$fold" "${lstms[@]}" || return 1
  local args=(--checkpoint "graph=$M/scratch_graph_heldout_f$fold/best.pt")
  for seed in 1 2 3; do args+=(--lstm "lstm_s$seed=$M/lstm_heldout_f${fold}_s$seed/best.pt"); done
  bench "$M/heldout_benchmark_f$fold" "$gpu" --holdout-fold "$fold" --folds "$FOLDS" "${args[@]}"
}

# ---- the graph-model queue: one job per GPU at a time ------------------------------
QUEUE=$LOGS/experiments_queue
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
  while job=$(next_job); do
    eval "$job" || say "job failed (GPU $GPU): $job"
  done
}

rm -rf "$QUEUE"
mkdir -p "$QUEUE"
echo 0 >"$QUEUE/counter"
{
  for fold in $(seq 1 "$FOLDS"); do
    echo "train_graph scratch_graph_heldout_f$fold \$GPU 1 --holdout-fold $fold --folds $FOLDS"
  done
  for fold in $(seq 1 $((FOLDS - 1))); do echo "fold_bench $fold \$GPU"; done
  for job in "${ABLATIONS[@]}"; do
    read -r -a parts <<<"$job"
    [ "${parts[0]}" = scratch_graph_s2 ] && continue
    echo "train_graph ${parts[*]:0:1} \$GPU ${parts[*]:1}"
  done
  echo "fold_bench $FOLDS \$GPU"
  echo "train_graph scratch_graph_s2 \$GPU 2"
} >"$QUEUE/jobs"
say "Graph-model queue ($(wc -l <"$QUEUE/jobs") jobs on GPUs ${GPUS[*]}):"
sed 's/^/    /' "$QUEUE/jobs"

lstm_stage &
lstm=$!
workers=()
for gpu in "${GPUS[@]}"; do
  worker "$gpu" &
  workers+=($!)
done
for pid in "${workers[@]}"; do wait "$pid"; done
wait "$lstm" || say "LSTM stage failed"
wait  # the LSTM tuning benchmark

say "3. routing ablation: knockout, benchmark (4 GPUs) and figures"
bash scripts/routing_ablation.sh || say "routing ablation failed"
say "Figures"
run "$LOGS/experiments_figures.log" "$PY" scripts/ungauged_analysis.py figures --folds "$FOLDS"
say "All done: $ROOT/figures/lstm_tuning, $ROOT/figures/ungauged, $ROOT/figures/routing_ablation"
