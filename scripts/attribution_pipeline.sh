#!/usr/bin/env bash
# Attribution experiments for the manuscript (docs/ATTRIBUTION.md): what carries the GNN's held-out skill?
# Every model follows the GEFS rule (observed data before the 00Z issue, GEFSv12 control rain to 240 h after it)
# and the current GNN's recipe (stage B, --loss nse-q --head lstm, NSE* + pinball, 3000 updates of 8 samples),
# with checkpoints selected on 2019 gauge scores only (--select gauges; for the existing GNN this picks the same
# checkpoints as before). Each phase is started by hand; nothing runs on into the next phase.
#
#   bash scripts/attribution_pipeline.sh PHASE        (DRY_RUN=1 prints the jobs only)
#   setsid -f nohup bash scripts/attribution_pipeline.sh PHASE </dev/null >>/data/tva_data/logs/attr/PHASE.log 2>&1
#
# Phases:
#   phase0  P1: the LSTM retrained with the GNN's history masking (57.5% hidden) and selection (shown + hidden),
#               3 seeds x (all gauges + 5 folds), scored on CPU as the original LSTM;
#           P2: NWM-retrospective skill at every gauge (leakage covariate); Exp.3: conservation audit of the GNN.
#   phase1  Exp.1, seed 1: A1 (no exchange), A3 (graph, no NWM), A5 (lumped + NWM; its stage A first),
#           A6 (lumped, no NWM), each for all gauges + 5 folds; full evaluation of each. A0 seed 1 is the
#           existing GNN (models/gnn_B*, gnn_evaluation), linked, not retrained.
#   seeds   Exp.1, more seeds: SEEDS (default "2 3") of SEED_VARIANTS (default "A0 A5"); gauges-only evaluation.
#   phase2  S1: A0 and A5 on the 3 blocked tributary folds; S2: A0 and A5 with 25% and 50% of the training
#           gauges (5 folds each) and the stage-A models as 0%; S3: transfer covariates.
#   extra   The deferred variants: A2 (instantaneous routing), A4 (no stage-B prior), seed 1.
#   report  Every report: figures/attribution/{phase1,blocked,coverage,leakage,transfer,conservation}.
#
# One job per GPU (GPUS, default "0 1 2 3"): a run on one GPU accumulates 8 samples per update, the same
# sample list as a 4-GPU run. A step whose folder holds DONE is skipped, so a rerun resumes (a run that was
# cut off restarts). Approximate GPU-hours (one L40S): GNN stage B 7.7, lumped stage B 4, lumped stage A 7,
# from-scratch runs (A3, A6) x 8000/3000 (SCRATCH_UPDATES).
set -uo pipefail
PHASE=${1:?phase: phase0, phase1, seeds, phase2, extra or report}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH=$REPO/src OMP_NUM_THREADS=4
ROOT=/data/tva_data
M=$ROOT/models
A=$M/attr
LOGS=$ROOT/logs/attr
PY=$ROOT/conda_envs/tva_gnn/bin/python
GAUGES=$ROOT/processed/gauges_public/gauges.nc
TEST=$ROOT/processed/test_preparation_2020_2025
REFORECAST=$ROOT/processed/rain_forecasts/gefs_reforecast_240
OPERATIONAL=$ROOT/processed/rain_forecasts/gefs_operational_240
GPUS=(${GPUS:-0 1 2 3})
SCRATCH_UPDATES=${SCRATCH_UPDATES:-8000}  # A3, A6: the A0 budget of stage A (5000) + stage B (3000) updates
SEEDS=${SEEDS:-2 3}
SEED_VARIANTS=${SEED_VARIANTS:-A0 A5}
mkdir -p "$A" "$LOGS"

say() { echo "[$(date '+%F %T')] $*"; }
step() {  # step OUT LOG command...: run unless OUT/DONE; mark DONE or FAILED
  local out=$1 log=$2; shift 2
  [ -f "$out/DONE" ] && return 0
  if [ -n "${DRY_RUN:-}" ]; then echo "  $*  >> $log"; return 0; fi
  rm -f "$out/FAILED"
  say "start $(basename "$out")"
  if "$@" >>"$log" 2>&1; then
    mkdir -p "$out"; touch "$out/DONE"; say "done $(basename "$out")"
  else
    mkdir -p "$out"; touch "$out/FAILED"; say "FAILED $(basename "$out") (see $log)"; return 1
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

# A0 seed 1 is the existing GNN: link it (the gauges-only selection picks the same checkpoints).
link_a0() {
  [ -e "$A/A0_s1" ] || ln -s "$M/gnn_B" "$A/A0_s1"
  for k in 1 2 3 4 5; do [ -e "$A/A0_s1_f$k" ] || ln -s "$M/gnn_B_f$k" "$A/A0_s1_f$k"; done
  [ -e "$A/eval_A0_s1" ] || ln -s "$M/gnn_evaluation" "$A/eval_A0_s1"
}

# ---- training ------------------------------------------------------------------------------------------------
B_COMMON=(--stage B --loss nse-q --head lstm --delta-weight 0 --accumulate 8 --select gauges)
B_UPDATES=(--updates 3000 --lr 3e-4 --warmup 100 --validate-every 500)
SCRATCH=(--updates "$SCRATCH_UPDATES" --lr 5e-4 --warmup 500 --validate-every 1000)

variant_args() {  # the stage-B arguments of a variant
  case $1 in
    A0) echo "--init $M/gefs_hydro_A/best.pt ${B_UPDATES[*]}" ;;
    A1) echo "--init $M/gefs_hydro_A/best.pt --off mixing broadcast ${B_UPDATES[*]}" ;;
    A2) echo "--init $M/gefs_hydro_A/best.pt --off routing ${B_UPDATES[*]}" ;;
    A3) echo "--no-nwm ${SCRATCH[*]}" ;;
    A4) echo "--init $M/gefs_hydro_A/best.pt --nwm-weight 0 ${B_UPDATES[*]}" ;;
    A5) echo "--lumped --init $A/lumped_A/best.pt ${B_UPDATES[*]}" ;;
    A6) echo "--lumped --no-nwm ${SCRATCH[*]}" ;;
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}

train() {  # train NAME GPU SEED [train arguments]
  local name=$1 gpu=$2 seed=$3; shift 3
  step "$A/$name" "$LOGS/$name.log" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" -W ignore::UserWarning -m tva_flood.hydro.train --out "$A/$name" --seed "$seed" "$@"
}

train_as() {  # train_as NAME VARIANT SEED FOLD GPU [extra arguments, e.g. --scheme blocked, --keep 0.25]
  local name=$1 v=$2 seed=$3 fold=$4 gpu=$5; shift 5
  [ "$v" = A5 ] && { wait_for "$A/lumped_A" || return 1; }
  # shellcheck disable=SC2046
  train "$name" "$gpu" "$seed" "${B_COMMON[@]}" $(variant_args "$v") --fold "$fold" "$@"
}

train_variant() {  # train_variant VARIANT SEED FOLD GPU: the Exp.1 model of that fold (random folds)
  train_as "${1}_s${2}$(suffix "$3")" "$@"
}

lumped_stage_a() {  # the lumped model's NWM pre-training, as models/gefs_hydro_A for the GNN
  train lumped_A "$1" 1 --stage A --lumped --accumulate 8 --updates 5000 --lr 5e-4 --warmup 500 --validate-every 1000
}

# ---- evaluation ----------------------------------------------------------------------------------------------
evaluate() {  # evaluate NAME GPU MODEL FOLD_TEMPLATE LABEL [evaluate arguments]
  local name=$1 gpu=$2 model=$3 folds=$4 label=$5; shift 5
  step "$A/$name" "$LOGS/$name.log" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" -W ignore::UserWarning -m tva_flood.hydro.evaluate --out "$A/$name" --model "$model" --fold-models "$folds" --label "$label" "$@"
}

evaluate_variant() {  # evaluate_variant VARIANT SEED GPU [evaluate arguments]: all gauges + 5 folds
  local v=$1 seed=$2 gpu=$3 need=() k; shift 3
  for k in 0 1 2 3 4 5; do need+=("$A/${v}_s${seed}$(suffix "$k")"); done
  wait_for "${need[@]}" || return 1
  evaluate "eval_${v}_s${seed}" "$gpu" "$A/${v}_s${seed}/best.pt" "$A/${v}_s${seed}_f{k}/best.pt" "$v" "$@"
}

FAST=(--rains gefs --no-reaches)  # seeds and phase 2: the gauge protocols with GEFS rain only

# ---- the LSTM with the GNN's masking (P1) ---------------------------------------------------------------------
lstm_matched() {  # lstm_matched SEED FOLD GPU
  local seed=$1 fold=$2 gpu=$3 name extra=()
  name=lstm_matched_s${seed}$(suffix "$fold")
  [ "$fold" -gt 0 ] && extra=(--holdout-fold "$fold" --folds 5)
  [ -f "$A/$name/DONE" ] || rm -rf "$A/$name"  # train-lstm refuses a folder holding a model
  step "$A/$name" "$LOGS/$name.log" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" -u -m tva_flood train-lstm --out "$A/$name" --seed "$seed" --gauges "$GAUGES" \
    --forecast-rain "$REFORECAST" --forecast-fraction 1 --unknown-rain-prob 0 --rain-cut-prob 0 \
    --rain-hours 240 --validation-rain "$REFORECAST" --validation-gauges camelsh --validation-split \
    --sample-budget 3300807 --evaluations 10 --min-count 100 \
    --batch-size 48 --lr 5e-4 --dropout 0 --balanced \
    --history-hide 0.575 --validation-hidden "${extra[@]}"
}

lstm_benchmark() {  # lstm_benchmark SEED FOLD (CPU, 12 processes; as models/gefs_benchmark*)
  local seed=$1 fold=$2 s name extra=()
  s=$(suffix "$fold")
  name=lstm_matched_s${seed}_bench$s
  [ "$fold" -gt 0 ] && extra=(--holdout-fold "$fold" --folds 5)
  wait_for "$A/lstm_matched_s${seed}$s" || return 1
  [ -f "$A/$name/DONE" ] || rm -rf "$A/$name"
  step "$A/$name" "$LOGS/$name.log" env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node 12 -m tva_flood evaluate-benchmark \
    --device cpu --preparation "$TEST" --array-cache "${TEST}_arrays" --gauges "$GAUGES" \
    --gefs "$OPERATIONAL" --out "$A/$name" --lstm "lstm=$A/lstm_matched_s${seed}$s/best.pt" "${extra[@]}"
}

# ---- the GPU queue -------------------------------------------------------------------------------------------
QUEUE=$LOGS/queue_$PHASE
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
run_queue() {  # the jobs on stdin, one GPU each, in order
  rm -rf "$QUEUE"
  mkdir -p "$QUEUE"
  echo 0 >"$QUEUE/counter"
  cat >"$QUEUE/jobs"
  say "$PHASE: $(wc -l <"$QUEUE/jobs") jobs on GPUs ${GPUS[*]}"
  sed 's/^/    /' "$QUEUE/jobs"
  local pids=() gpu
  for gpu in "${GPUS[@]}"; do
    worker "$gpu" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

report() {
  local section
  for section in "$@"; do
    case $section in
      phase1 | blocked | coverage)
        [ -n "${DRY_RUN:-}" ] && { echo "  attribution_report.py $section"; continue; }
        "$PY" -W ignore scripts/attribution_report.py "$section" >>"$LOGS/report.log" 2>&1 || say "report $section failed (see $LOGS/report.log)" ;;
      leakage | transfer)
        [ -n "${DRY_RUN:-}" ] && { echo "  attribution_analysis.py $section"; continue; }
        "$PY" -W ignore scripts/attribution_analysis.py "$section" >>"$LOGS/report.log" 2>&1 || say "analysis $section failed (see $LOGS/report.log)" ;;
    esac
  done
}

link_a0
case $PHASE in
  phase0)
    {
      for seed in 1 2 3; do for fold in 0 1 2 3 4 5; do echo "lstm_matched $seed $fold \$GPU"; done; done
      echo "step $A/conservation_A0 $LOGS/conservation_A0.log env CUDA_VISIBLE_DEVICES=\$GPU $PY -W ignore scripts/attribution_analysis.py conservation --model $M/gnn_B/best.pt"
    } | run_queue &
    gpu_jobs=$!
    (for seed in 1 2 3; do for fold in 0 1 2 3 4 5; do lstm_benchmark "$seed" "$fold"; done; done) &
    cpu_jobs=$!
    report leakage
    wait "$gpu_jobs" "$cpu_jobs"
    report phase1
    say "phase0 finished: $ROOT/figures/attribution/{phase1,leakage,conservation}/report.md" ;;
  phase1)
    {
      echo "lumped_stage_a \$GPU"
      for fold in 0 1 2 3 4 5; do echo "train_variant A3 1 $fold \$GPU"; done
      for fold in 0 1 2 3 4 5; do echo "train_variant A6 1 $fold \$GPU"; done
      for fold in 0 1 2 3 4 5; do echo "train_variant A1 1 $fold \$GPU"; done
      for fold in 0 1 2 3 4 5; do echo "train_variant A5 1 $fold \$GPU"; done
      for v in A1 A5 A6 A3; do echo "evaluate_variant $v 1 \$GPU"; done
    } | run_queue
    report phase1 leakage
    say "phase1 finished: $ROOT/figures/attribution/phase1/report.md" ;;
  seeds)
    {
      for seed in $SEEDS; do for v in $SEED_VARIANTS; do for fold in 0 1 2 3 4 5; do echo "train_variant $v $seed $fold \$GPU"; done; done; done
      for seed in $SEEDS; do for v in $SEED_VARIANTS; do echo "evaluate_variant $v $seed \$GPU ${FAST[*]}"; done; done
    } | run_queue
    report phase1
    say "seeds finished: $ROOT/figures/attribution/phase1/report.md" ;;
  phase2)
    for v in A0 A5; do  # S2 0%: the stage-A model as every fold model, seeing no gauge
      stage_a=$M/gefs_hydro_A/best.pt
      [ "$v" = A5 ] && stage_a=$A/lumped_A/best.pt
      if [ -z "${DRY_RUN:-}" ] && [ ! -f "$A/S2_${v}_k000_f5/DONE" ]; then
        "$PY" scripts/attribution_analysis.py zero-coverage --stage-a "$stage_a" --out-template "$A/S2_${v}_k000_f{k}" >>"$LOGS/phase2.log" 2>&1
      fi
    done
    {
      for v in A0 A5; do for b in 1 2 3; do echo "train_as S1_${v}_s1_b$b $v 1 $b \$GPU --scheme blocked"; done; done
      for keep in 025 050; do for v in A0 A5; do for fold in 1 2 3 4 5; do
        echo "train_as S2_${v}_k${keep}_f$fold $v 1 $fold \$GPU --keep 0.${keep#0} --keep-seed 1"
      done; done; done
      for v in A0 A5; do
        echo "wait_for $A/S1_${v}_s1_b1 $A/S1_${v}_s1_b2 $A/S1_${v}_s1_b3 && evaluate eval_S1_${v}_s1 \$GPU $A/${v}_s1/best.pt $A/S1_${v}_s1_b{k}/best.pt S1_$v ${FAST[*]}"
        for keep in 000 025 050; do
          echo "wait_for $A/S2_${v}_k${keep}_f1 $A/S2_${v}_k${keep}_f2 $A/S2_${v}_k${keep}_f3 $A/S2_${v}_k${keep}_f4 $A/S2_${v}_k${keep}_f5 && evaluate eval_S2_${v}_k$keep \$GPU $A/${v}_s1/best.pt $A/S2_${v}_k${keep}_f{k}/best.pt S2_${v}_$keep ${FAST[*]}"
        done
      done
    } | run_queue
    report blocked coverage transfer
    say "phase2 finished: $ROOT/figures/attribution/{blocked,coverage,transfer}/report.md" ;;
  extra)
    {
      for v in A2 A4; do for fold in 0 1 2 3 4 5; do echo "train_variant $v 1 $fold \$GPU"; done; done
      for v in A2 A4; do echo "evaluate_variant $v 1 \$GPU"; done
    } | run_queue
    report phase1 ;;
  report)
    report phase1 leakage blocked coverage transfer ;;
  *)
    echo "unknown phase $PHASE" >&2; exit 1 ;;
esac
