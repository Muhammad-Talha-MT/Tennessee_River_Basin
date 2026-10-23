#!/usr/bin/env bash
# The regulated graph-routing model (V3: tva_flood.hydro, RegulatedModel) on all four GPUs:
# observed weather, flow and dam records before the 00Z issue, GEFSv12 rain to 240 h after it.
#
#   setsid -f nohup bash scripts/v3_pipeline.sh </dev/null >>/data/tva_data/logs/v3_pipeline.log 2>&1
#
# models/v3_A                 stage A: NWM prior off the dams' reach, dam releases and inflows
# models/v3_B, v3_B_f1..5     stage B: forecast on all gauges, and with each gauge fold withheld
# models/v3_evaluation        the four protocols (hydro.evaluate)
#
# No stage C: training the gauge head further on a frozen stage-B model (hydro.head) made it
# worse on unseen years in every setting tried (2017-2018 and 2019, its own loss and NSE, at
# learning rates 1e-3 and 1e-4): stage B already shows the head about a million (issue,
# gauge) examples, so more passes only overfit.
#
# Needs processed/hydro/dams.npz (python -m tva_flood hydro-dams). A step whose output
# holds DONE is skipped, so a rerun resumes. DRY_RUN=1 prints the commands.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH=$REPO/src OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3
M=/data/tva_data/models
LOGS=/data/tva_data/logs
PY=/data/tva_data/conda_envs/tva_gnn/bin/python
RUN="$PY -W ignore::UserWarning -m torch.distributed.run --standalone --nproc_per_node 4"
mkdir -p "$LOGS"

say() { echo "[$(date '+%F %T')] $*"; }
step() {  # step OUT LOG command...: run unless OUT/DONE
  local out=$1 log=$2; shift 2
  [ -f "$out/DONE" ] && return 0
  say "start $(basename "$out") ($(basename "$log"))"
  if [ -n "${DRY_RUN:-}" ]; then echo "  $*"; return 0; fi
  if "$@" >>"$log" 2>&1; then say "done $(basename "$log")"; else say "FAILED (see $log)"; return 1; fi
}
mark() { [ -n "${DRY_RUN:-}" ] || touch "$1/DONE"; }

stage_a() {
  [ -f "$M/v3_A/DONE" ] && return 0
  step "$M/v3_A" "$LOGS/v3_A.log" $RUN -m tva_flood.hydro.train --stage A --out "$M/v3_A" \
    --updates 5000 --accumulate 2 --lr 5e-4 --warmup 500 --validate-every 1000 && mark "$M/v3_A"
}

stage_b() {  # stage_b FOLD (0 = all gauges)
  local fold=$1 name=v3_B
  [ "$fold" -gt 0 ] && name=v3_B_f$fold
  [ -f "$M/$name/DONE" ] && return 0
  # A run of this output started by an earlier pipeline: let it finish and keep it.
  if [ -z "${DRY_RUN:-}" ] && pgrep -f -- "--out $M/$name --" >/dev/null; then
    say "waiting for the running $name"
    while pgrep -f -- "--out $M/$name --" >/dev/null; do sleep 60; done
    if grep -q "TRAINING COMPLETE" "$LOGS/$name.log"; then mark "$M/$name"; say "done $name"; return 0; fi
  fi
  step "$M/$name" "$LOGS/$name.log" $RUN -m tva_flood.hydro.train --stage B --init "$M/v3_A/best.pt" \
    --fold "$fold" --out "$M/$name" --updates 3000 --accumulate 2 --lr 3e-4 --warmup 100 --validate-every 500 || return 1
  mark "$M/$name"
}

evaluate() {
  [ -f "$M/v3_evaluation/DONE" ] && return 0
  step "$M/v3_evaluation" "$LOGS/v3_evaluation.log" $RUN -m tva_flood.hydro.evaluate --out "$M/v3_evaluation" \
    && mark "$M/v3_evaluation"
}

stage_a || exit 1
for k in 0 1 2 3 4 5; do stage_b "$k" || exit 1; done
evaluate
say "V3 pipeline finished: $M/v3_evaluation/report.md"
