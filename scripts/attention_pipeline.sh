#!/usr/bin/env bash
# AttentionGNN: the graph model with causal self-attention over time in place of the per-reach LSTM
# (--trunk attention), trained exactly as scripts/gnn_pipeline.sh trains the LSTM model, plus its own
# stage A (the trunk differs, so gefs_hydro_A cannot be reused): stage A, then stage B for all gauges and
# with each fold's gauges withheld (joint LSTM-style gauge head, NSE* + pinball loss), then the three
# test protocols. Forecast rule: observed data before the 00Z issue, GEFSv12 rain after it.
#
#   setsid -f nohup bash scripts/attention_pipeline.sh </dev/null >>/data/tva_data/logs/attention_pipeline.log 2>&1
#
# Outputs (in /data/tva_data/models): attn_A, attn_B, attn_B_f1..5, attn_evaluation (report.md).
# A step whose output holds DONE is skipped, so a rerun resumes. DRY_RUN=1 prints the commands.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH=$REPO/src OMP_NUM_THREADS=4
M=/data/tva_data/models
LOGS=/data/tva_data/logs
PY=/data/tva_data/conda_envs/tva_gnn/bin/python
GPUS=0,1,2,3
N=4  # processes, one per GPU; 2 issues accumulated each: 8 issues per update

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
train() {  # train STAGE OUT [extra arguments]
  local stage=$1 out=$2; shift 2
  env CUDA_VISIBLE_DEVICES=$GPUS "$PY" -W ignore::UserWarning -m torch.distributed.run --standalone --nproc_per_node $N \
    -m tva_flood.hydro.train --stage "$stage" --out "$out" --trunk attention --accumulate 2 "$@"
}

step "$M/attn_A" "$LOGS/attn_A.log" train A "$M/attn_A" --updates 5000 --lr 5e-4 --warmup 500 --validate-every 1000 || exit 1

for fold in 0 1 2 3 4 5; do
  name=attn_B; [ "$fold" -gt 0 ] && name=attn_B_f$fold
  step "$M/$name" "$LOGS/$name.log" train B "$M/$name" --init "$M/attn_A/best.pt" --fold "$fold" \
    --loss nse-q --head lstm --delta-weight 0 --updates 3000 --lr 3e-4 --warmup 100 --validate-every 500 || exit 1
done

step "$M/attn_evaluation" "$LOGS/attn_evaluation.log" env CUDA_VISIBLE_DEVICES=$GPUS \
  "$PY" -W ignore::UserWarning -m torch.distributed.run --standalone --nproc_per_node $N \
  -m tva_flood.hydro.evaluate --out "$M/attn_evaluation" --model "$M/attn_B/best.pt" --fold-models "$M/attn_B_f{k}/best.pt" --label AttentionGNN \
  || exit 1
say "pipeline finished: $M/attn_evaluation/report.md"
