#!/usr/bin/env bash
# The graph model, one training method for every model: stage B from the shared stage A (models/gefs_hydro_A),
# trained jointly with the LSTM-style gauge head (--head lstm) and the NSE* + pinball tail loss (--loss nse-q),
# for all gauges and with each fold's gauges withheld; then the three test protocols. Forecast rule: observed
# data before the 00Z issue, GEFSv12 rain after it (reforecast to train and select, operational to test).
#
#   setsid -f nohup bash scripts/gnn_pipeline.sh </dev/null >>/data/tva_data/logs/gnn_pipeline.log 2>&1
#
# Outputs (in /data/tva_data/models): gnn_B, gnn_B_f1..5, gnn_evaluation (report.md).
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

for fold in 0 1 2 3 4 5; do
  name=gnn_B; [ "$fold" -gt 0 ] && name=gnn_B_f$fold
  step "$M/$name" "$LOGS/$name.log" env CUDA_VISIBLE_DEVICES=$GPUS \
    "$PY" -W ignore::UserWarning -m torch.distributed.run --standalone --nproc_per_node $N \
    -m tva_flood.hydro.train --stage B --out "$M/$name" --init "$M/gefs_hydro_A/best.pt" --fold "$fold" \
    --loss nse-q --head lstm --delta-weight 0 --accumulate 2 --updates 3000 --lr 3e-4 --warmup 100 --validate-every 500 \
    || exit 1
done

step "$M/gnn_evaluation" "$LOGS/gnn_evaluation.log" env CUDA_VISIBLE_DEVICES=$GPUS \
  "$PY" -W ignore::UserWarning -m torch.distributed.run --standalone --nproc_per_node $N \
  -m tva_flood.hydro.evaluate --out "$M/gnn_evaluation" --model "$M/gnn_B/best.pt" --fold-models "$M/gnn_B_f{k}/best.pt" --label GNN \
  || exit 1
say "pipeline finished: $M/gnn_evaluation/report.md"
