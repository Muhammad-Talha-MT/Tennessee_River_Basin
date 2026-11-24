#!/usr/bin/env bash
# GNN (M0) forecasts with the realized (AORC) rain in place of the GEFS forecast rain after the issue: a labelled "+aorc" diagnostic (same weights as the GEFS evaluation,
# no training). Compares with the LSTM "lstm+aorc" that is already saved in the ladder_L4_s1_bench* files.
#   GPUS="1 2" bash scripts/m0_aorc_evaluate.sh gauged_heldout  ->  models/attr/eval_M0_s1_aorc      (all-gauge model + random folds M0_s1_f{k})
#   GPUS="3"   bash scripts/m0_aorc_evaluate.sh blocked         ->  models/attr/eval_S1_M0_s1_aorc   (blocked folds S1_M0_s1_b{k})
set -uo pipefail
KIND=${1:?gauged_heldout or blocked}
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
export PYTHONPATH=$REPO/src OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=${DATA_ROOT:-/data/tva_data}; export TVA_ROOT=$ROOT
A=$ROOT/models/attr; LOGS=$ROOT/logs/attr; PY=$ROOT/conda_envs/tva_gnn/bin/python
case $KIND in
  gauged_heldout) NAME=eval_M0_s1_aorc; TEMPLATE=$A/M0_s1_f{k}/best.pt; LABEL=M0_aorc; EXTRA=(--model "$A/M0_s1/best.pt") ;;
  blocked) NAME=eval_S1_M0_s1_aorc; TEMPLATE=$A/S1_M0_s1_b{k}/best.pt; LABEL=S1_M0_aorc; EXTRA=(--heldout-only) ;;
  *) echo "gauged_heldout or blocked"; exit 1 ;;
esac
OUT=$A/$NAME; LOG=$LOGS/$NAME.log
GPUS=(${GPUS:-1 2}); LIST=$(IFS=,; echo "${GPUS[*]}")
say() { echo "[$(date '+%F %T')] $*"; }
[ -f "$OUT/DONE" ] && { say "$NAME already DONE"; exit 0; }
if ! mkdir "$OUT.claim" 2>/dev/null; then say "$NAME is claimed: $(cat "$OUT.claim/owner" 2>/dev/null)"; exit 1; fi
echo "local:$$" >"$OUT.claim/owner"; rm -f "$OUT/FAILED"
say "start $NAME on GPUs $LIST"
if env CUDA_VISIBLE_DEVICES="$LIST" HIP_VISIBLE_DEVICES="$LIST" "$PY" -W ignore::UserWarning -m torch.distributed.run --rdzv-backend c10d \
    --rdzv-endpoint 127.0.0.1:0 --local-addr 127.0.0.1 --nproc_per_node "${#GPUS[@]}" -m tva_flood.hydro.evaluate --out "$OUT" \
    --fold-models "$TEMPLATE" --label "$LABEL" --rains aorc --no-reaches "${EXTRA[@]}" >>"$LOG" 2>&1; then
  mkdir -p "$OUT"; touch "$OUT/DONE"; rm -rf "$OUT.claim"; say "done $NAME"
else
  mkdir -p "$OUT"; touch "$OUT/FAILED"; rm -rf "$OUT.claim"; say "FAILED $NAME (see $LOG)"; exit 1
fi
