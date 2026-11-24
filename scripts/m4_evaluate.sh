#!/usr/bin/env bash
# M4 held-out evaluation (no all-gauge model, gauge protocols with GEFS rain only), as the M1 blocked evaluation:
#   torch.distributed.run -m tva_flood.hydro.evaluate --out OUT --fold-models TEMPLATE --label LABEL --rains gefs --no-reaches --heldout-only
# Same claim/DONE markers as scripts/attribution_pipeline.sh (step).
#   GPUS="0 1" bash scripts/m4_evaluate.sh heldout   ->  eval_M4_s1     (random folds, M4_s1_f{k})
#   GPUS="2 3" bash scripts/m4_evaluate.sh blocked   ->  eval_S1_M4_s1  (blocked folds, S1_M4_s1_b{k})
set -uo pipefail
KIND=${1:?heldout or blocked}
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
export PYTHONPATH=$REPO/src OMP_NUM_THREADS=4
ROOT=${DATA_ROOT:-/data/tva_data}; export TVA_ROOT=$ROOT
A=$ROOT/models/attr; LOGS=$ROOT/logs/attr; PY=$ROOT/conda_envs/tva_gnn/bin/python
case $KIND in
  heldout) NAME=eval_M4_s1; TEMPLATE=$A/M4_s1_f{k}/best.pt; LABEL=M4; NEED=(M4_s1_f1 M4_s1_f2 M4_s1_f3 M4_s1_f4 M4_s1_f5) ;;
  blocked) NAME=eval_S1_M4_s1; TEMPLATE=$A/S1_M4_s1_b{k}/best.pt; LABEL=S1_M4; NEED=(S1_M4_s1_b1 S1_M4_s1_b2 S1_M4_s1_b3) ;;
  *) echo "heldout or blocked"; exit 1 ;;
esac
OUT=$A/$NAME; LOG=$LOGS/$NAME.log
GPUS=(${GPUS:-0 1}); LIST=$(IFS=,; echo "${GPUS[*]}")
say() { echo "[$(date '+%F %T')] $*"; }
[ -f "$OUT/DONE" ] && { say "$NAME already DONE"; exit 0; }
for d in "${NEED[@]}"; do [ -f "$A/$d/DONE" ] || { say "$d is not DONE"; exit 1; }; done
if ! mkdir "$OUT.claim" 2>/dev/null; then say "$NAME is claimed: $(cat "$OUT.claim/owner" 2>/dev/null)"; exit 1; fi
echo "local:$$" >"$OUT.claim/owner"; rm -f "$OUT/FAILED"
say "start $NAME on GPUs $LIST"
if env CUDA_VISIBLE_DEVICES="$LIST" HIP_VISIBLE_DEVICES="$LIST" "$PY" -W ignore::UserWarning -m torch.distributed.run --rdzv-backend c10d \
    --rdzv-endpoint 127.0.0.1:0 --local-addr 127.0.0.1 --nproc_per_node "${#GPUS[@]}" -m tva_flood.hydro.evaluate --out "$OUT" \
    --fold-models "$TEMPLATE" --label "$LABEL" --rains gefs --no-reaches --heldout-only >>"$LOG" 2>&1; then
  mkdir -p "$OUT"; touch "$OUT/DONE"; rm -rf "$OUT.claim"; say "done $NAME"
else
  mkdir -p "$OUT"; touch "$OUT/FAILED"; rm -rf "$OUT.claim"; say "FAILED $NAME (see $LOG)"; exit 1
fi
