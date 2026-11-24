#!/usr/bin/env bash
# One M4 blocked-basin stage-B run on the GPUs in $GPUS, with the pipeline's claim/DONE markers (scripts/attribution_pipeline.sh: claim, step), so the
# running `m4` queue skips this fold. Same arguments as `train_as_all S1_M4_s1_b$FOLD M4 1 $FOLD --scheme blocked` (variant_args M4, B_COMMON, B_UPDATES).
#   GPUS="0 1" bash scripts/m4_blocked_fold.sh FOLD
set -uo pipefail
FOLD=${1:?blocked fold 1-3}
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
export PYTHONPATH=$REPO/src OMP_NUM_THREADS=4
ROOT=${DATA_ROOT:-/data/tva_data}; export TVA_ROOT=$ROOT   # the data root (same default as the pipeline)
A=$ROOT/models/attr; LOGS=$ROOT/logs/attr; PY=$ROOT/conda_envs/tva_gnn/bin/python
OUT=$A/S1_M4_s1_b$FOLD; LOG=$LOGS/S1_M4_s1_b$FOLD.log
GPUS=(${GPUS:-0 1}); LIST=$(IFS=,; echo "${GPUS[*]}")
say() { echo "[$(date '+%F %T')] $*"; }
[ -f "$OUT/DONE" ] && { say "$(basename "$OUT") already DONE"; exit 0; }
[ -f "$A/instant_A/DONE" ] || { say "instant_A is not DONE"; exit 1; }
if ! mkdir "$OUT.claim" 2>/dev/null; then say "$(basename "$OUT") is claimed by another job: $(cat "$OUT.claim/owner" 2>/dev/null)"; exit 1; fi
echo "local:$$" >"$OUT.claim/owner"
rm -f "$OUT/FAILED"
say "start $(basename "$OUT") on GPUs $LIST"
if env CUDA_VISIBLE_DEVICES="$LIST" HIP_VISIBLE_DEVICES="$LIST" "$PY" -W ignore::UserWarning -m torch.distributed.run --rdzv-backend c10d \
    --rdzv-endpoint 127.0.0.1:0 --local-addr 127.0.0.1 --nproc_per_node "${#GPUS[@]}" -m tva_flood.hydro.train --out "$OUT" --seed 1 --accumulate $((8 / ${#GPUS[@]})) \
    --stage B --loss nse-q --head lstm --delta-weight 0 --select gauges --resume --workers "${TRAIN_WORKERS:-5}" \
    --router instant --init "$A/instant_A/best.pt" --updates 3000 --lr 3e-4 --warmup 100 --validate-every 500 --fold "$FOLD" --scheme blocked >>"$LOG" 2>&1; then
  mkdir -p "$OUT"; touch "$OUT/DONE"; rm -rf "$OUT.claim"; say "done $(basename "$OUT")"
else
  mkdir -p "$OUT"; touch "$OUT/FAILED"; rm -rf "$OUT.claim"; say "FAILED $(basename "$OUT") (see $LOG)"; exit 1
fi
