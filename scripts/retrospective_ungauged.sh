#!/usr/bin/env bash
# The held-out fold models (ungauged mode) and NWM medium-range forecasts scored against
# the NWM retrospective at every reach, 00Z issues 2020-2022 (every second day), as
# retrospective_comparison.py does for the all-gauge models. Folds run on GPUs 0-3, then
# fold 5; the NWM forecasts (nwm_all_reaches.py, then --reference-only on CPU) alongside.
#
#   setsid -f nohup bash scripts/retrospective_ungauged.sh </dev/null \
#     >>/data/tva_data/logs/retrospective_ungauged.log 2>&1
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH=$REPO/src
source scripts/train_lib.sh
PYDL=$ROOT/conda_envs/tva_download/bin/python  # h5py, requests
SCRIPT=scripts/retrospective_comparison.py

fold() {  # fold K GPU
  local k=$1 gpu=$2
  [ -f "$M/retrospective_comparison_heldout_f$k/metrics.npz" ] && return 0
  say "fold $k (GPU $gpu)"
  run "$LOGS/retrospective_heldout_f$k.log" env CUDA_VISIBLE_DEVICES="$gpu" "$PY" $SCRIPT compute \
    --every 2 --checkpoint "$M/scratch_graph_heldout_f$k/best.pt" --holdout-fold "$k" --tag "heldout_f$k"
}
nwm() {
  say "NWM medium-range streamflow at every reach"
  PYTHONPATH=$REPO/src run "$LOGS/nwm_all_reaches.log" "$PYDL" scripts/nwm_all_reaches.py || return 1
  [ -f "$M/retrospective_comparison_nwm/metrics.npz" ] && return 0
  say "NWM forecasts against the retrospective (CPU, 12 processes)"
  local k
  for k in $(seq 0 11); do
    run "$LOGS/retrospective_nwm_part$k.log" env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 "$PY" $SCRIPT \
      compute --device cpu --every 2 --reference-only --tag nwm --shard "$k" --shards 12 &
  done
  wait
  run "$LOGS/retrospective_nwm.log" "$PY" $SCRIPT merge --tag nwm
}

if [ "${1:-}" = nwm ]; then nwm; exit; fi  # the NWM part alone
nwm &
fold 1 0 & fold 2 1 & fold 3 2 & fold 4 3 &
wait
fold 5 0
say "All parts done"
