#!/bin/bash
# One Frontier node = 8 GPUs (GCDs): runs the jobs of one attribution phase, one training run per GPU.
# Submit several copies to use several nodes; they share one job list through claim folders, a cut-off run resumes
# from its last checkpoint, and finished runs are skipped. See scripts/frontier/README.md.
#
#   sbatch --export=ALL,PHASE=phase1 scripts/frontier/run_phase.sbatch      # PHASE: phase1, seeds, phase2, extra
#
#SBATCH -A geo173
#SBATCH -J tva_attr
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -o /lustre/orion/geo173/proj-shared/tva_data/logs/slurm-%j.out
#SBATCH --signal=B:TERM@180

set -uo pipefail
PHASE=${PHASE:?submit with --export=ALL,PHASE=phase1}
export TVA_ROOT=${TVA_ROOT:-/lustre/orion/geo173/proj-shared/tva_data}
REPO=${REPO:-$TVA_ROOT/Tennessee_River_Basin}
module load PrgEnv-gnu cray-python rocm
# shellcheck disable=SC1091
source "$TVA_ROOT/venv/bin/activate"

export PYTHON=$TVA_ROOT/venv/bin/python
export GPUS="0 1 2 3 4 5 6 7"
export TRAIN_WORKERS=${TRAIN_WORKERS:-4}
export EVAL_EXTRA="--no-reaches --rains gefs"
export NO_REPORT=1
export SCRATCH_UPDATES=${SCRATCH_UPDATES:-8000}   # the full length of A0 (5000 stage A + 3000 stage B)
export QUEUE_DIR=/tmp/tva_queue_$SLURM_JOB_ID      # node-local: flock does not work reliably on Lustre
export OMP_NUM_THREADS=4
mkdir -p "$TVA_ROOT/logs/attr" "$QUEUE_DIR"
cd "$REPO"

release() {  # at the wall-time signal: kill the runs (they resume next time) and free this job's claims
  pkill -P $$ 2>/dev/null
  for c in "$TVA_ROOT"/models/attr/*.claim; do [ "$(cat "$c/owner" 2>/dev/null)" = "$SLURM_JOB_ID" ] && rm -rf "$c"; done
}
trap 'release; exit 143' TERM

srun -N1 -n1 -c 56 --gpus-per-node=8 --gpu-bind=none bash scripts/attribution_pipeline.sh "$PHASE" &
wait $!
release
