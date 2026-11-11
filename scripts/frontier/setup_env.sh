#!/usr/bin/env bash
# One-time Python environment for Frontier (AMD MI250X, ROCm). Run on a LOGIN node (compute nodes have no internet):
#   bash scripts/frontier/setup_env.sh
# Module names and the PyTorch wheel index change between OLCF software releases: if a line fails, check
# `module avail rocm cray-python` and https://pytorch.org (ROCm wheels) and override MODULES / TORCH_INDEX.
set -euo pipefail
PROJ=${PROJ:-geo173}
export TVA_ROOT=${TVA_ROOT:-/lustre/orion/$PROJ/proj-shared/tva_data}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
MODULES=${MODULES:-"PrgEnv-gnu cray-python rocm"}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}

module load $MODULES
python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python {sys.version.split()[0]}: tva_flood needs 3.11+ (try `module load miniforge3` or another cray-python)")
PY
python3 -m venv "$TVA_ROOT/venv"
# shellcheck disable=SC1091
source "$TVA_ROOT/venv/bin/activate"
pip install --upgrade pip
pip install torch --index-url "$TORCH_INDEX"
pip install "numpy>=1.24,<3" "pandas>=2,<3" "xarray>=2024.1" "netCDF4>=1.6" "pyarrow>=14" "scipy>=1.10" "tqdm>=4.66"
pip install -e "$REPO" --no-deps
python -c "import tva_flood.hydro.train, tva_flood.hydro.evaluate; print('tva_flood imports ok')"
echo "environment ready: $TVA_ROOT/venv  (load with: module load $MODULES; source $TVA_ROOT/venv/bin/activate)"
