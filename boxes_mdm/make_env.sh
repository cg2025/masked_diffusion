#!/usr/bin/env bash
# Build the venv the sbatch expects. Run once on a login node (needs internet), NOT inside the job.
# usage: ROOT=/data/scratch/cgoyal/boxes-diffusion bash boxes_mdm/make_env.sh
set -euo pipefail
ROOT=${ROOT:-/data/scratch/cgoyal/boxes-diffusion}
export TMPDIR=${TMPDIR:-/data/scratch/cgoyal/tmp}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/data/scratch/cgoyal/uv-python}
mkdir -p "$TMPDIR"
command -v uv >/dev/null || { echo "uv not found: pip install --user uv  (or use conda create -n diffusion python=3.9)"; exit 1; }
uv venv --python 3.9 "$ROOT/.venv"
uv pip install -p "$ROOT/.venv/bin/python" -r "$ROOT/diffusion-vs-ar/requirements.txt" \
    -f https://download.pytorch.org/whl/torch_stable.html
"$ROOT/.venv/bin/python" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
