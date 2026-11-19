#!/bin/bash
#SBATCH --job-name=boxes-smoke
#SBATCH --partition=drl-titanrtx
#SBATCH --qos=drl-main
#SBATCH --account=drl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# All 3 earlier jobs landed on gpu20-2 and saw no GPU there; remove this line once that node is fixed.
#SBATCH --exclude=gpu20-2
# NOTE: the logs/ directory must already exist or the job dies silently.
#SBATCH --output=/data/scratch/cgoyal/boxes-diffusion/logs/%x-%j.out
#SBATCH --error=/data/scratch/cgoyal/boxes-diffusion/logs/%x-%j.err

set -uo pipefail
ROOT=${ROOT:-/data/scratch/cgoyal/boxes-diffusion}

echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=[${CUDA_VISIBLE_DEVICES-unset}]  SLURM_JOB_GPUS=[${SLURM_JOB_GPUS-unset}]"
nvidia-smi -L || { echo "NO GPU on this node (nvidia-smi failed) - stopping"; exit 1; }

[ -f /data/scratch/cgoyal/env.sh ] && source /data/scratch/cgoyal/env.sh
export TMPDIR=/data/scratch/cgoyal/tmp
export UV_PYTHON_INSTALL_DIR=/data/scratch/cgoyal/uv-python
mkdir -p "$TMPDIR"

[ -f "$ROOT/.venv/bin/activate" ] || { echo "venv missing at $ROOT/.venv - run boxes_mdm/make_env.sh on a login node first"; exit 1; }
source "$ROOT/.venv/bin/activate"
command -v python >/dev/null || { echo "python not found after activating venv"; exit 1; }

python -c "import torch,sys; print(torch.__version__, 'cuda available:', torch.cuda.is_available(), 'count:', torch.cuda.device_count()); sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "torch cannot see the GPU - stopping"; exit 1; }

cd "$ROOT" || exit 1
[ -f diffusion-vs-ar/src/train_bash.py ] || { echo "submodules missing: run 'git submodule update --init' in $ROOT"; exit 1; }
# one-time data prep (idempotent)
[ -f diffusion-vs-ar/data/boxes_lengths.json ] || bash boxes_mdm/setup_boxes.sh || exit 1

cd diffusion-vs-ar
NGPU=1 EPOCHS=2 bash scripts/boxes/train-mdm.sh
