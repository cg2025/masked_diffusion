#!/bin/bash
#SBATCH --job-name=boxes-smoke
#SBATCH --partition=drl-titanrtx
#SBATCH --qos=drl-main
#SBATCH --account=drl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/scratch/cgoyal/boxes-diffusion/logs/%x-%j.out
#SBATCH --error=/data/scratch/cgoyal/boxes-diffusion/logs/%x-%j.err

set -uo pipefail

echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=[${CUDA_VISIBLE_DEVICES-unset}]  SLURM_GPUS_PER_TASK=[${SLURM_GPUS_PER_TASK-unset}]"
echo "--- before env.sh ---"; nvidia-smi -L || echo "nvidia-smi failed"

source /data/scratch/cgoyal/env.sh
source /data/scratch/cgoyal/boxes-diffusion/.venv/bin/activate
export TMPDIR=/data/scratch/cgoyal/tmp
export UV_PYTHON_INSTALL_DIR=/data/scratch/cgoyal/uv-python
mkdir -p $TMPDIR

echo "--- after env.sh ---  CUDA_VISIBLE_DEVICES=[${CUDA_VISIBLE_DEVICES-unset}]"
nvidia-smi -L || echo "nvidia-smi failed"
python - <<'PY'
import torch
print(torch.__version__, "cuda available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
PY
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "NO GPU VISIBLE - stopping"; exit 1; }

cd /data/scratch/cgoyal/boxes-diffusion/diffusion-vs-ar
NGPU=1 EPOCHS=2 bash scripts/boxes/train-mdm.sh
