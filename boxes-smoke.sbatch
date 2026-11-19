#!/bin/bash
# Cluster wrapper around the portable runner. Everything real lives in boxes_mdm/run_boxes.sh.
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
# The logs/ directory must already exist (mkdir -p logs) or the job dies silently.
#SBATCH --output=/data/scratch/cgoyal/boxes-diffusion/logs/%x-%j.out
#SBATCH --error=/data/scratch/cgoyal/boxes-diffusion/logs/%x-%j.err
set -uo pipefail
ROOT=${ROOT:-/data/scratch/cgoyal/boxes-diffusion}
nvidia-smi -L || { echo "NO GPU on this node - stopping"; exit 1; }
[ -f /data/scratch/cgoyal/env.sh ] && source /data/scratch/cgoyal/env.sh
export TMPDIR=/data/scratch/cgoyal/tmp; mkdir -p "$TMPDIR"
[ -f "$ROOT/.venv/bin/activate" ] || { echo "venv missing: run boxes_mdm/make_env.sh (or pip install -r requirements.txt in your own env)"; exit 1; }
source "$ROOT/.venv/bin/activate"
cd "$ROOT" && EPOCHS=2 bash boxes_mdm/run_boxes.sh
