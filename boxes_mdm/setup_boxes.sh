#!/usr/bin/env bash
# usage: bash boxes_mdm/setup_boxes.sh [REPO] [KS_REPO] [SPLIT]
# Defaults: the diffusion-vs-ar / entity-tracking-lms submodules next to boxes_mdm/.
# Needs: submodules checked out (git submodule update --init), `unzip`, and the project venv
# active (prep_boxes.py imports transformers). The Google Drive data from the diffusion-vs-ar
# README is NOT needed for boxes: prep_boxes.py creates data/dataset_info.json itself.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
REPO=${1:-$ROOT/diffusion-vs-ar}; KS=${2:-$ROOT/entity-tracking-lms}; SPLIT=${3:-t5_boxes_nso_exp2_max3}

[ -f "$REPO/src/train_bash.py" ] || { echo "ERROR: $REPO is empty/not diffusion-vs-ar. Run: git submodule update --init"; exit 1; }
[ -f "$KS/data/boxes-dataset-v1.zip" ] || { echo "ERROR: $KS/data/boxes-dataset-v1.zip missing. Run: git submodule update --init"; exit 1; }
command -v unzip >/dev/null || { echo "ERROR: unzip not found"; exit 1; }
python -c "import transformers" 2>/dev/null || { echo "ERROR: transformers not importable; activate the venv (see make_env.sh)"; exit 1; }

TMP=${TMPDIR:-/tmp}/boxes_$(id -un); mkdir -p "$TMP"   # keep unzipped data OUT of any git repo (authors' request)
unzip -oq -P iamnotaLM "$KS/data/boxes-dataset-v1.zip" -d "$TMP"
python "$HERE/prep_boxes.py" --boxes_dir "$TMP/boxes-dataset-v1" --repo "$REPO" --split "$SPLIT"
python "$HERE/patch_metric.py" "$REPO"
mkdir -p "$REPO/scripts/boxes"
cp "$HERE"/scripts_boxes/* "$REPO/scripts/boxes/"
echo "ready. smoke test:  cd $REPO && NGPU=1 EPOCHS=2 bash scripts/boxes/train-mdm.sh"
