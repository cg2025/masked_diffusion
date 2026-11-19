#!/usr/bin/env bash
# usage: bash setup_boxes.sh [REPO=~/diffusion-vs-ar] [KS_REPO=~/entity-tracking-lms] [SPLIT=t5_boxes_nso_exp2_max3]
# Run AFTER downloading the diffusion-vs-ar data (Google Drive link in its README) into $REPO/data.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${1:-$HOME/diffusion-vs-ar}; KS=${2:-$HOME/entity-tracking-lms}; SPLIT=${3:-t5_boxes_nso_exp2_max3}
TMP=${TMPDIR:-/tmp}/boxes_$USER; mkdir -p $TMP      # keep unzipped data OUT of any git repo (authors' request)
unzip -oq -P iamnotaLM $KS/data/boxes-dataset-v1.zip -d $TMP
python $HERE/prep_boxes.py --boxes_dir $TMP/boxes-dataset-v1 --repo $REPO --split $SPLIT
python $HERE/patch_metric.py $REPO
mkdir -p $REPO/scripts/boxes
cp $HERE/scripts_boxes/* $REPO/scripts/boxes/
echo "ready. smoke test:  cd $REPO && NGPU=1 EPOCHS=2 bash scripts/boxes/train-mdm.sh"
