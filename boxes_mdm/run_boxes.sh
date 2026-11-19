#!/usr/bin/env bash
# Portable end-to-end run: data prep (once) -> train -> evaluate -> score.
# Works on CPU only, one GPU, or several GPUs (auto-detected). Use from a laptop, Colab, or a cluster.
#
#   bash boxes_mdm/run_boxes.sh                      # full run (MODE=mdm), settings below
#   QUICK=1 bash boxes_mdm/run_boxes.sh              # small sanity run (4000 train / 500 test examples, 3 epochs)
#   MODE=ar bash boxes_mdm/run_boxes.sh              # autoregressive baseline
#
# Knobs (env vars): MODE=mdm|ar  EPOCHS  MODEL  LR  BS  T  MAX_TRAIN  MAX_TEST  VAL_SIZE  NGPU  QUICK
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
REPO=$ROOT/diffusion-vs-ar

NGPU=${NGPU:-$(python -c "import torch; print(torch.cuda.device_count())")}
if [ "$NGPU" -ge 1 ]; then DEV="$NGPU GPU(s)"; DEF_BS=128; DEF_EP=100; else DEV="CPU"; DEF_BS=16; DEF_EP=3; fi
if [ "${QUICK:-0}" = "1" ]; then MAX_TRAIN=${MAX_TRAIN:-4000}; MAX_TEST=${MAX_TEST:-500}; DEF_EP=3; fi

MODE=${MODE:-mdm}; EPOCHS=${EPOCHS:-$DEF_EP}; MODEL=${MODEL:-model_config_tiny_boxes}
LR=${LR:-1e-3}; BS=${BS:-$DEF_BS}; T=${T:-20}; VAL_SIZE=${VAL_SIZE:-448}
MAX_TRAIN=${MAX_TRAIN:-}; MAX_TEST=${MAX_TEST:-}
WORKERS=$(python -c "import os; print(min(8, os.cpu_count() or 1))")
[ "$MODE" = "mdm" ] || [ "$MODE" = "ar" ] || { echo "MODE must be mdm or ar"; exit 1; }
echo ">> device: $DEV | mode: $MODE | epochs: $EPOCHS | batch/device: $BS | train cap: ${MAX_TRAIN:-all} | test cap: ${MAX_TEST:-all}"

# 1. one-time setup (needs the submodules); patches are idempotent, so always re-applied
if [ ! -f "$REPO/src/train_bash.py" ]; then
    git -C "$ROOT" submodule update --init || { echo "ERROR: run 'git clone --recursive' or 'git submodule update --init'"; exit 1; }
fi
[ -f "$REPO/data/boxes_lengths.json" ] || bash "$HERE/setup_boxes.sh"
python "$HERE/patch_metric.py" "$REPO" >/dev/null
python "$HERE/patch_device.py" "$REPO" >/dev/null
mkdir -p "$REPO/scripts/boxes"; cp "$HERE"/scripts_boxes/score_boxes.py "$REPO/scripts/boxes/"

cd "$REPO"
export WANDB_DISABLED=true
CUT=$(python -c "import json;print(json.load(open('data/boxes_lengths.json'))['cutoff_len'])")
NEW=$(python -c "import json;print(json.load(open('data/boxes_lengths.json'))['max_new_tokens'])")
exp=output/boxes/$MODE-bs$BS-lr$LR-ep$EPOCHS-$(date +%Y%m%d-%H%M%S)
mkdir -p "$exp"

STAGE=$([ "$MODE" = "mdm" ] && echo mdm || echo sft)
MDM_TRAIN=(); MDM_EVAL=(); AR_EVAL=()
if [ "$MODE" = "mdm" ]; then
    MDM_TRAIN=(--diffusion_steps "$T" --save_safetensors False --token_reweighting True --time_reweighting linear --topk_decoding True --alpha 0.25 --gamma 1)
    MDM_EVAL=(--diffusion_steps "$T" --topk_decoding True --decoding_strategy stochastic0.5-linear)
else
    AR_EVAL=(--predict_with_generate --max_new_tokens "$NEW")
fi

# launcher: plain python on CPU / 1 GPU, accelerate for several GPUs
if [ "$NGPU" -gt 1 ]; then
    LAUNCH=(accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes "$NGPU" --main_process_port "${PORT:-20099}")
else
    LAUNCH=(python)
fi
FP16=(); [ "$NGPU" -ge 1 ] && FP16=(--fp16)
CAP_TRAIN=(); [ -n "$MAX_TRAIN" ] && CAP_TRAIN=(--max_samples "$MAX_TRAIN")
CAP_TEST=();  [ -n "$MAX_TEST" ]  && CAP_TEST=(--max_samples "$MAX_TEST")

echo ">> training (log: $exp/train.log)"
"${LAUNCH[@]}" src/train_bash.py --stage "$STAGE" --overwrite_output_dir --cache_dir ./cache \
    --model_name_or_path "$MODEL" --do_train --dataset boxes_train --finetuning_type full \
    --cutoff_len "$CUT" --output_dir "$exp" --overwrite_cache \
    --per_device_train_batch_size "$BS" --gradient_accumulation_steps 1 --lr_scheduler_type cosine \
    --logging_steps 1 --val_size "$VAL_SIZE" --per_device_eval_batch_size 32 \
    --evaluation_strategy steps --eval_steps "${EVAL_STEPS:-100}" --save_steps "${SAVE_STEPS:-500}" \
    --learning_rate "$LR" --num_train_epochs "$EPOCHS" --plot_loss --run_name "boxes_$MODE" \
    --preprocessing_num_workers "$WORKERS" --save_total_limit 1 --remove_unused_columns False \
    "${FP16[@]}" "${MDM_TRAIN[@]}" "${CAP_TRAIN[@]}" > "$exp/train.log" 2>&1 \
    || { echo "TRAINING FAILED, last lines of $exp/train.log:"; tail -25 "$exp/train.log"; exit 1; }

echo ">> evaluating on boxes_test (log: $exp/boxes_test/eval.log)"
mkdir -p "$exp/boxes_test"
CUDA_VISIBLE_DEVICES=$([ "$NGPU" -ge 1 ] && echo 0 || echo "") \
python -u src/train_bash.py --stage "$STAGE" --overwrite_output_dir --cache_dir ./cache \
    --model_name_or_path "$MODEL" --do_predict --cutoff_len "$CUT" --dataset boxes_test \
    --finetuning_type full --output_dir "$exp/boxes_test" --checkpoint_dir "$exp" \
    --per_device_eval_batch_size "${EVAL_BS:-$([ "$NGPU" -ge 1 ] && echo 128 || echo 32)}" --remove_unused_columns False \
    "${MDM_EVAL[@]}" "${AR_EVAL[@]}" "${CAP_TEST[@]}" > "$exp/boxes_test/eval.log" 2>&1 \
    || { echo "EVAL FAILED, last lines of $exp/boxes_test/eval.log:"; tail -25 "$exp/boxes_test/eval.log"; exit 1; }

echo ">> scores"
python scripts/boxes/score_boxes.py --pred "$exp/boxes_test/generated_predictions.jsonl" \
    --meta data/boxes_test.meta.jsonl --truncate | tee "$exp/scores.txt"
echo "outputs in $REPO/$exp"
