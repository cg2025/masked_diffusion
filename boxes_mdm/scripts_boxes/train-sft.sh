# AR baseline. usage: NGPU=8 EPOCHS=100 MODEL=model_config_tiny_boxes LR=1e-3 bash scripts/boxes/train-sft.sh
export WANDB_DISABLED=true
NGPU=${NGPU:-8}; EPOCHS=${EPOCHS:-100}; MODEL=${MODEL:-model_config_tiny_boxes}; LR=${LR:-1e-3}; BS=${BS:-128}
CUT=$(python -c "import json;print(json.load(open('data/boxes_lengths.json'))['cutoff_len'])")
NEW=$(python -c "import json;print(json.load(open('data/boxes_lengths.json'))['max_new_tokens'])")
GPUS=$(seq -s, 0 $((NGPU-1)))

exp=output/boxes/ar-bs$((BS*NGPU))-lr$LR-ep$EPOCHS-`date "+%Y%m%d-%H%M%S"`
mkdir -p $exp

CUDA_VISIBLE_DEVICES=$GPUS \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes $NGPU \
src/train_bash.py \
    --stage sft --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path $MODEL \
    --do_train \
    --dataset boxes_train \
    --finetuning_type full \
    --cutoff_len $CUT \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size $BS \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 500 \
    --learning_rate $LR \
    --num_train_epochs $EPOCHS \
    --plot_loss \
    --run_name boxes_ar \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    > $exp/train.log

for dataset in boxes_test
do
mkdir -p $exp/$dataset
CUDA_VISIBLE_DEVICES=0 \
python3 -u src/train_bash.py \
    --stage sft --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path $MODEL \
    --do_predict \
    --cutoff_len $CUT \
    --per_device_eval_batch_size 128 \
    --dataset $dataset \
    --finetuning_type full \
    --output_dir $exp/${dataset} \
    --checkpoint_dir $exp \
    --predict_with_generate \
    --max_new_tokens $NEW \
    > $exp/${dataset}/eval.log
python scripts/boxes/score_boxes.py --pred $exp/$dataset/generated_predictions.jsonl --meta data/boxes_test.meta.jsonl
done
