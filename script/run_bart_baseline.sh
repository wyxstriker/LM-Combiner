#!/bin/bash

NUM_GPU=1

export OMP_NUM_THREADS=2
task_name=cbart_large

torchrun --nproc_per_node ${NUM_GPU} \
    --master_port 31442 \
    ./src/main.py \
    --model_name_or_path ./pt_model/bart-large \
    --do_train \
    --do_eval \
    --num_train_epochs 10 \
    --train_file ./data/fcgec/train.json \
    --validation_file ./data/fcgec/dev.json \
    --output_dir ./output/${task_name}/ \
    --cache_dir ./cache \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --max_source_length 256 \
    --max_target_length 512 \
    --overwrite_output_dir \
    --seed 42 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.01 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_steps 1000 \
    --lr_scheduler_type polynomial \
    --num_beams 1 \
    --save_total_limit 3 \
    --bf16 \
    --logging_strategy steps \
    --include_inputs_for_metrics True \
    --predict_with_generate True \
    --logging_steps 50 \
    --load_best_model_at_end True \
    --generation_max_length 512 \
    --generation_num_beams 1 \
    --metric_for_best_model F05 \
    --greater_is_better True
    # --label_smoothing_factor 0.1
    # --save_strategy epoch \
    # --evaluation_strategy epoch \