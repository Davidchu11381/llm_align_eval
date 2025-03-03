#!/bin/bash

# TRAINING

# Define an array of datasets
#datasets=("ed" "anti_ed" "body_image" "lifestyle" "keto" "drugs")
datasets=("anti_ed")

# Iterate over each dataset in the array
for dataset in "${datasets[@]}"; do

    dir=../../ckps-edtwt/llama3_instruct/$dataset

    deepspeed --include localhost:0,1,2,3 --master_port 29500 ../../src/train_bash.py \
        --deepspeed ../deepspeed/ds_z3_config.json \
        --stage sft \
        --do_train \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset alpaca_gpt4_en,$dataset \
        --dataset_dir ../../data \
        --template llama3 \
        --finetuning_type full \
        --output_dir $dir \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len 1024 \
        --preprocessing_num_workers 32 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --warmup_steps 20 \
        --save_steps 2000 \
        --eval_steps 2000 \
        --evaluation_strategy steps \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --val_size 0.1 \
        --ddp_timeout 180000000 \
        --plot_loss \
        --fp16 \
        --save_total_limit 0

    # Remove the checkpoint directories
    rm -rf $dir/checkpoint*
done



