#!/bin/bash

# Declare an array of datasets
datasets=("ed_neda" "anti_ed_neda" "body_image_neda" "lifestyle_neda" "keto_neda" "drugs_neda")

# Declare an array of corresponding output directories
output_dirs=("ed" "anti_ed" "body_image" "lifestyle" "keto" "drugs")

# Loop through the arrays
for (( i=0; i<${#datasets[@]}; i++ )); do
    dataset=${datasets[$i]}
    output_dir=${output_dirs[$i]}

    echo "Processing dataset: $dataset with output directory: $output_dir"

    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        --config_file ../accelerate/single_config.yaml \
        --num_processes 4 \
        --main_process_port 29500 \
        ../../src/train_bash.py \
        --stage sft \
        --do_predict \
        --model_name_or_path "/home/mhchu/llama3/ckps-edtwt/llama3_instruct/$output_dir" \
        --dataset "$dataset" \
        --dataset_dir ../../data \
        --template llama3 \
        --finetuning_type full \
        --output_dir "/home/mhchu/llama3/prediction/neda/finetune/$output_dir" \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len 1024 \
        --preprocessing_num_workers 32 \
        --per_device_eval_batch_size 64 \
        --predict_with_generate
done

echo "All datasets have been processed."
