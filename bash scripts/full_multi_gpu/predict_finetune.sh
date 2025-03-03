#!/bin/bash

# Declare an array of datasets
datasets=("ed_infer" "anti_ed_infer" "body_image_infer" "lifestyle_infer" "keto_infer" "drugs_infer")
#datasets=("ed_infer" "anti_ed_infer" "body_image_infer")
# Declare an array of corresponding output directories
output_dirs=("ed" "anti_ed" "body_image" "lifestyle" "keto" "drugs")
#output_dirs=("ed" "anti_ed" "body_image")

# Loop through the datasets and output directories
for (( i=0; i<${#datasets[@]}; i++ )); do
    dataset=${datasets[$i]}
    output_dir=${output_dirs[$i]}
    model_path="/dev/shm/dchu/ckps-edtwt/llama3_instruct/${output_dir}"
    output_path="/home/mhchu/llama3/prediction/finetune/${output_dir}"

    echo "Processing $dataset..."

    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        --config_file ../accelerate/single_config.yaml \
        --main_process_port 29500 \
        --num_processes 4 \
        ../../src/train_bash.py \
        --stage sft \
        --do_predict \
        --model_name_or_path "$model_path" \
        --dataset "$dataset" \
        --dataset_dir ../../data \
        --template llama3 \
        --finetuning_type full \
        --output_dir "$output_path" \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len 1024 \
        --preprocessing_num_workers 2 \
        --per_device_eval_batch_size 64 \
        --predict_with_generate
done

echo "All processes completed."
