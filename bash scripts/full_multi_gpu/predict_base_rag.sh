# Declare an array of datasets
datasets=("ed_rag" "anti_ed_rag" "body_image_rag" "lifestyle_rag" "keto_rag" "drugs_rag")

# Declare an array of corresponding output directories
output_dirs=("ed" "anti_ed" "body_image" "lifestyle" "keto" "drugs")

# Loop over the datasets
for i in "${!datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
      --main_process_port 29501 \
      --config_file ../accelerate/single_config.yaml \
      --num_processes 4 \
      ../../src/train_bash.py \
      --stage sft \
      --do_predict \
      --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
      --dataset "${datasets[$i]}" \
      --dataset_dir ../../data \
      --template jailbreak_llama3 \
      --finetuning_type full \
      --output_dir /home/mhchu/llama3/prediction/rag/${output_dirs[$i]} \
      --overwrite_cache \
      --overwrite_output_dir \
      --cutoff_len 8000 \
      --preprocessing_num_workers 32 \
      --per_device_eval_batch_size 4 \
      --predict_with_generate
done