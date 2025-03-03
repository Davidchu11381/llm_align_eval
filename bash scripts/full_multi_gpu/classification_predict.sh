#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
      --config_file ../accelerate/single_config.yaml \
      --num_processes 4 \
      --main_process_port 29501 \
      ../../src/train_bash.py \
      --stage sft \
      --do_predict \
      --model_name_or_path /home/mhchu/llama3/ckps-edtwt/llama3_cls \
      --dataset comm_cls_pred_ft \
      --dataset_dir ../../data \
      --template llama3 \
      --finetuning_type full \
      --output_dir /home/mhchu/llama3/prediction/cls/ft \
      --overwrite_cache \
      --overwrite_output_dir \
      --cutoff_len 1024 \
      --preprocessing_num_workers 32 \
      --per_device_eval_batch_size 40 \
      --predict_with_generate