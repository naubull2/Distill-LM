#!/bin/bash

torchrun --nproc_per_node=1 train.py \
  --distillation_dataset ./distillation_trainset \
  --model_name torch-models/Qwen2.5-0.5B-Instruct \
  --output_dir ./distilled_model \
  --prune_ratio_heads 0.5 \
  --prune_ratio_layers 0.5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8
