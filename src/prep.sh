#!/bin/bash
torchrun --nproc_per_node=1 build_dataset.py \
  --model_name torch-models/Qwen2.5-0.5B-Instruct \
  --dataset_dir ./sample_data \
  --block_size 256 \
  --emb_type last \
  --save_path ./distillation_trainset
