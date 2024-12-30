# Distill LLM to BERT embedding models

By distilling a high-performance LLM down to a small(6-12 layer BERT) embedding model, we can efficiently build a powerful classifier or an embedding model with much less compute.

# How to Run

## 1. Preparing the Train Data

- `args.emb_type` lets you choose whether to store the first token's hidden state or the last token's hidden state.

- Use a distributed approach (`torchrun`, Deepspeed, etc.) to handle large models if your model don't fit in a single GPU memory:

```bash
torchrun --nproc_per_node=4 build_dataset.py \
  --model_name workspace/my-large-model \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --block_size 256 \
  --emb_type last \
  --save_path ./distillation_trainset
```

### Output

This gives you a dataset containing:

- `input_ids`/`attention_mask` (original tokens)
- `teacher_embedding` (the single-vector teacher output for each sequence)


You can now train a *BERT-like* student model to match these embeddings (e.g., MSE or Cosine Similarity loss) instead of an entire seq2seq teacher distribution.

## 2. Training a Student Model

There are numerous ways to distill a large transformer model down to a BERT-like model:

1. Attention Pruning: Removes redundant attention heads, directly reducing complexity.
2. Width Reduction: Adds a projection layer to shrink hidden dimensions.
3. Layer Pruning: Removes less important or redundant layers of the original model.


*Example using torchrun*
```bash
torchrun --nproc_per_node=4 train.py \
  --distillation_dataset ./distillation_dataset \
  --model_name myorg/my-large-32b-model \
  --output_dir ./distilled_model \
  --prune_ratio_heads 0.5 \
  --prune_ratio_layers 0.5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2
```

*Example using deepspeed*
```bash
deepspeed train.py --deepspeed_config ds_config.json \
  --distillation_dataset ./distillation_dataset \
  --model_name myorg/my-large-32b-model \
  --output_dir ./distilled_model \
  --prune_ratio_heads 0.5 \
  --prune_ratio_layers 0.5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2
```



