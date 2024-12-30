"""
build_dataset.py

Example usage:
  torchrun --nproc_per_node=4 build_dataset.py \
    --model_name myorg/my-large-32b-model \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --max_samples 1000 \
    --block_size 256 \
    --emb_type last \
    --save_path ./distillation_dataset
"""

import argparse
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="myorg/my-large-32b-model",
                        help="Path or name of the LLM to load (GPT-style).")
    parser.add_argument("--dataset_dir", type=str, default="path/to/my/data",
                        help="Dataset path name.")
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="Hugging Face dataset name.")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1",
                        help="Config name or subset for the dataset.")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum number of samples to process.")
    parser.add_argument("--block_size", type=int, default=256,
                        help="Maximum sequence length for tokenization.")
    parser.add_argument("--emb_type", type=str, choices=["first", "last"], default="last",
                        help="Which token's hidden state to store as teacher embedding.")
    parser.add_argument("--save_path", type=str, default="./distillation_dataset",
                        help="Output folder for the saved dataset.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # fallback for GPT-like

    # Load the large LLM in standard HF
    # For multi-GPU or large models, set device_map="auto" or use a distributed strategy (FSDP/DeepSpeed).
    print(f"Loading teacher model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        return_dict_in_generate=True,
        output_hidden_states=True  # crucial for extracting embeddings
    )
    model.eval()

    # Load dataset
    print(f"Loading dataset: {args.dataset_name} ({args.dataset_config_name})")
    #ds = load_dataset(args.dataset_name, args.dataset_config_name, split="train")
    ds = load_dataset("text", data_dir=args.dataset_dir, split="train")
    if len(ds) > args.max_samples:
        ds = ds.select(range(args.max_samples))

    # Tokenize
    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, max_length=args.block_size)

    ds = ds.map(tokenize_fn, batched=False, remove_columns=["text"])

    # Prepare lists
    input_ids_list = []
    attention_mask_list = []
    teacher_emb_list = []

    # Process each sample
    for i, record in enumerate(ds):
        # Convert to tensors
        input_ids = torch.tensor(record["input_ids"], dtype=torch.long).unsqueeze(0).to('mps')
        attention_mask = torch.tensor(record["attention_mask"], dtype=torch.long).unsqueeze(0).to('mps')

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # `outputs.hidden_states` is a tuple of length [num_layers+1], each shape [batch, seq_len, hidden_dim].
            # The last element in that tuple is the final hidden state of shape [1, seq_len, hidden_dim].
            final_hidden_state = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]

        # Extract either first or last token's embedding
        if args.emb_type == "first":
            emb = final_hidden_state[:, 0, :]  # shape [1, hidden_dim]
        else:
            # Find the last non-padding token or just the last position
            seq_len = input_ids.shape[1]
            # or simply last position: final_hidden_state[:, -1, :]
            # If you want "last non-padding", do:
            last_valid_idx = attention_mask.sum(dim=1) - 1
            emb = []
            for b_idx in range(final_hidden_state.size(0)):
                emb.append(final_hidden_state[b_idx, last_valid_idx[b_idx], :].unsqueeze(0))
            emb = torch.cat(emb, dim=0)  # shape [1, hidden_dim]

        emb = emb.squeeze(0).cpu().numpy()

        input_ids_list.append(record["input_ids"])
        attention_mask_list.append(record["attention_mask"])
        teacher_emb_list.append(emb)

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(ds)} samples")

    # Create new dataset
    new_ds = Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "teacher_embedding": teacher_emb_list
    })

    # Save
    print(f"Saving distillation dataset to {args.save_path}")
    new_ds.save_to_disk(args.save_path)
    print("Done.")

if __name__ == "__main__":
    main()

