import argparse
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)

###################################
# 1) Manual Head Pruning for QWen
###################################
def custom_prune_heads_in_qwen_attention(attn_module, heads_to_prune: set):
    """
    Manually prunes heads in a Qwen2Attention module that defines:
      (q_proj): [in=896, out=896]
      (k_proj): [in=896, out=128]
      (v_proj): [in=896, out=128]
      (o_proj): [in=896, out=896]
    
    We'll assume:
      num_heads_q = attn_module.num_heads   # e.g., 7 heads for queries
      head_dim_q  = 896 // num_heads_q      # e.g., 128
      num_heads_kv = ??? (depends on config)
      head_dim_kv  = ??? (depends on config)
    
    Steps:
      1) Slice q_proj.weight/bias to keep only heads_to_keep for queries.
      2) Slice k_proj, v_proj similarly to keep the same heads for K/V if the config uses the same num_heads 
         or adapt if there's a separate num_heads_kv.
      3) Slice o_proj accordingly.
      4) Update attn_module.num_heads, etc.
    
    Note: The code below assumes (for example) the same num_heads 
          for Q and K/V. If your Qwen config differs, adapt the slicing.
    """
    if not heads_to_prune:
        return

    device = next(attn_module.parameters()).device
    old_num_heads = attn_module.num_heads
    old_num_kv_heads = attn_module.num_key_value_heads
    print(f"n_heads: {old_num_heads}, n_kv_heads: {old_num_kv_heads}")
    # Example: out_features=896 for q_proj => queries dimension
    #          so head_dim_q = 896 / old_num_heads
    head_dim_q = attn_module.q_proj.out_features // old_num_heads  # e.g., 128

    # For K/V, out_features=128 => total dimension for all heads
    # So if K shares the same num_heads, we do head_dim_k = 128 / old_num_heads
    head_dim_k = attn_module.k_proj.out_features // old_num_kv_heads  # e.g., 128 / 7 => ???

    # Which heads we keep
    heads_to_keep = sorted(set(range(old_num_heads)) - heads_to_prune)
    new_num_heads = len(heads_to_keep)
    new_num_kv_heads = (new_num_heads * old_num_kv_heads)//old_num_heads 
    if new_num_heads == 0:
        return  # pruning all heads would break the module

    # --------------------------------
    # 1) q_proj
    # q_proj.weight: [out_features=896, in_features=896]
    # q_proj.bias:   [out_features=896]
    # We interpret out_features as (num_heads_q * head_dim_q).
    W_q = attn_module.q_proj.weight  # shape [896, 896]
    b_q = attn_module.q_proj.bias    # shape [896]

    # Reshape => [num_heads_q, head_dim_q, in_features=896]
    W_q = W_q.view(old_num_heads, head_dim_q, -1)
    b_q = b_q.view(old_num_heads, head_dim_q)

    # Slice out the kept heads
    W_q = W_q[heads_to_keep, :, :]  # => [new_num_heads, head_dim_q, in_features]
    b_q = b_q[heads_to_keep, :]

    # Reshape back => [new_num_heads * head_dim_q, in_features]
    W_q = W_q.view(new_num_heads * head_dim_q, -1)
    b_q = b_q.view(new_num_heads * head_dim_q)

    # Assign
    attn_module.q_proj.weight = nn.Parameter(W_q.to(device))
    attn_module.q_proj.bias   = nn.Parameter(b_q.to(device))

    # --------------------------------
    # 2) k_proj
    # k_proj.weight: [out_features=128, in_features=896]
    # k_proj.bias:   [out_features=128]
    # out_features => (num_heads_kv * head_dim_k), assume num_heads_kv == old_num_heads for demonstration
    W_k = attn_module.k_proj.weight  # [128, 896]
    b_k = attn_module.k_proj.bias    # [128]

    W_k = W_k.view(old_num_kv_heads, head_dim_k, -1)  # => [num_heads_kv, head_dim_k, in_features=896]
    b_k = b_k.view(old_num_kv_heads, head_dim_k)

    W_k = W_k[heads_to_keep, :, :]
    b_k = b_k[heads_to_keep, :]

    W_k = W_k.view(new_num_heads * head_dim_k, -1)
    b_k = b_k.view(new_num_heads * head_dim_k)

    attn_module.k_proj.weight = nn.Parameter(W_k.to(device))
    attn_module.k_proj.bias   = nn.Parameter(b_k.to(device))

    # --------------------------------
    # 3) v_proj
    W_v = attn_module.v_proj.weight  # [128, 896]
    b_v = attn_module.v_proj.bias    # [128]

    W_v = W_v.view(old_num_kv_heads, head_dim_k, -1)
    b_v = b_v.view(old_num_kv_heads, head_dim_k)

    W_v = W_v[heads_to_keep, :, :]
    b_v = b_v[heads_to_keep, :]

    W_v = W_v.view(new_num_heads * head_dim_k, -1)
    b_v = b_v.view(new_num_heads * head_dim_k)

    attn_module.v_proj.weight = nn.Parameter(W_v.to(device))
    attn_module.v_proj.bias   = nn.Parameter(b_v.to(device))

    # --------------------------------
    # 4) o_proj
    # o_proj.weight: [out_features=896, in_features=896]
    # We interpret in_features as (num_heads_kv * head_dim_k) => 128, out_features => (num_heads_q * head_dim_q) => 896
    # So we slice the "input" dimension to remove pruned heads from K/V,
    # but keep all queries in the "output" dimension?
    # If Q & K/V heads are the same set, adapt the slice for in_features.

    W_o = attn_module.o_proj.weight  # shape [896, 896]
    b_o = attn_module.o_proj.bias    # shape [out_features=896 or None if bias=False]

    # out_features => old_num_heads * head_dim_q = 896
    # in_features => old_num_heads * head_dim_k = 128
    out_features_o = W_o.size(0)  # 896
    in_features_o  = W_o.size(1)  # 896

    # We'll interpret in_features_o as (old_num_heads * head_dim_k).
    # Reshape => [out_features_o, old_num_heads, head_dim_k]
    W_o = W_o.view(out_features_o, old_num_heads, head_dim_k)

    W_o = W_o[:, heads_to_keep, :]  # keep only heads_to_keep in the 'in_features' dimension
    W_o = W_o.view(out_features_o, new_num_heads * head_dim_k)

    attn_module.o_proj.weight = nn.Parameter(W_o.to(device))
    # Some Qwen2Attentions have no bias in o_proj, check if b_o is not None
    if b_o is not None:
        attn_module.o_proj.bias = nn.Parameter(b_o.to(device))

    # --------------------------------
    # 5) Update attn_module.num_heads
    attn_module.num_heads = new_num_heads
    attn_module.num_key_value_heads = new_num_kv_heads
    attn_module.num_key_value_groups = new_num_heads // new_num_kv_heads



###################################
# 2) Compute Attention Entropy
###################################
def safe_compute_entropy(attn_probs, eps=1e-12):
    """
    attn_probs: [batch, n_heads, seq_len, seq_len], in [0,1].
    Returns: [batch, n_heads, seq_len] of entropies.
    """
    # Clamp attention to avoid log(0)
    attn_probs = attn_probs.clamp(min=eps)
    log_attn = attn_probs.log()  # safe log
    entropy = -(attn_probs * log_attn).sum(dim=-1)  # sum over last dim
    # Optionally replace any lingering NaNs or infinities
    entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    return entropy


def compute_attention_entropy(model, dataset, max_samples=100, block_size=256):
    """
    For Qwen, model.model.layers is a list of QWenBlock.
    Each block has .attn => QWenAttention, 
    which can return attention weights if we do output_attentions=True.
    
    We'll do a small sample of data -> forward pass -> measure -sum(p log p).
    Return shape [num_layers, num_heads].
    """
    device = next(model.parameters()).device
    model.eval()

    num_layers = model.config.num_hidden_layers
    # We'll assume num_heads is consistent across layers
    num_heads = model.config.num_attention_heads

    head_entropies = torch.zeros(num_layers, num_heads, device=device)
    sample_count = 0

    for example in dataset:
        if sample_count >= max_samples:
            break
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
        if input_ids.size(1) > block_size:
            input_ids = input_ids[:, :block_size]
            attention_mask = attention_mask[:, :block_size]

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            # out.attentions => tuple of length num_layers 
            # each => [batch=1, n_heads, seq_len, seq_len]

        for layer_idx, attn in enumerate(out.attentions):
            # shape: [1, num_heads, seq_len, seq_len]
            eps = 1e-12
            #log_attn = (attn + eps).log()
            #entropy = -(attn * log_attn).sum(dim=-1)     # [1, num_heads, seq_len]
            entropy = safe_compute_entropy(attn, eps)
            entropy = entropy.mean(dim=(0, 2))           # [num_heads]
            head_entropies[layer_idx] += entropy

        sample_count += 1

    if sample_count > 0:
        head_entropies /= sample_count
    return head_entropies  # [num_layers, num_heads]


###################################
# 3) Prune Heads & Layers
###################################
def prune_attention_heads(model, head_entropies, prune_ratio_heads=0.5):
    """
    'High entropy => less important => remove'
    We'll keep heads with lower entropy.
    """
    num_layers, num_heads = head_entropies.shape
    keep_heads_per_layer = int((1 - prune_ratio_heads) * num_heads)

    importance = -head_entropies  # lower entropy => higher importance
    for layer_idx in range(num_layers):
        scores = importance[layer_idx]
        sorted_heads = torch.argsort(scores, descending=True)
        keep_heads = sorted_heads[:keep_heads_per_layer].tolist()
        all_heads = set(range(num_heads))
        prune_heads_set = set(all_heads - set(keep_heads))
        if prune_heads_set:
            attn_module = model.model.layers[layer_idx].self_attn
            custom_prune_heads_in_qwen_attention(attn_module, prune_heads_set)
            model.config.num_attention_heads = attn_module.num_heads
    return model

def prune_layers(model, prune_ratio_layers=0.5):
    """
    Qwen uses model.model.layers, a nn.ModuleList.
    We'll keep only the first X blocks (or whichever strategy you choose).
    """
    original_layers = model.model.layers
    num_original_layers = len(original_layers)
    keep_count = int((1 - prune_ratio_layers) * num_original_layers)
    new_layers = nn.ModuleList([original_layers[i] for i in range(keep_count)])
    model.model.layers = new_layers
    model.config.num_hidden_layers = keep_count
    return model


###################################
# 4) Distillation Model
###################################
class DistillationModel(nn.Module):
    def __init__(self, model, temperature=2.0, emb_type="last"):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.emb_type = emb_type

    def forward(self, input_ids, attention_mask=None, teacher_embedding=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = out.hidden_states[-1]

        if self.emb_type == "first":
            emb = student_logits[:, 0, :]  # shape [1, hidden_dim]
        else:
            # Find the last non-padding token or just the last position
            seq_len = input_ids.shape[1]
            # or simply last position: final_hidden_state[:, -1, :]
            # If you want "last non-padding", do:
            last_valid_idx = attention_mask.sum(dim=1) - 1
            emb = []
            for b_idx in range(student_logits.size(0)):
                emb.append(student_logits[b_idx, last_valid_idx[b_idx], :].unsqueeze(0))
            emb = torch.cat(emb, dim=0)  # shape [1, hidden_dim]
        student_embedding = emb

        loss = None
        if teacher_embedding is not None:
            T = self.temperature
            # Flatten
            s_log_probs = F.log_softmax(student_embedding.view(-1, student_embedding.size(-1)) / T, dim=-1)
            t_probs = F.softmax(teacher_embedding.view(-1, teacher_embedding.size(-1)) / T, dim=-1)
            loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T ** 2)
        return {"loss": loss, "logits": student_embedding}


###################################
# 5) Main Script
###################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distillation_dataset", type=str, default="./distillation_dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B", help="Qwen2.5 / Qwen-7B or similar")
    parser.add_argument("--output_dir", type=str, default="./distilled_model")
    parser.add_argument("--prune_ratio_heads", type=float, default=0.5)
    parser.add_argument("--prune_ratio_layers", type=float, default=0.5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--entropy_max_samples", type=int, default=100)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--emb_type", type=str, default="last")
    return parser.parse_args()

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # fallback for causal models

    # 1) Load Qwen model in HF
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_attentions=True,
        output_hidden_states=True,
        attn_implementation="eager"
    )

    # 2) Load offline distillation dataset
    ds = load_from_disk(args.distillation_dataset)
    ## 3) Compute actual attention entropy
    #print("Computing attention entropies (QWen style)...")
    #head_entropies = compute_attention_entropy(
    #    model, ds, max_samples=args.entropy_max_samples, block_size=args.block_size
    #)

    ## 4) Prune heads
    #print("Pruning attention heads...")
    #model = prune_attention_heads(model, head_entropies, prune_ratio_heads=args.prune_ratio_heads)

    # 5) Prune layers
    print("Pruning layers...")
    teacher_size = model.num_parameters()
    model = prune_layers(model, args.prune_ratio_layers)
    student_size = model.num_parameters()
    print(f"Pruning down to {100* student_size/teacher_size:.2f}%")

    # 6) Distillation wrapper
    distill_model = DistillationModel(model, temperature=args.temperature, emb_type=args.emb_type)

    # 7) Data collator
    def collate_fn(examples):
        input_ids = [torch.tensor(e["input_ids"], dtype=torch.long) for e in examples]
        attention_masks = [torch.tensor(e["attention_mask"], dtype=torch.long) for e in examples]
        teacher_logits = [torch.tensor(e["teacher_embedding"], dtype=torch.float) for e in examples]

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_masks = nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        teacher_logits = torch.stack(teacher_logits, dim=0)  # [B, seq_len, vocab_size]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "teacher_embedding": teacher_logits
        }
    # 8) Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        dataloader_drop_last=True,
        report_to="none"
    )

    trainer = Trainer(
        model=distill_model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=collate_fn
    )

    # 9) Train
    trainer.train()

    # 10) Save final
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done. Pruned & distilled Qwen model saved.")

if __name__ == "__main__":
    main()

