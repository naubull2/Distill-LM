import torch
import numpy as np
from transformers import AutoModel

def prune_attention_heads(model, layer_to_heads):
    """
    Prunes specified attention heads from each layer.
    
    Args:
        model: Hugging Face transformer model
        layer_to_heads: Dictionary specifying heads to prune for each layer.
                        Example: {0: [0, 1], 2: [3]}  # Prune heads 0, 1 from layer 0, and head 3 from layer 2
    
    Returns:
        Pruned model.
    """
    for layer_idx, heads_to_prune in layer_to_heads.items():
        model.encoder.layer[layer_idx].attention.self.prune_heads(set(heads_to_prune))
    print("Pruning completed.")
    return model

"""
# Example usage
teacher_model_name = "bert-base-uncased"  # Replace with your model
model = AutoModel.from_pretrained(teacher_model_name)

# Define which heads to prune (layer index: [head indices])
layer_to_heads = {0: [0, 1], 1: [2], 2: [0, 1, 2]}  # Example
pruned_model = prune_attention_heads(model, layer_to_heads)
"""



def calculate_attention_entropy(attention_weights):
    # Calculate entropy for each head
    head_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-12), dim=-1)
    return head_entropy.mean(dim=-2).cpu().numpy()  # Mean across tokens


if __name__=="__main__":
    model = AutoModel.from_pretrained("bert-base-uncased")
    input_ids = torch.tensor([[101, 2009, 2003, 1037, 2742, 102]])  # Example input

    with torch.no_grad():
        outputs = model(input_ids)
        for layer_idx, layer in enumerate(outputs.attentions):
            print(f"Layer {layer_idx} entropy: {calculate_attention_entropy(layer)}")


