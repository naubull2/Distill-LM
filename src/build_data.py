# coding:utf-8
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from itertools import chain

from tqdm import tqdm


def create_distillation_dataset(raw_data_dir, output_path, model_name_or_path, max_length=512, device="cuda"):
    """
    Create a distillation dataset by generating teacher outputs for student training.
    
    Args:
        raw_data_dir       (str): Directory containing CSV,JSONL files with instructions/text data.
        output_path        (str): Path to save result in a format compatible with `datasets.load_dataset`.
        model_name_or_path (str): Name of the teacher model to use for inference.
        max_length         (int): Maximum input sequence length.
    """
    # Initialize the teacher model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    model.eval()
    
    # Read and combine all CSV / JSONL files from the raw data directory
    all_texts = []
    for file in tqdm(chain(
        Path(raw_data_dir).rglob("*.csv"),
        Path(raw_data_dir).rglob("*.jsonl")
    )):
        df = pd.read_csv(file) if file.suffix == ".csv" else pd.read_json(file, lines=True)
        if "instruction" in df.columns:
            all_texts.extend(df["instruction"].tolist())
        elif "text" in df.columns:
            all_texts.extend(df["text"].tolist())
        elif "conversations" in df.columns:
            for lst in df.conversations.tolist():
                all_texts.extend(lst)
        else:
            raise ValueError(f"'instruction' or 'text' column missing in {str(file)}")
    
    print(f"Loaded {len(all_texts):,} instructions from {raw_data_dir}")
    
    # Tokenize and generate teacher outputs (soft labels)
    processed_data = {"input": [], "teacher_logits": []}
    for idx, text in enumerate(all_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # Extract the last-token logits

        # Store the input text and soft labels
        processed_data["input"].append(text)
        processed_data["teacher_logits"].append(logits.squeeze(0).cpu().numpy().tolist())
        
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(all_texts)} instructions")
    
    # Convert processed data to a Hugging Face Dataset and save
    dataset = Dataset.from_dict(processed_data)
    dataset.save_to_disk(output_path)
    print(f"Distillation dataset saved to {output_path}")


def cli(
    data_path: str, # "./data/source"
    output_path: str = "./data/distillation",
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    max_length: int = 1024,
    device: str = "cuda"
):
    """ Create a distillation dataset
      data_path   (str): Directory containing files with instructions or text data.
      output_path (str): Path to save the generated dataset.
      model_name  (str): Name of the teacher model to use for inference.
      max_length  (int): Maximum input sequence length.
      device      (str): cuda, mps, cpu
    """
    create_distillation_dataset(data_path, output_path, model_name)


if __name__=="__main__":
    import fire
    fire.Fire(cli)
