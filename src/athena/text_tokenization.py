import pandas as pd
import torch
from transformers import AutoTokenizer
from pathlib import Path


def load_text_data(csv_path, image_id_column="image_id", text_column="annotation"):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned annotation CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = {image_id_column, text_column} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {sorted(missing_columns)}")

    print(f"[INFO] Loaded {len(df)} text descriptions.")
    return df[image_id_column].astype(str).tolist(), df[text_column].astype(str).tolist()


def tokenize_texts(texts, tokenizer, max_length=128):
    tokenized_outputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    print(f"[INFO] Tokenized {len(texts)} text descriptions.")
    return tokenized_outputs


def tokenize_csv(input_csv, output_pt, model_name="t5-small", max_length=128):
    print("=== Starting Text Tokenization ===")
    output_pt = Path(output_pt)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_ids, texts = load_text_data(input_csv)
    tokenized_texts = tokenize_texts(texts, tokenizer, max_length=max_length)

    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_ids": image_ids,
            "tokenized_texts": tokenized_texts,
            "model_name": model_name,
            "max_length": max_length,
        },
        output_pt,
    )
    print(f"[INFO] Tokenized text data saved to {output_pt}")

    print("=== Text Tokenization Complete ===")
    return output_pt


def main():
    from .defaults import DEFAULT_CLEANED_CSV, DEFAULT_TOKENIZED_TEXTS

    tokenize_csv(DEFAULT_CLEANED_CSV, DEFAULT_TOKENIZED_TEXTS)

if __name__ == "__main__":
    main()
