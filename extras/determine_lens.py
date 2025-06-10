import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Analyze token lengths for prompt/target fields in a dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to your dataset (json or jsonl)")
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True, help="HF model name or path")
    parser.add_argument("--input_field", type=str, default="src_fm_fc_ms", help="Key for input prompt/context")
    parser.add_argument("--target_field", type=str, default="target", help="Key for target/response")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples for speed (optional)")
    args = parser.parse_args()

    # Load dataset
    data_path = Path(args.data_path)
    if data_path.suffix == ".jsonl":
        with open(data_path) as f:
            data = [json.loads(line) for line in f]
    elif data_path.suffix == ".json":
        with open(data_path) as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported file format. Use .json or .jsonl")

    if args.limit:
        data = data[:args.limit]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)

    input_lengths = []
    target_lengths = []

    for example in tqdm(data, desc="Processing"):
        input_text = example[args.input_field]
        target_text = example[args.target_field]

        # If you want to apply a chat template, do it here (uncomment next lines):
        messages = [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": target_text}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        input_ids = tokenizer(input_text, truncation=False, add_special_tokens=True)["input_ids"]
        target_ids = tokenizer(target_text, truncation=False, add_special_tokens=True)["input_ids"]
        input_lengths.append(len(input_ids))
        target_lengths.append(len(target_ids))

    def stats(lengths, name):
        print(f"\n--- {name} ---")
        print(f"Mean: {np.mean(lengths):.2f}")
        print(f"Median: {np.median(lengths)}")
        print(f"95th percentile: {int(np.percentile(lengths, 95))}")
        print(f"99th percentile: {int(np.percentile(lengths, 99))}")
        print(f"Max: {max(lengths)}")

    stats(input_lengths, "Input (prompt) lengths")
    stats(target_lengths, "Target (response) lengths")

    suggested_input = int(np.percentile(input_lengths, 95) + 7) // 8 * 8  # round up to nearest 8
    suggested_target = int(np.percentile(target_lengths, 95) + 7) // 8 * 8

    print(f"\n>>> Suggested max_input_len: {suggested_input}")
    print(f">>> Suggested max_target_len: {suggested_target}")

if __name__ == "__main__":
    main()
