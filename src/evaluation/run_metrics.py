import sys, argparse
from datasets import load_dataset
import re
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from metrics import METRIC_REGISTRY

def extract_reference_from_prompt(prompt: str):
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
    matches = re.findall(pattern, prompt, flags=re.DOTALL)
    if matches:
        for ref in reversed(matches):
            clean = ref.strip()
            if clean:
                return clean
    return ""

def run_metrics(orig, labels):

    # ### REMOVE THIS ####
    # orig = "src/data/datasets/methods2test/processed/llama_3_2_3B/processed_test.json"
    # labels = "src/evaluation/results/methods2test/llama_3_2_1B/predictions_1750583407.json"
    # ### REMOVE THIS ####

    ds = load_dataset("json", data_files={"test": orig})["test"]
    references = [extract_reference_from_prompt(ex["prompt"]) for ex in ds]

    with open(labels, 'r') as f:
        predictions = json.load(f)

    for name, metric_func in METRIC_REGISTRY.items():
        try:
            score = metric_func(predictions, references)
            print(f"{name}: {score}")
        except Exception as e:
            print(f"{name}: ERROR ({e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset", "-od", type=str, default=None)
    parser.add_argument("--predictions", "-p", type=str, default=None)
    args = parser.parse_args()

    run_metrics(args.original_dataset, args.predictions)