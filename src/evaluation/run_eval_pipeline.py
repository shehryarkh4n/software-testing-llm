import sys, argparse
from datasets import load_dataset
import re
import os
import json

# Move to src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.evaluation.metrics import METRIC_REGISTRY

def extract_reference_from_llama(prompt: str):
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
    matches = re.findall(pattern, prompt, flags=re.DOTALL)
    if matches:
        for ref in reversed(matches):
            clean = ref.strip()
            if clean:
                return clean
    return ""

def extract_reference_from_qwen(prompt: str) -> str:
    pattern = r"<\|im_start\|>assistant\s*(.*?)\s*<\|im_end\|>"
    matches = re.findall(pattern, prompt, flags=re.DOTALL)
    for chunk in reversed(matches):           # walk from last to first
        clean = chunk.strip()
        if clean:
            return clean
    return ""


def run_metrics(orig, labels, model, num_examples=None):
    
    # input methods + features
    ds = load_dataset("json", data_files={"test": orig})["test"]
    if num_examples is not None: ds = ds.select(range(min(num_examples, len(ds))))
    references = []

    if model == 'qwen':
        references = [extract_reference_from_qwen(ex["prompt"]) for ex in ds]
    elif model == 'llama':
        references = [extract_reference_from_llama(ex["prompt"]) for ex in ds]
    else:
        raise ValueError("No model or incorrect model choice.")

    # predictions
    with open(labels, 'r') as f: predictions = json.load(f)
    if num_examples is not None: predictions = predictions[:num_examples]

    # setup for eval loop
    current_predictions = predictions
    current_references = references

    # Start with all original indices
    original_indices = list(range(len(current_predictions)))
    
    for name, metric_func in METRIC_REGISTRY.items():
        try:
            # Compute score and filtered indices
            score, filtered_idx = metric_func(current_predictions, current_references)
            print(f"-----\n{name}: {score} | Successful: {len(filtered_idx)}/{len(current_predictions)} | % of Total: {len(filtered_idx)/len(predictions)} \n-----")
    
            # Map filtered indices to original indices
            updated_predictions = [current_predictions[i] for i in filtered_idx]
            updated_references = [current_references[i] for i in filtered_idx]
            updated_indices = [original_indices[i] for i in filtered_idx]
    
            if name in ['parse_rate', 'compile_rate']:
                # Save failed outputs with original indices
                failed_idx = [i for i in range(len(current_predictions)) if i not in filtered_idx]
                # failed_outputs = [
                #     {
                #         "original_idx": original_indices[i],
                #         "prediction": current_predictions[i]
                #     }
                #     for i in failed_idx
                # ]
                # with open(f"failed_{name.split('_')[0]}.json", 'w') as f:
                #     json.dump(failed_outputs, f, indent=2)
    
                # Save successful predictions with original indices if final step
                if name == 'parse_rate' and updated_predictions:
                    success_outputs = [
                        {
                            "original_idx": idx,
                            "prediction": pred
                        }
                        for idx, pred in zip(updated_indices, updated_predictions)
                    ]
                    with open(f"success_{name.split('_')[0]}.json", 'w') as f:
                        json.dump(success_outputs, f, indent=2)
                    return
                # Update for next round
                current_predictions = updated_predictions
                current_references = updated_references
                original_indices = updated_indices
    
        except Exception as e:
            print(f"{name}: ERROR ({e})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset", "-od", type=str, default=None)
    parser.add_argument("--predictions", "-p", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--num_examples", "-n", type=int, default=None)
    args = parser.parse_args()

    run_metrics(args.original_dataset, args.predictions, args.model, args.num_examples)