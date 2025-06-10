import sys
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from metrics import METRIC_REGISTRY

def extract_reference_from_prompt(prompt: str):
    # Find all assistant turns, return the last one
    parts = prompt.split("<|start_header_id|>assistant<|end_header_id|>")
    # Last part after that marker is the reference (strip eot if present)
    if len(parts) > 1:
        ref = parts[-1]
        # Remove <|eot_id|> and whitespace
        ref = ref.replace("<|eot_id|>", "").strip()
        return ref
    return ""


def infer(config_path, num_samples):
    cfg = OmegaConf.load(config_path)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load dataset and select first num_samples examples
    ds = load_dataset("json", data_files={"test": cfg.dataset.processed_path})["test"]
    examples = ds.select(range(min(num_samples, len(ds))))
    inputs = [ex["prompt"] for ex in examples]
    references = [extract_reference_from_prompt(ex["prompt"]) for ex in examples]
    
    # Run inference
    batch_size = getattr(cfg.dataset, "batch_size", 8)
    predictions = []
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        enc = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=cfg.dataset.max_input_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model.generate(**enc, max_new_tokens=128)
        predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    
    # Display input, reference, prediction for each example
    for idx, (inp, ref, pred) in enumerate(zip(inputs, references, predictions)):
        print(f"\n--- Example {idx+1} ---")
        print("Input:")
        print(inp)
        print("Reference:")
        print(ref)
        print("Prediction:")
        print(pred)
    
    # Run metrics and display
    print("\n--- Metric Results ---")
    for name, metric_func in METRIC_REGISTRY.items():
        try:
            score = metric_func(predictions, references)
            print(f"{name}: {score:.4f}")
        except Exception as e:
            print(f"{name}: ERROR ({e})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mini_eval.py <config_path.yaml> <num_examples>")
        sys.exit(1)
    mini_eval(sys.argv[1], int(sys.argv[2]))
