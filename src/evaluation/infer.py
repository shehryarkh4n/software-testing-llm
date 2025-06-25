import sys, argparse
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import trange
import os
import re
import json
import time

# Fix import path for src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from evaluation.metrics import METRIC_REGISTRY

def extract_reference_from_prompt(prompt: str):
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
    matches = re.findall(pattern, prompt, flags=re.DOTALL)
    if matches:
        for ref in reversed(matches):
            clean = ref.strip()
            if clean:
                return clean
    return ""


def get_prompt_for_inference(prompt: str):
    matches = list(re.finditer(r"<\|start_header_id\|>assistant<\|end_header_id\|>", prompt))
    if not matches:
        return prompt.strip()
    if len(matches) == 1:
        return prompt[:matches[0].end()].strip()
    return prompt[:matches[-2].end()].strip()

def extract_java_code(text: str) -> str:
    # quick check to save a regex run
    if not text.lstrip().startswith("```java"):
        return text

    # Strip the first line
    after_open = text.lstrip().split("\n", 1)
    if len(after_open) == 1:
        # There was no newline after the opening fence (unlikely but safe-guard :D)
        return text
    body = after_open[1]

    # … then look for the first closing fence
    closing_pos = body.find("```")
    if closing_pos != -1:
        body = body[:closing_pos]

    return body.rstrip()     # drop trailing whitespace from the extracted code

def infer(config_path, num_samples=None, save_predictions=False):
    if not config_path:
        print("No Config Path!")
        return
        
    cfg = OmegaConf.load(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    FT_MODEL = cfg.model.model_dir or cfg.model.name
    print("Using: ", FT_MODEL)
    
    model = AutoModelForCausalLM.from_pretrained(
        FT_MODEL,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(FT_MODEL)
    model.eval()
    
    if hasattr(cfg.model, "tokenizer_pad_token") and cfg.model.tokenizer_pad_token:
        if cfg.model.tokenizer_pad_token == "eos_token":
            tokenizer.pad_token = tokenizer.eos_token
    if hasattr(cfg.model, "tokenizer_padding_side") and cfg.model.tokenizer_padding_side:
        tokenizer.padding_side = cfg.model.tokenizer_padding_side

    tokenizer.padding_side="left"
    ds = load_dataset("json", data_files={"test": cfg.dataset.processed_path})["test"]
    if num_samples is None:
        num_samples = len(ds)
    examples = ds.select(range(min(num_samples, len(ds))))
    inputs = [get_prompt_for_inference(ex["prompt"]) for ex in examples]
    references = [extract_reference_from_prompt(ex["prompt"]) for ex in examples]

    batch_size = getattr(cfg.dataset, "batch_size", 8)
    predictions = []
    for i in trange(0, len(inputs), batch_size, desc="Generating"):
        batch_inputs = inputs[i:i+batch_size]
        enc = tokenizer(
            batch_inputs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=cfg.dataset.max_input_len
        ).to(device)

        # length of the *unpadded* prompt for every sample
        orig_len = enc.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=512,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        for seq in outputs:
            gen_only   = seq[orig_len:]             # strip entire input (pads + prompt)
            prediction = tokenizer.decode(gen_only, skip_special_tokens=True)
            predictions.append(extract_java_code(prediction))

    print("\n--- Metric Results ---")
    for name, metric_func in METRIC_REGISTRY.items():
        try:
            score = metric_func(predictions, references)
            print(f"{name}: {score}")
        except Exception as e:
            print(f"{name}: ERROR ({e})")

    if save_predictions:
        pred_label = cfg.infer.pred_label or "UNKNOWN"
        out_path = f"{cfg.infer.save_path}/predictions_{pred_label}_{int(time.time())}.json"
        with open(out_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"\nSaved predictions to {out_path}")

    # Free up memory
    del model, tokenizer
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--num_examples", "-n", type=int, default=None,
                        help="How many records to run (default: whole set)")
    parser.add_argument("--save_predictions",  action="store_true",
                        help="Write predictions_<timestamp>.json when done")
    args = parser.parse_args()

    infer(args.config, args.num_examples, args.save_predictions)