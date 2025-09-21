import sys, argparse
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import trange
import os
import re
import json
import gc
import time

from bin.log_setup import setup_logging
from src.evaluation.helper import extract_java_code
# Fix import path for src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from evaluation.metrics import METRIC_REGISTRY

def infer_main(config_path, num_samples=None, save_predictions=False, show_metrics=False):
    
    cfg = OmegaConf.load(config_path)

    # One logger; handlers only on rank 0
    logger = setup_logging(cfg.misc.log_dir, cfg.misc.log_name)
    logger.info("=" * 25)
    logger.info("STAGE 1: LOADING CONFIG & MODEL")
    logger.info("=" * 25 + "\n")
    
    start_time = time.time()
    if not config_path:
        print("[1.1] No Config Path!")
        return
    cfg = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[1.1] Config loaded successfully")
    
    FT_MODEL = cfg.model.model_dir or cfg.model.name
    if save_predictions:
        pred_label = cfg.infer.pred_label or "UNKNOWN"
        out_path = f"{cfg.infer.save_path}/predictions_{pred_label}_{int(time.time())}.json"
        logger.info(f"[1.1] Inference Output will be saved to: {out_path}")
    
    logger.info(f"[1.2] Using: {FT_MODEL}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_dir or cfg.model.name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    logger.info("[1.2] Model loaded...")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_dir or cfg.model.name)
    MAX_CTX       = tokenizer.model_max_length       # 4096 for Llama-3
    MAX_NEW_TOK   = 512                              # keep this
    max_input_len = MAX_CTX - MAX_NEW_TOK
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model.eval()
    logger.info(f"[1.2] Model Details:\n - max_input_len: {max_input_len}\n - new_tokens: {MAX_NEW_TOK}\n - padding_side: left\n...")

    logger.info("=" * 25)
    logger.info("STAGE 2: LOADING & PROCESSING DATASETS")
    logger.info("=" * 25 + "\n")
    
    # Load correctly processed test data
    ds = load_dataset("json", data_files={"test": cfg.dataset.processed_path})["test"]
    if num_samples is None:
        num_samples = len(ds)
    examples = ds.select(range(min(num_samples, len(ds))))
    
    logger.info(f"[2.1] Examples selected: {num_samples}/{len(ds)}")
    
    # Now your data should have separate 'prompt' and 'reference' columns
    inference_prompts = [ex["prompt"] for ex in examples]
    references = [ex["reference"] for ex in examples]
    # Generate predictions
    batch_size = getattr(cfg.dataset, "batch_size", 4)
    predictions = []

    logger.info(f"[2.2]: Inference Prompts & References Separated")
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("=" * 25)
    logger.info("STAGE 3: Inference")
    logger.info("=" * 25 + "\n")
    
    logger.info(f"Starting inference with batch size {batch_size}")
    logger.info(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    for i in trange(0, len(inference_prompts), batch_size, desc="Generating"):
        if i > 0:  # Don't clear on first iteration (nothing to clear)
            torch.cuda.empty_cache()
            gc.collect()
        
        batch_inputs = inference_prompts[i:i+batch_size]
        
        enc = tokenizer(
            batch_inputs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_input_len
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                max_new_tokens=MAX_NEW_TOK,
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        for j, seq in enumerate(outputs):
            input_length = (enc.input_ids[j] != tokenizer.pad_token_id).sum().item()
            generated_tokens = seq[input_length:]
            prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            predictions.append(extract_java_code(prediction))
            
    if show_metrics:
        logger.info("\n--- Metric Results ---")
        for name, metric_func in METRIC_REGISTRY.items():
            try:
                score = metric_func(predictions, references)
                print(f"{name}: {score}")
            except Exception as e:
                print(f"{name}: ERROR ({e})")

    logger.info("=" * 25)
    logger.info("STAGE 4: SAVING & CLEANUP")
    logger.info("=" * 25 + "\n")
    
    if save_predictions:
        pred_label = cfg.infer.pred_label or "UNKNOWN"
        out_path = f"{cfg.infer.save_path}/predictions_{pred_label}_{int(time.time())}.json"
        with open(out_path, "w") as f:
            json.dump(predictions, f, indent=2)
        logger.info(f" [4.1] Saved predictions to {out_path}")

    # Free up memory
    logger.info(f" [4.2] Cleaning up memory...")
    del model, tokenizer
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()

    elapsed_hours = (time.time() - start_time) / 3600.0
    logger.info("\n" + "=" * 25)
    logger.info(f"INFERENCE PIPELINE FINISHED | TIME: {elapsed_hours} HOURS")
    logger.info("=" * 25 + "\n")