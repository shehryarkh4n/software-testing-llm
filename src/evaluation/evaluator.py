import os
import time
import json
import torch

from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from datasets import load_dataset

from src.evaluation.metrics import METRIC_REGISTRY
from src.evaluation.postprocessing import POSTPROCESS_REGISTRY
from bin.log_setup import setup_logging

def evaluate_main(config_path):
    # 1. Load config and logger
    cfg = OmegaConf.load(config_path)
    logger = setup_logging(cfg.misc.log_dir, cfg.misc.log_name)
    logger.info("Config loaded:\n%s", OmegaConf.to_yaml(cfg))

    start_time = time.time()
    logger.info("--------BEGINNING EVALUATION PIPELINE--------")

    # 2. Model and tokenizer loading
    if cfg.model.is_local:
        model_name_or_path = cfg.model.model_dir
    else:
        model_name_or_path = cfg.model.name

    if cfg.model.architecture == "causal":
        model_class = AutoModelForCausalLM
    elif cfg.model.architecture == "seq2seq":
        model_class = AutoModelForSeq2SeqLM
    else:
        raise ValueError(f"Unknown model architecture: {cfg.model.architecture}")

    if getattr(cfg, "bnb", None) and getattr(cfg.bnb, "is_bnb", False):
        logger.info("Adding BnB Config..")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=cfg.bnb.bnb_4bit_compute_dtype,
        )
        model = model_class.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto"
        )
    else:
        logger.info("Skipping BnB Config..")
        model = model_class.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if hasattr(cfg.model, "tokenizer_pad_token") and cfg.model.tokenizer_pad_token:
        if cfg.model.tokenizer_pad_token == "eos_token":
            tokenizer.pad_token = tokenizer.eos_token
    if hasattr(cfg.model, "tokenizer_padding_side") and cfg.model.tokenizer_padding_side:
        tokenizer.padding_side = cfg.model.tokenizer_padding_side

    if getattr(cfg, "peft", None) and getattr(cfg.peft, "is_peft", False):
        from peft import LoraConfig, get_peft_model
        logger.info("Adding PEFT Config...")
        if cfg.peft.type == "lora":
            peft_cfg = LoraConfig(
                r=cfg.peft.r,
                lora_alpha=cfg.peft.lora_alpha,
                lora_dropout=cfg.peft.lora_dropout,
                bias=cfg.peft.bias,
                target_modules=cfg.peft.target_modules,
                task_type=cfg.peft.task_type,
            )
            model = get_peft_model(model, peft_cfg)
            logger.info("LoRA applied :D")
        # QLoRA/bnb: implement as needed
    else:
        logger.info("Skipping PEFT Config...")
    model.eval()

    # 3. Load test dataset
    test_file = cfg.dataset.processed_path if os.path.exists(cfg.dataset.processed_path) else cfg.dataset.tokenized_test_location
    ds = load_dataset("json", data_files={"test": test_file})
    test_dataset = ds["test"]

    # 4. Prepare inputs & refs
    inputs = [ex["src_fm_fc_ms"] for ex in test_dataset]
    references = [ex["target"] for ex in test_dataset]

    # 5. Generate outputs (with batching)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_size = getattr(cfg.dataset, "batch_size", 8)
    predictions = []
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i + batch_size]
        enc = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=cfg.dataset.max_input_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model.generate(**enc, max_new_tokens=cfg.get("generation", {}).get("max_new_tokens", 128))
        predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    # 6. Post-processing
    for post_proc in cfg.get("postprocessing", []):
        func = POSTPROCESS_REGISTRY[post_proc["name"]]
        predictions = func(predictions, options=post_proc.get("options", {}))

    # 7. Metrics (add CodeBLEU and others)
    metric_results = {}
    for metric in cfg.metrics:
        metric_func = METRIC_REGISTRY[metric["name"]]
        options = metric.get("options", {})
        try:
            metric_results[metric["name"]] = metric_func(predictions, references, **options)
        except Exception as e:
            logger.error(f"Metric {metric['name']} failed: {e}")
            metric_results[metric["name"]] = None

    # 8. Save results to src/results/
    os.makedirs("src/results", exist_ok=True)
    result_fname = f"{cfg.model.name.replace('/', '_')}_{int(time.time())}.json"
    result_path = os.path.join("src/results", result_fname)
    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metric_results,
        "time_sec": round(time.time() - start_time, 2)
    }
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation Results:", json.dumps(metric_results, indent=2))
    logger.info(f"Evaluation complete. Results saved to {result_path}")

    # Free up memory
    del model, tokenizer
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()

    logger.info("--------SHUTTING DOWN EVALUATION PIPELINE--------")

if __name__ == "__main__":
    import sys
    evaluate_main(sys.argv[1])
