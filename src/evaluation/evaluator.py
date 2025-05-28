from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from datasets import load_dataset

from src.evaluation.metrics import METRIC_REGISTRY
from src.evaluation.postprocessing import POSTPROCESS_REGISTRY

def evaluate_main(config_path):
    cfg = OmegaConf.load(config_path)
    print("Config loaded:", OmegaConf.to_yaml(cfg))

    # 1. Load model & tokenizer
    if cfg.model.architecture == "causal":
        model_class = AutoModelForCausalLM
    else:
        model_class = AutoModelForSeq2SeqLM

    model_name_or_path = cfg.model.model_dir if getattr(cfg.model, "is_local", False) else cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path)
    model.eval()

    # 2. Load dataset
    ds = load_dataset("json", data_files={"test": cfg.dataset.path})
    test_dataset = ds["test"]

    # 3. Prepare inputs
    # (Assume text is under "input" and reference under "target")
    inputs = [ex["input"] for ex in test_dataset]
    references = [ex["target"] for ex in test_dataset]

    # 4. Generate outputs (batching for speed)
    predictions = []
    batch_size = cfg.dataset.batch_size
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        enc = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=cfg.dataset.max_input_len)
        with torch.no_grad():
            outputs = model.generate(**{k: v for k, v in enc.items()}, max_new_tokens=128)
        predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    # 5. Post-processing (as specified in config)
    for post_proc in cfg.get("postprocessing", []):
        func = POSTPROCESS_REGISTRY[post_proc["name"]]
        predictions = func(predictions, options=post_proc.get("options", {}))

    # 6. Metrics (plug-in via registry)
    metric_results = {}
    for metric in cfg.metrics:
        metric_func = METRIC_REGISTRY[metric["name"]]
        options = metric.get("options", {})
        metric_results[metric["name"]] = metric_func(predictions, references, **options)

    # 7. Print/save results
    print("Evaluation Results:")
    for k, v in metric_results.items():
        print(f"{k}: {v}")

    return metric_results
