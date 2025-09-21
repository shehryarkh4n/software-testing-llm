#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional

import typer
from omegaconf import OmegaConf
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# Local imports
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from bin.log_setup import setup_logging
from src.data.pipeline import build_datasets


app = typer.Typer(add_completion=False)


def _exists(path: str | Path) -> bool:
    try:
        return Path(path).exists()
    except Exception:
        return False


def _safe_load_len_json(path: str | Path) -> Optional[int]:
    if not _exists(path):
        return None
    try:
        ds = load_dataset("json", data_files={"x": str(path)})["x"]
        return len(ds)
    except Exception:
        return None


def _count_lengths_tokenized(ds: Dataset, has_labels: bool) -> Dict[str, Any]:
    """
    Compute simple stats for tokenized datasets.
    For padded inputs, effective length = sum(attention_mask).
    For labels, count tokens != -100.
    """
    if len(ds) == 0:
        return {"n": 0}

    def per_example(batch):
        res = {}
        am = batch["attention_mask"]
        res["input_eff_len"] = [int(sum(mask)) for mask in am]
        if has_labels and "labels" in batch:
            labels = batch["labels"]
            res["label_len"] = [int(sum(1 for t in seq if t != -100)) for seq in labels]
        return res

    mapped = ds.map(per_example, batched=True, remove_columns=[], desc="computing length stats")
    n = len(mapped)
    in_lens = mapped["input_eff_len"]
    in_min = int(min(in_lens))
    in_max = int(max(in_lens))
    in_avg = float(sum(in_lens) / n)

    out: Dict[str, Any] = {
        "n": n,
        "input_eff_len": {"min": in_min, "max": in_max, "avg": in_avg},
    }

    if has_labels and "label_len" in mapped.column_names:
        ll = mapped["label_len"]
        out["label_len"] = {
            "min": int(min(ll)),
            "max": int(max(ll)),
            "avg": float(sum(ll) / n),
        }
    return out


def _preview_prompts(path: str | Path, n: int, logger):
    if not _exists(path):
        logger.warning("Processed file not found: %s", path)
        return
    ds = load_dataset("json", data_files={"x": str(path)})["x"]
    n = min(n, len(ds))
    logger.info("---- Prompt preview (%d rows) from %s ----", n, path)
    for i in range(n):
        p = ds[i].get("prompt", "")
        snippet = p[:400].replace("\n", "\\n")
        logger.info("[%d] %s%s", i, snippet, "..." if len(p) > 400 else "")


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c", help="Path to training config YAML."),
    preview: int = typer.Option(2, "--preview", help="Preview N prompts from processed splits."),
):
    """
    Dry-run the data pipeline:
      - load tokenizer
      - run build_datasets (process + tokenize as configured)
      - report dataset sizes and token stats
      - optionally preview processed prompts
    It will also save processed/tokenized json files as per your YAML paths.
    """
    cfg = OmegaConf.load(str(config))

    # Logger
    logger = setup_logging(cfg.misc.log_dir, f"debug_pipeline_{Path(cfg.misc.log_name).stem}.log")

    logger.info("=== Debug Data Pipeline ===")
    logger.info("Config file: %s", config)
    logger.info("Model: %s (local=%s)", cfg.model.hf_name, cfg.model.is_local)

    # Tokenizer (only)
    model_name_or_path = cfg.model.model_dir if cfg.model.is_local else cfg.model.hf_name
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    if getattr(cfg.model, "tokenizer_pad_token", None) == "eos_token":
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    if getattr(cfg.model, "tokenizer_padding_side", None):
        tok.padding_side = cfg.model.tokenizer_padding_side

    # Show fairness setting if present
    fair_cfg = getattr(cfg, "fairness", None)
    if fair_cfg and getattr(fair_cfg, "enabled", False):
        logger.info("Fairness: ENABLED | mode=%s | context_window=%s | reserved_new=%s | tokenizers=%s",
                    getattr(fair_cfg, "mode", "truncate"),
                    getattr(fair_cfg, "context_window", None),
                    getattr(fair_cfg, "reserved_new", None),
                    getattr(fair_cfg, "tokenizers", None))
    else:
        logger.info("Fairness: disabled")

    # Report raw counts (if files exist)
    raw_train_n = _safe_load_len_json(cfg.dataset.train_path)
    raw_eval_n  = _safe_load_len_json(cfg.dataset.eval_path)
    raw_test_n  = _safe_load_len_json(cfg.dataset.test_path)
    logger.info("Raw counts -> train=%s eval=%s test=%s", raw_train_n, raw_eval_n, raw_test_n)

    # Run pipeline (this will log chosen functions and also save processed/tokenized files)
    train_ds, eval_ds, test_ds = build_datasets(cfg, tok, logger)

    # Report processed counts if files written
    proc_train_n = _safe_load_len_json(cfg.dataset.processed_train_location)
    proc_eval_n  = _safe_load_len_json(cfg.dataset.processed_eval_location)
    proc_test_n  = _safe_load_len_json(cfg.dataset.processed_test_location)
    logger.info("Processed counts -> train=%s eval=%s test=%s", proc_train_n, proc_eval_n, proc_test_n)

    # Report tokenized counts
    tok_train_n = _safe_load_len_json(cfg.dataset.tokenized_train_location)
    tok_eval_n  = _safe_load_len_json(cfg.dataset.tokenized_eval_location)
    tok_test_n  = _safe_load_len_json(cfg.dataset.tokenized_test_location)
    logger.info("Tokenized counts -> train=%s eval=%s test=%s", tok_train_n, tok_eval_n, tok_test_n)

    # Compute stats on the in-memory tokenized datasets returned
    train_stats = _count_lengths_tokenized(train_ds, has_labels=True)
    eval_stats  = _count_lengths_tokenized(eval_ds,  has_labels=True)
    test_stats  = _count_lengths_tokenized(test_ds,  has_labels=False)

    logger.info("Train stats: %s", train_stats)
    logger.info("Eval  stats: %s", eval_stats)
    logger.info("Test  stats: %s", test_stats)

    # Optional prompt previews from processed files
    if preview and preview > 0:
        _preview_prompts(cfg.dataset.processed_train_location, preview, logger)
        _preview_prompts(cfg.dataset.processed_eval_location, preview, logger)
        _preview_prompts(cfg.dataset.processed_test_location, preview, logger)

    logger.info("=== Debug pipeline complete ===")


if __name__ == "__main__":
    app()
