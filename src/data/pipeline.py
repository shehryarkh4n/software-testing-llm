from __future__ import annotations
import logging
from functools import partial
from pathlib import Path
from typing import Dict, Set

from datasets import load_dataset
from src.data.transforms import SPECIAL_PROCESSING_FUNCS
from src.data.fairness import split_intersections  # helper from earlier snippet

def _load_json(path):
    return load_dataset("json", data_files={"x": str(path)})["x"]

def _load_raw_splits(cfg):
    return {
        "train": _load_json(cfg.dataset.train_path),
        "eval":  _load_json(cfg.dataset.eval_path),
        "test":  _load_json(cfg.dataset.test_path),
    }


def _load_tokenized_splits(cfg):
    return {
        "train": _load_json(cfg.dataset.tokenized_train_location),
        "eval":  _load_json(cfg.dataset.tokenized_eval_location),
        "test":  _load_json(cfg.dataset.tokenized_test_location),
    }


def _load_processed_splits(cfg):
    return {
        "train": _load_json(cfg.dataset.processed_train_location),
        "eval":  _load_json(cfg.dataset.processed_eval_location),
        "test":  _load_json(cfg.dataset.processed_test_location),
    }


def _save_processed_splits(cfg, splits):
    for split, ds in splits.items():
        ds.to_json(getattr(cfg.dataset, f"processed_{split}_location"))


def _tokenize_and_save_split(cfg, tokenizer, ds, token_fn_key: str, logger, split: str):
    if token_fn_key not in SPECIAL_PROCESSING_FUNCS:
        raise ValueError(f"Tokenizer func '{token_fn_key}' not registered.")
    token_fn = partial(
        SPECIAL_PROCESSING_FUNCS[token_fn_key],
        tokenizer,
        max_input_len=cfg.dataset.max_input_len,
    )
    logger.info("Tokenizing %s with %s …", split, token_fn_key)
    tokenized = ds.map(
        token_fn,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=cfg.training.dataloader_num_workers,
    )
    # Drop idx column if present (training not needed)
    if "idx" in tokenized.column_names:
        tokenized = tokenized.remove_columns("idx")
    out_path = getattr(cfg.dataset, f"tokenized_{split}_location")
    tokenized.to_json(out_path)
    return tokenized


def _resolve_processing_funcs(cfg) -> Dict[str, str]:
    sp_map = getattr(cfg.dataset, "special_processing_funcs", None)
    if sp_map is None:
        base = getattr(cfg.dataset, "special_processing_func", "dataset_preprocessing_with_target")
        test_key = "dataset_preprocessing_prompt_only" if "dataset_preprocessing_prompt_only" in SPECIAL_PROCESSING_FUNCS else base
        sp_map = {"train": base, "eval": base, "test": test_key}
    return sp_map


def _resolve_tokenizer_funcs(cfg) -> Dict[str, str]:
    tok_map = getattr(cfg.dataset, "tokenizer_funcs", None)
    if tok_map is None:
        base = getattr(cfg.dataset, "tokenizer_func", "dataset_tokenization")
        test_key = "dataset_tokenization_infer" if "dataset_tokenization_infer" in SPECIAL_PROCESSING_FUNCS else base
        tok_map = {"train": base, "eval": base, "test": test_key}
    return tok_map

def build_datasets(cfg, tokenizer, logger: logging.Logger):
    """Return train, eval, test datasets after split‑aware processing & fairness filtering."""

    logger.info("=" * 25)
    logger.info("STAGE 2: BUILDING DATASETS")
    logger.info("=" * 25 + "\n")

    sp_map = _resolve_processing_funcs(cfg)
    tok_map = _resolve_tokenizer_funcs(cfg)

    logger.info("[2.1] Special processing funcs: %s", sp_map)
    logger.info("[2.1] Tokenizer funcs: %s", tok_map)

    # ---------------------- 2.1: Processing ----------------------
    if not getattr(cfg.dataset, "need_special_processing", False):
        logger.info("[2.1] No special processing. Loading processed splits …")
        processed = _load_processed_splits(cfg)
    else:
        logger.info("[2.1] Running special processing per split …")
        raw = _load_raw_splits(cfg)
        processed = {}
        for split, ds in raw.items():
            func_key = sp_map[split]
            if func_key not in SPECIAL_PROCESSING_FUNCS:
                raise ValueError(f"Special processing func '{func_key}' not registered.")
            logger.info("[2.1] Processing split='%s' using: %s", split, func_key)
            processed_ds = SPECIAL_PROCESSING_FUNCS[func_key](tokenizer, ds)
            logger.info("[2.1] → %s processed size: %d", split, len(processed_ds))
            processed[split] = processed_ds
        _save_processed_splits(cfg, processed)

    # ---------------------- 2.2: Fairness filtering ----------------------
    reg_yaml = getattr(cfg.misc, "registered_models_yaml", None)
    shared_idx_map: Dict[str, Set[int]] | None = None
    if reg_yaml and Path(reg_yaml).exists():
        logger.info("[2.2] Fairness filtering activated via: %s", reg_yaml)
        shared_idx_map = split_intersections(reg_yaml)
        logger.info("[2.2] Shared index sizes:")
        for split in ["train", "eval", "test"]:
            logger.info("    %s → %d examples", split, len(shared_idx_map[split]))

        for split in processed:
            before = len(processed[split])
            if shared_idx_map[split]:
                processed[split] = processed[split].filter(
                    lambda ex, idx_set=shared_idx_map[split]: ex["idx"] in idx_set
                )
                after = len(processed[split])
                logger.info("[2.2] Fairness filter on '%s' – kept %d / %d (%.2f%%)",
                            split, after, before, 100 * after / before if before else 0)
            else:
                logger.warning("[2.2] Shared idx set for split='%s' is empty! Keeping all %d.", split, before)

    else:
        logger.info("[2.2] Fairness disabled – no registered_models_yaml provided.")

    # ---------------------- 2.3: Tokenization ----------------------
    tokenized = {}
    for split, ds in processed.items():
        logger.info("[2.3] Handling tokenization for split='%s'", split)
        tok_path = Path(getattr(cfg.dataset, f"tokenized_{split}_location"))
        needs_tokenize = not tok_path.exists() or getattr(cfg.dataset, "need_tokenization", False)

        if tok_path.exists() and not needs_tokenize:
            existing_ds = _load_json(tok_path)
            expected_len = len(shared_idx_map[split]) if shared_idx_map else len(ds)
            
            if len(existing_ds) != expected_len:
                logger.warning("[2.3] Existing tokenized '%s' has %d rows but expected %d → Retokenizing.",
                               split, len(existing_ds), expected_len)
                needs_tokenize = True
            else:
                tokenized[split] = existing_ds
                logger.info("[2.3] Loaded existing tokenized '%s' (%d examples).", split, len(existing_ds))

        if needs_tokenize:
            logger.info("[2.3] Tokenizing split='%s' from scratch …", split)
            tokenized[split] = _tokenize_and_save_split(cfg, tokenizer, ds, tok_map[split], logger, split)
            logger.info("[2.3] → Tokenized size: %d", len(tokenized[split]))

    logger.info("\n[✓] STAGE 2 complete: Train/Eval/Test datasets ready.\n")
    return tokenized["train"], tokenized["eval"], tokenized["test"]

