from __future__ import annotations
import time
import logging
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from trl import SFTTrainer

from bin.log_setup import setup_logging
from src.training.modeling import build_model_and_tokenizer
from src.training.args import build_sft_config
from src.data.pipeline import build_datasets


def train_main(config_path: str):
    start_time = time.time()
    cfg = OmegaConf.load(config_path)

    # One logger; handlers only on rank 0
    logger = setup_logging(cfg.misc.log_dir, cfg.misc.log_name)

    logger.info("Config loaded. Beginning training pipeline …")

    try:
        # --- Stage 1: Build model & tokenizer ---
        model, tokenizer = build_model_and_tokenizer(cfg, logger)
    except Exception as e:
        logger.exception("[1] Failed to build model and tokenizer: %s", e)
        raise SystemExit(1)
    
    try:
        # --- Stage 2: Build datasets ---
        train_dataset, eval_dataset, _ = build_datasets(cfg, tokenizer, logger)
    except Exception as e:
        logger.exception("[2] Failed to build datasets: %s", e)
        raise SystemExit(2)
    
    try:
        # --- Stage 3: Build SFT args ---
        sft_args = build_sft_config(cfg, logger)
    except Exception as e:
        logger.exception("[3] Failed to configure SFT arguments: %s", e)
        raise SystemExit(3)
    
    try:
        # --- Stage 3.5: Initialize Trainer ---
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
    except Exception as e:
        logger.exception("[3.5] Failed to initialize trainer: %s", e)
        raise SystemExit(4)
    
    try:
        # --- Stage 3.6: Seed + Cache cleanup ---
        torch.manual_seed(cfg.training.seed)
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning("[3.6] Seed setting or cache cleanup failed (non-fatal): %s", e)
    
    try:
        # --- Stage 4: Train ---
        logger.info("=" * 25)
        logger.info("STAGE 4: TRAINING")
        logger.info("=" * 25 + "\n")
        trainer.train()
        trainer.save_model(cfg.training.output_dir)
        logger.info("[4.1] Training complete. Model saved to %s", cfg.training.output_dir)
    except Exception as e:
        logger.exception("❌ [4] Training failed: %s", e)
        raise SystemExit(5)


    # --- Cleanup ---
    logger.info("[4.2] Cleaning up distributed state and CUDA memory …")
    del trainer, model, tokenizer
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass

    elapsed_hours = (time.time() - start_time) / 3600.0
    logger.info("\n" + "=" * 25)
    logger.info(f"TRAIN PIPELINE FINISHED | TIME: {elapsed_hours} HOURS")
    logger.info("=" * 25 + "\n")