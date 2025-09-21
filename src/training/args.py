from __future__ import annotations
import logging
import torch
from trl import SFTConfig


def _is_bfloat16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def build_sft_config(cfg, logger) -> SFTConfig:
    # Ensure bf16 default matches hardware if user left it true
    logger.info("=" * 25)
    logger.info("STAGE 3: BUILDING SFT ARGUMENTS")
    logger.info("=" * 25 + "\n")
    
    bf16_flag = bool(getattr(cfg.training, "bf16", _is_bfloat16_supported()))

    # Summary log
    logger.info("[3.1] Training Configuration Summary:")
    logger.info("    Output Dir       : %s", cfg.training.output_dir)
    logger.info("    Epochs           : %d", cfg.training.num_train_epochs)
    logger.info("    Batch Size       : %d", cfg.training.per_device_train_batch_size)
    logger.info("    Learning Rate    : %.2e", cfg.training.learning_rate)
    logger.info("    Grad Accum Steps : %d", cfg.training.gradient_accumulation_steps)
    logger.info("    Save Strategy    : %s", cfg.training.save_strategy)
    logger.info("    Logging Steps    : %d", cfg.training.logging_steps)
    logger.info("    Report To        : %s", cfg.training.report_to)
    logger.info("    Seed             : %d", cfg.training.seed) 
    logger.info("    BF16 Enabled     : %s", bf16_flag) 

    configObj = SFTConfig(
        output_dir=cfg.training.output_dir,
        fsdp                = "full_shard",
        fsdp_config         = cfg.misc.accelerate_config,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        learning_rate=cfg.training.learning_rate,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        eval_strategy=cfg.training.eval_strategy,
        save_strategy=cfg.training.save_strategy,
        save_total_limit=cfg.training.save_total_limit,
        logging_steps=cfg.training.logging_steps,
        report_to=cfg.training.report_to,
        seed=cfg.training.seed,
        bf16=bf16_flag,
        dataset_text_field=cfg.training.dataset_text_field,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        ddp_find_unused_parameters=cfg.training.ddp_find_unused_parameters,
        dataloader_drop_last=getattr(cfg.training, "dataloader_drop_last", True),
        remove_unused_columns=getattr(cfg.training, "remove_unused_columns", False),
        lr_scheduler_type=getattr(cfg.training, "lr_scheduler_type", "linear"),
        optim=getattr(cfg.training, "optim", None),
        weight_decay=getattr(cfg.training, "weight_decay", 0.0),
        warmup_steps=getattr(cfg.training, "warmup_steps", 0),
        label_names=["labels"]
    )
    
    logger.info("[✓] STAGE 3 complete: Arguments built.\n\n")
    return configObj

# Debugging usage
def set_library_verbosity(logger: logging.Logger):
    """Align HF/Datasets logging with our logger level."""
    level = logger.getEffectiveLevel()
    try:
        import transformers
        transformers.utils.logging.set_verbosity(level)
    except Exception:
        pass
    try:
        import datasets
        datasets.utils.logging.set_verbosity(level)
    except Exception:
        pass