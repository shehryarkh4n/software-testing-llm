from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    SFTConfig, 
    SFTTrainer
)
from datasets import load_dataset
import torch
import logging

from src.data.transforms import SPECIAL_PROCESSING_FUNCS

logger = logging.getLogger(__name__)

def train_main(config_path: str):
    """Main entry point for training/fine-tuning with config control."""
    # 1. Load config
    cfg = OmegaConf.load(config_path)
    logger.info("Config loaded:\n%s", OmegaConf.to_yaml(cfg))

    assert hasattr(cfg, "model"), "Config missing model section"
    assert hasattr(cfg, "training"), "Config missing training section"

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
        raise ValueError("Unknown model architecture: {}".format(cfg.model.architecture))

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path)
    model.to(cfg.misc.device)

    # 3. Data loading
    ds = load_dataset("json", data_files={"train": cfg.dataset.path})
    train_dataset = ds["train"]

    # 4. Optional special processing
    if getattr(cfg.dataset, "need_special_processing", False):
        logger.info("Dataset needs special processing...")
        func_name = cfg.dataset.special_processing_func
        if func_name not in SPECIAL_PROCESSING_FUNCS:
            raise ValueError(f"Special processing function '{func_name}' not found in registry.")
        logger.info(f"Applying special processing function: {func_name}")
        train_dataset = SPECIAL_PROCESSING_FUNCS[func_name](train_dataset, cfg)

    # 5. Tokenization
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], 
            max_length=cfg.dataset.max_input_len, 
            truncation=True, 
            padding="max_length"
        )
    train_dataset = train_dataset.map(tokenize_fn, batched=True)

    # 6. Training arguments
    if cfg.training.training_args_type == "TrainingArguments":
        training_args = TrainingArguments(
            output_dir=cfg.training.output_dir,
            num_train_epochs=cfg.training.num_train_epochs,
            per_device_train_batch_size=cfg.dataset.batch_size,
            learning_rate=cfg.training.learning_rate,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            evaluation_strategy=cfg.training.eval_strategy,
            save_strategy=cfg.training.save_strategy,
            save_total_limit=cfg.training.save_total_limit,
            logging_steps=cfg.training.logging_steps,
            report_to=cfg.training.report_to,
            seed=cfg.training.seed,
            fp16=True if cfg.misc.device == "cuda" else False,
        )
    elif cfg.training.training_args_type == "SFTConfig":
        training_args = SFTConfig(
            output_dir=cfg.training.output_dir,
            num_train_epochs=cfg.training.num_train_epochs,
            per_device_train_batch_size=cfg.dataset.batch_size,
            learning_rate=cfg.training.learning_rate,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            evaluation_strategy=cfg.training.eval_strategy,
            save_strategy=cfg.training.save_strategy,
            save_total_limit=cfg.training.save_total_limit,
            logging_steps=cfg.training.logging_steps,
            report_to=cfg.training.report_to,
            seed=cfg.training.seed,
            fp16=True if cfg.misc.device == "cuda" else False,
        )
    else:
        raise ValueError("Unknown TrainingArguments type")

    # 7. Trainer setup
    if cfg.training.trainer_type == "Trainer":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )
    elif cfg.training.trainer_type == "SFTTrainer":
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError("Unknown Trainer type")

    # 8. Set seeds for reproducibility
    torch.manual_seed(cfg.training.seed)

    # 9. Train and save
    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(cfg.training.output_dir)
    logger.info("Training complete. Model saved to %s", cfg.training.output_dir)
