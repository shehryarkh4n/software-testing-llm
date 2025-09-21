from __future__ import annotations
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def _select_model_class(architecture: str):
    if architecture == "causal":
        return AutoModelForCausalLM
    if architecture == "seq2seq":
        return AutoModelForSeq2SeqLM
    raise ValueError(f"Unknown model architecture: {architecture}")


def build_model_and_tokenizer(cfg, logger: logging.Logger):
    """Create model + tokenizer with optional bitsandbytes and PEFT.
    Returns (model, tokenizer).
    """
    logger.info("=" * 25)
    logger.info("STAGE 1: BUILDING MODEL & TOKENIZER")
    logger.info("=" * 25 + "\n")

    # --- Model source ---
    logger.info("[1.1] Determining model source …")
    if cfg.model.is_local:
        model_name_or_path = cfg.model.model_dir
        logger.info("Model will be loaded from local path: %s", model_name_or_path)
    else:
        model_name_or_path = cfg.model.hf_name
        logger.info("Model will be loaded from Hugging Face hub: %s", model_name_or_path)

    model_cls = _select_model_class(cfg.model.architecture)
    logger.info("Model architecture selected: %s", cfg.model.architecture)
    
    # --- Model load (BnB optional) ---
    logger.info("[1.2] Loading model weights …")
    if getattr(cfg.bnb, "is_bnb", False):
        logger.info("Using bitsandbytes 4-bit quantization …")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=getattr(torch, str(cfg.bnb.bnb_4bit_compute_dtype))
            if isinstance(cfg.bnb.bnb_4bit_compute_dtype, str)
            else cfg.bnb.bnb_4bit_compute_dtype,
        )
        model = model_cls.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        logger.info("Model loaded with bitsandbytes config.")
    else:
        logger.info("Loading model with torch_dtype=bfloat16, device_map=None …")
        model = model_cls.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        logger.info("Model loaded in bfloat16.")

    # --- Tokenizer ---
    logger.info("[1.3] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    logger.info("Tokenizer loaded.")

    if getattr(cfg.model, "tokenizer_pad_token", None):
        if cfg.model.tokenizer_pad_token == "eos_token":
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Tokenizer pad token set to eos_token.")

    if getattr(cfg.model, "tokenizer_padding_side", None):
        tokenizer.padding_side = cfg.model.tokenizer_padding_side
        logger.info("Tokenizer padding side set to: %s", tokenizer.padding_side)

    # --- PEFT ---
    logger.info("[1.4] Applying PEFT (Parameter-Efficient Fine-Tuning) if selected …")
    if getattr(cfg.peft, "is_peft", False):
        logger.info("PEFT enabled. Type: %s", cfg.peft.type)
        # model = prepare_model_for_kbit_training(model)
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
            logger.info("LoRA applied.")
        elif cfg.peft.type in {"qlora", "bnb"}:
            logger.warning("PEFT type %s declared but not implemented; proceeding without.", cfg.peft.type)
        else:
            raise ValueError(f"Unknown PEFT type: {cfg.peft.type}")
    else:
        logger.info("PEFT disabled.")
    
    logger.info("[✓] STAGE 1 complete: Model and tokenizer ready.\n\n")
    return model, tokenizer
