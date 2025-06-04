from functools import partial
import os
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    SFTConfig, 
    SFTTrainer,
    BitsAndBytesConfig
)
from transformers.utils import is_bfloat16_supported
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import torch.distributed as dist

from bin.log_setup import setup_logging
from src.data.transforms import SPECIAL_PROCESSING_FUNCS


def train_main(config_path: str):
    """Main entry point for training/fine-tuning with config control."""
    # 1. Load config
    cfg = OmegaConf.load(config_path)
    logger = setup_logging(cfg.misc.log_dir, cfg.misc.log_name)
    logger.info("Config loaded:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("--------BEGINNING TRAINING PIPELINE :D--------\n\n")

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

    if getattr(cfg.bnb, "is_bnb", False): 
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
            device_map="auto",
            trust_remote_code=True
        )
    else:
        logger.info("Skipping BnB Config..")
        model = model_class.from_pretrained(
            model_name_or_path,
            device_map="auto",
        )
        
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if hasattr(cfg.model, "tokenizer_pad_token") and cfg.model.tokenizer_pad_token:
        if cfg.model.tokenizer_pad_token == "eos_token":
            tokenizer.pad_token = tokenizer.eos_token
        # else: Don't set it right now.
        
    # Set padding side if specified in config
    if hasattr(cfg.model, "tokenizer_padding_side") and cfg.model.tokenizer_padding_side:
        tokenizer.padding_side = cfg.model.tokenizer_padding_side
        
    # model.to(cfg.misc.device) DO I NEED THIS? LAST TIME IT MESSED WITH MULTI-GPU USAGE

    if getattr(cfg.peft, "is_peft", False):
        logger.info("Adding PEFT Config...")
        model = prepare_model_for_kbit_training(model) # should be universal for any quantization
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
        elif cfg.peft.type == "qlora":
            # QLoRA setup (bitsandbytes + LoRA config)
            pass
        elif cfg.peft.type == "bnb":
            # BitsAndBytes quantization config
            pass
        else:
            raise ValueError(f"Unknown PEFT type: {cfg.peft.type}")
    else:
        logger.info("Skipping PEFT Config...")

    # 3. Data loading

    # 4. Optional special processing
    if getattr(cfg.dataset, "need_special_processing", False):
        ds = load_dataset("json", data_files={"train": cfg.dataset.path})
        train_dataset = ds["train"]

        logger.info("Dataset needs special processing...")
        
        func_name = cfg.dataset.special_processing_func
        if func_name not in SPECIAL_PROCESSING_FUNCS:
            raise ValueError(f"Special processing function '{func_name}' not found in registry.")
        
        logger.info(f"Applying special processing function: {func_name}")
        train_dataset = SPECIAL_PROCESSING_FUNCS[func_name](tokenizer, train_dataset)
        
        logger.info(f"Saving processed dataset for the model: {cfg.model.name}")
        output_path = os.path.join(cfg.dataset.processed_path, f"{cfg.model.name}_processed.json")
        train_dataset.to_json(output_path)
        logger.info(f"Processed dataset saved at: {output_path}")
    else:
        logger.info("Dataset DOES NOT need special processing; Using the stored processed version...")
        if cfg.dataset.processed_path:
            ds = load_dataset("json", data_files={"train": cfg.dataset.processed_path})
            train_dataset = ds["train"]
        else:
            raise ValueError("Special Processing False & No Processed Path Provided")
    

    if getattr(cfg.dataset, "need_tokenization", False):
        logger.info("Applying Tokenization to the dataset...")
        
        tokenize_func = partial(SPECIAL_PROCESSING_FUNCS[cfg.dataset.tokenizer_func], tokenizer) # ??
        train_dataset = train_dataset.map(tokenize_func, batched=True, remove_columns=train_dataset.column_names)
        output_path = os.path.join(cfg.dataset.tokenized_path, f"{cfg.model.name}_tokenized.json")
        train_dataset.to_json(output_path)
        logger.info(f"Tokenized dataset saved at: {output_path}")
    else:
        logger.info("Skipping tokenization...")

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
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
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
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),   
        )
    else:
        raise ValueError("Unknown TrainingArguments type")

    # 7. Trainer setup
    if cfg.training.trainer_type == "Trainer":
        trainer = Trainer(
            model=model,
            args=training_args,
            dataset_text_field=cfg.training.dataset_text_field,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )
    elif cfg.training.trainer_type == "SFTTrainer":
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            dataset_text_field=cfg.training.dataset_text_field,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError("Unknown Trainer type")

    # 8. Set seeds for reproducibility
    torch.manual_seed(cfg.training.seed)

    # 9. Train and save
    logger.info("%%%% Starting training... %%%%")
    trainer.train()
    trainer.save_model(cfg.training.output_dir)
    logger.info("Training complete. Model saved to %s", cfg.training.output_dir)
    
    logger.info("Killing processes & freeing up memory...")
    del trainer, model, tokenizer
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()        # Release unreferenced memory to CUDA driver
    torch.cuda.ipc_collect()        # Clean up IPC resources, especially in multi-process setups

    logger.info("--------SHUTTING DOWN TRAINING PIPELINE :D--------\n\n")
    
