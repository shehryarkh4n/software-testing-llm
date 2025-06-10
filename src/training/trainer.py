from functools import partial
import os
import time
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    BitsAndBytesConfig
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset, DatasetDict
import torch
import torch.distributed as dist

from bin.log_setup import setup_logging
from src.data.transforms import SPECIAL_PROCESSING_FUNCS


def train_main(config_path: str):
    """Main entry point for training/fine-tuning with config control."""
    # 1. Load config
    start_time = time.time()
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
            trust_remote_code=True
        )
    else:
        logger.info("Skipping BnB Config..")
        model = model_class.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if hasattr(cfg.model, "tokenizer_pad_token") and cfg.model.tokenizer_pad_token:
        if cfg.model.tokenizer_pad_token == "eos_token":
            tokenizer.pad_token = tokenizer.eos_token
        # else: Don't set it right now.
        
    # Set padding side if specified in config
    if hasattr(cfg.model, "tokenizer_padding_side") and cfg.model.tokenizer_padding_side:
        tokenizer.padding_side = cfg.model.tokenizer_padding_side

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
    def get_split_dataset(dataset, split_ratio):
        # split_ratio: [train, eval, test]
        train_test = dataset.train_test_split(test_size=split_ratio[2])
        train = train_test['train']
        test = train_test['test']
        # Now split train into train and eval
        eval_size = split_ratio[1] / (split_ratio[0] + split_ratio[1])
        train_eval = train.train_test_split(test_size=eval_size)
        train = train_eval['train']
        eval = train_eval['test']
        return {'train': train, 'eval': eval, 'test': test}
    
    def save_splits(ds_dict, prefix_path):
        paths = {}
        for split, ds in ds_dict.items():
            path = f"{prefix_path}_{split}.json"
            ds.to_json(path)
            paths[split] = path
        return paths
    
    def load_splits(prefix_path):
        out = {}
        for split in ["train", "eval", "test"]:
            path = f"{prefix_path}_{split}.json"
            out[split] = load_dataset("json", data_files={split: path})[split]
        return out
        
    if not getattr(cfg.dataset, "need_tokenization", False):
        logger.info("Tokenization not required, loading tokenized splits...")
        train_dataset = load_dataset("json", data_files={"train": cfg.dataset.tokenized_train_location})["train"]
        eval_dataset  = load_dataset("json", data_files={"eval":  cfg.dataset.tokenized_eval_location})["eval"]
        test_dataset  = load_dataset("json", data_files={"test":  cfg.dataset.tokenized_test_location})["test"]
    
    elif not getattr(cfg.dataset, "need_special_processing", False):
        logger.info("Special processing not required, loading processed splits for tokenization...")
        train_dataset = load_dataset("json", data_files={"train": cfg.dataset.processed_train_location})["train"]
        eval_dataset  = load_dataset("json", data_files={"eval":  cfg.dataset.processed_eval_location})["eval"]
        test_dataset  = load_dataset("json", data_files={"test":  cfg.dataset.processed_test_location})["test"]
    
        # Now tokenize and save
        tokenize_func = partial(SPECIAL_PROCESSING_FUNCS[cfg.dataset.tokenizer_func], tokenizer, max_input_len=cfg.dataset.max_input_len)
        for split, ds in [("train", train_dataset), ("eval", eval_dataset), ("test", test_dataset)]:
            out = ds.map(tokenize_func, batched=True, remove_columns=ds.column_names, num_proc=cfg.training.dataloader_num_workers)
            out_path = getattr(cfg.dataset.processed_path, f"tokenized_{split}_location")
            out.to_json(out_path)
    
    else:
        logger.info("Loading raw data, special processing, then split and save...")
        # Load and process
        raw_ds = load_dataset("json", data_files={"data": cfg.dataset.path})["data"]
        func_name = cfg.dataset.special_processing_func
        if func_name not in SPECIAL_PROCESSING_FUNCS:
            raise ValueError(f"Special processing function '{func_name}' not found in registry.")
        processed_ds = SPECIAL_PROCESSING_FUNCS[func_name](tokenizer, raw_ds)
        # Split
        splits = get_split_dataset(processed_ds, cfg.dataset.split)
        # Save processed splits
        processed_paths = save_splits(splits, cfg.dataset.processed_path)
        # Tokenize and save tokenized splits
        tokenize_func = partial(SPECIAL_PROCESSING_FUNCS[cfg.dataset.tokenizer_func], tokenizer, max_input_len=cfg.dataset.max_input_len)
        for split, ds in splits.items():
            out = ds.map(tokenize_func, batched=True, remove_columns=ds.column_names, num_proc=cfg.training.dataloader_num_workers)
            out_path = getattr(cfg.dataset, f"tokenized_{split}_location")
            out.to_json(out_path)

    # 6. Training arguments
    if cfg.training.training_args_type == "TrainingArguments":
        training_args = TrainingArguments(
            output_dir=cfg.training.output_dir,
            num_train_epochs=cfg.training.num_train_epochs,
            per_device_train_batch_size=cfg.dataset.batch_size,
            learning_rate=cfg.training.learning_rate,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            eval_strategy=cfg.training.eval_strategy,
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
            eval_strategy=cfg.training.eval_strategy,
            save_strategy=cfg.training.save_strategy,
            save_total_limit=cfg.training.save_total_limit,
            logging_steps=cfg.training.logging_steps,
            report_to=cfg.training.report_to,
            seed=cfg.training.seed,
            fp16=cfg.training.fp16,
            bf16=cfg.training.bf16,   
            dataset_text_field=cfg.training.dataset_text_field,
            dataloader_num_workers=cfg.training.dataloader_num_workers,
            ddp_find_unused_parameters=cfg.training.ddp_find_unused_parameters,
            dataloader_drop_last=getattr(cfg.training, 'dataloader_drop_last', True),
            remove_unused_columns=getattr(cfg.training, 'remove_unused_columns', False),
        )
    else:
        raise ValueError("Unknown TrainingArguments type")

    # 7. Trainer setup
    if cfg.training.trainer_type == "Trainer":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
    elif cfg.training.trainer_type == "SFTTrainer":
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
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
    end_time = time.time()
    logger.info("--------SHUTTING DOWN TRAINING PIPELINE :D--------\n\n")
    logger.info(f"Training completed in {(end_time - start_time) / 3600:.2f} hours.")
    
