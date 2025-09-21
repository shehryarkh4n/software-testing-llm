# Exploring LLM Finetuning for Software Testing 

## Motivation

Writing tests is critical yet often overlooked in real-world software development. Automating this with language models is promising, but progress is limited by two key issues:

1. **Data scarcity**: available test-suite datasets (e.g., *methods2test*, *Defects4J*) are few, narrow in scope, and mostly Java-only.
2. **Evaluation uncertainty**: it remains unclear which models are “good enough” for test generation: small vs. large, general-purpose vs. code-pretrained, or base vs. finetuned.

This repository addresses both challenges. It provides a **reproducible pipeline** to train and evaluate instruction-tuned models on software testing datasets, with consistent processing and code-aware metrics. Our experiments show that even modestly sized models, once finetuned, can achieve comparable parse and CodeBLEU scores to larger models, demonstrating that cost-effective solutions are possible. More importantly, the pipeline enables others to easily train, infer, and benchmark their own models in a fair and extensible way.


## Contents

This repository provides a clean, reproducible pipeline to **fine-tune code-capable LLMs** to generate **minimal, compilable JUnit tests**, and to **evaluate** them with code-aware metrics.

1. **Train / fine-tune** with TRL’s `SFTTrainer` (optionally LoRA + bitsandbytes).
2. **Infer** on processed prompts with batched generation.
3. **Evaluate** predictions with BLEU/CodeBLEU, exact match, parse/compile rate, coverage, edit distance, etc.
4. **Fairness filtering** ensures models are compared on the **exact same example indices** across splits.
---
* [Quick Start](#quick-start)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Configuration (YAML)](#configuration-yaml)
* [Data & Processing](#data--processing)
* [Training](#training)
* [Inference](#inference)
* [Metrics & Evaluation](#metrics--evaluation)
* [Extending the System](#extending-the-system)
* [Multi-GPU / FSDP](#multi-gpu--fsdp)
* [Troubleshooting](#troubleshooting)
* [License & Citation](#license--citation)

---

## Quick Start

```bash
# 0) (Optional) Create a fresh env
python -m venv .venv && source .venv/bin/activate

# 1) Install deps (PyTorch install varies by CUDA; see notes below)
pip install -r requirements.txt  # or install the core packages listed below

# 2) (Optional) Configure accelerate for multi-GPU
accelerate config

# 3) Train (single- or multi-GPU)
accelerate launch --config_file configs/accelerate/qwen.json \
  bin/run.py train -c configs/train.yaml

# 4) Inference (optionally show metrics and/or save predictions)
python bin/run.py infer -c configs/infer.yaml --show_metrics --save_predictions

# 5) Or run metrics explicitly
python bin/run.py metrics \
  -o src/data/datasets/methods2test/processed/Qwen2_5-Coder-3B-Instruct/processed_test.json \
  -p src/evaluation/results/methods2test/Qwen2.5-Coder-3B-Instruct/predictions_FINETUNED_<timestamp>.json
```

---

## Repository Structure

```
bin/
  run.py                # Typer CLI: train / infer / metrics
  log_setup.py          # rank-aware logging helpers

configs/
  train.yaml            # sample training config (see below)
  infer.yaml            # sample inference config
  common/registered_models.yaml  # lists processed splits for fairness intersections
  accelerate/qwen.json  # FSDP config for accelerate (multi-GPU)

src/
  training/
    trainer.py          # end-to-end training orchestration
    modeling.py         # HF model/tokenizer loader + LoRA/BnB hooks
    args.py             # SFTConfig builder (TRL)
  data/
    pipeline.py         # split-aware processing, fairness filtering, tokenization
    transforms.py       # prompt building & tokenization funcs (registered)
    fairness.py         # split_intersections via registered_models.yaml
    datasets/...        # raw/processed/tokenized JSON files
  evaluation/
    infer.py            # batched generation + optional inline metrics
    helper.py           # e.g., extract_java_code()
    metrics/            # metric implementations & registry
      helpers/...
      bleu.py, codebleu.py, exact_match.py, ...
      compile_rate.py, parse_rate.py, coverage.py, edit_distance.py
    results/...         # saved predictions & scores

logs/
outputs/                # saved checkpoints (fine-tuned weights)
```

---

## Installation

The core stack is:

* **Python** ≥ 3.10
* **PyTorch** (install per your CUDA setup)
* `transformers`, `trl`, `accelerate`, `datasets`, `peft`, `omegaconf`, `typer`, `tqdm`
* Optional (depending on config): `bitsandbytes`, `codebleu` (or your CodeBLEU impl), `tensorboard`

Example (CUDA 12.x users—adjust as needed):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers trl accelerate datasets peft omegaconf typer tqdm bitsandbytes tensorboard
```

> If you don’t plan to use 4-bit quantization, you can skip `bitsandbytes`.

---

## Configuration (YAML)

All behavior is configured via YAMLs under `configs/`. Two main ones:

### `train.yaml` (excerpt)

```yaml
model:
  is_local: false                 # load from HF hub or local path
  model_dir: "./"                 # used if is_local: true
  hf_name: "Qwen/Qwen2.5-Coder-3B-Instruct"
  name: "Qwen2.5-Coder-3B-Instruct"
  architecture: "causal"          # or "seq2seq"
  tokenizer_pad_token: "eos_token"
  tokenizer_padding_side: "right"

peft:
  is_peft: true
  type: "lora"
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  bias: "none"
  target_modules: ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
  task_type: "CAUSAL_LM"

bnb:
  is_bnb: false                   # set true to load 4-bit
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: "bfloat16"

dataset:
  name: "methods2test"
  train_path: "src/data/datasets/methods2test/original/files/train.json"
  eval_path:  "src/data/datasets/methods2test/original/files/eval.json"
  test_path:  "src/data/datasets/methods2test/original/files/test.json"
  need_special_processing: false
  need_tokenization: false
  special_processing_funcs:
    train: "dataset_preprocessing_with_target"
    eval:  "dataset_preprocessing_with_target"
    test:  "dataset_preprocessing_prompt_only"
  tokenizer_funcs:
    train: "dataset_tokenization"
    eval:  "dataset_tokenization"
    test:  "dataset_tokenization_infer"
  max_input_len: 3584
  processed_*_location: ".../processed_*.json"
  tokenized_*_location: ".../tokenized_*.json"

training:
  output_dir: "src/outputs/Qwen2_5-Coder-3B-Instruct/"
  dataset_text_field: "prompt"
  num_train_epochs: 10
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  eval_strategy: "epoch"
  save_strategy: "epoch"
  save_total_limit: 2
  logging_steps: 100
  optim: "adamw_8bit"
  bf16: true
  dataloader_num_workers: 32
  ddp_find_unused_parameters: false
  remove_unused_columns: false

misc:
  accelerate_config: "configs/accelerate/qwen.json"
  registered_models_yaml: "configs/common/registered_models.yaml"
  log_dir: "logs/default/Qwen2_5-Coder-3B-Instruct"
  log_name: "Qwen2_5-Coder-3B-Instruct.log"
```

### `infer.yaml` (excerpt)

```yaml
model:
  is_local: false
  model_dir: "src/outputs/Qwen2_5-Coder-3B-Instruct/checkpoint-29151"
  name: "Qwen/Qwen2.5-Coder-3B-Instruct"
  architecture: "causal"
  tokenizer_pad_token: "eos_token"
  tokenizer_padding_side: "left"

infer:
  save_path: "src/evaluation/results/methods2test/Qwen2.5-Coder-3B-Instruct"
  pred_label: "FINETUNED"   # used to label saved files

dataset:
  processed_path: "src/data/datasets/methods2test/processed/Qwen2_5-Coder-3B-Instruct/processed_test.json"
  batch_size: 384
  max_input_len: 720

misc:
  log_dir: "logs/default"
  log_name: "QWEN.log"
```

### `registered_models.yaml`

Lists **processed split files** for multiple models. During dataset build, the pipeline computes **split intersections** so every model is trained/evaluated on the **same example indices**.

```yaml
models:
  llama_3B:
    processed_train: ".../llama_3_2_3B/processed_train.json"
    processed_eval:  ".../llama_3_2_3B/processed_eval.json"
    processed_test:  ".../llama_3_2_3B/processed_test.json"
  qwen_coder:
    processed_train: ".../Qwen2_5-Coder-3B-Instruct/processed_train.json"
    processed_eval:  ".../Qwen2_5-Coder-3B-Instruct/processed_eval.json"
    processed_test:  ".../Qwen2_5-Coder-3B-Instruct/processed_test.json"
```

---

## Data & Processing

### Raw → Processed → Tokenized

* **Raw JSON splits** (under `src/data/datasets/<dataset>/original/files/`) are loaded by `pipeline.py`.
* **Processed splits** are produced by **special processing functions** in `transforms.py` (registered via `@register_func`):

  * `dataset_preprocessing_with_target` (train/eval) builds a **chat-style prompt** and keeps an `idx` for fairness.
  * `dataset_preprocessing_prompt_only` (test) builds prompts **without target** so generation starts from scratch.
* **Tokenization** is performed by `dataset_tokenization` (train/eval) or `dataset_tokenization_infer` (test).

> **Test split for inference**: The file referenced by `infer.dataset.processed_path` should contain **`prompt`** for generation and **`reference`** (gold test) for evaluation metrics. If your current processed test file does not include `reference`, add it (or supply it to the `metrics` CLI separately).

### Prompt Template (high level)

The default prompt encourages one **minimal, compilable** JUnit test:

* One `@Test` method inside `public class GeneratedTest`.
* Minimal happy-path setup.
* Use `org.junit.Assert.*`.
* Output **only** Java test code (no commentary).

---

## Training

The training CLI is in `bin/run.py` (Typer):

```bash
# Single GPU
python bin/run.py train -c configs/train.yaml

# Multi-GPU with accelerate + FSDP
accelerate launch --config_file configs/accelerate/qwen.json \
  bin/run.py train -c configs/train.yaml
```

Under the hood (`trainer.py`):

1. **Model & tokenizer** are built (`modeling.py`):

   * Loads from HF Hub or local path (`is_local`).
   * Optional **bitsandbytes 4-bit** quantization.
   * Optional **LoRA** via PEFT.
2. **Datasets** are built (`pipeline.py`):

   * Loads raw/processed/tokenized JSON.
   * Applies **fairness filtering** using `registered_models.yaml` (shared indices).
3. **SFT args** are built (`args.py`) → **`SFTTrainer`** (TRL).
4. **Training** runs, checkpoints are saved to `training.output_dir`.

---

## Inference

```bash
python bin/run.py infer -c configs/infer.yaml \
  --num_examples 200 \
  --save_predictions \
  --show_metrics
```

Key behavior (`evaluation/infer.py`):

* Loads the **fine-tuned** (or base) model.
* Uses **left-padding** for batched generation.
* Builds batches (`dataset.batch_size`) and calls `model.generate(...)`.
* Extracts Java with `extract_java_code(...)`.
* Optionally prints metrics immediately (`--show_metrics`).
* Saves predictions to `infer.save_path` with a timestamp if `--save_predictions` is set.

Saved file pattern:

```
src/evaluation/results/<dataset>/<model>/predictions_<LABEL>_<timestamp>.json
```

---

## Metrics & Evaluation

### Run metrics inline (during inference)

Use `--show_metrics` with `infer`. It loads the **metric registry** and prints scores.

### Run metrics as a separate step

```bash
python bin/run.py metrics \
  -o path/to/processed_test.json \
  -p path/to/predictions.json
```

**Implemented metrics** (see `src/evaluation/metrics/`):

* `exact_match` — string equality.
* `bleu`, `codebleu` — n-gram/code structure similarity.
* `edit_distance` — Levenshtein character-level distance.
* `parse_rate` — % outputs that parse as Java.
* `compile_rate` — % outputs that compile (requires JDK if enabled).
* `coverage` — heuristics over references vs predictions (token/AST coverage; see implementation).

> The metrics operate over `references` (gold tests) vs `predictions` (generated tests). Make sure your **test split** provides references, or use a separate ground-truth file when calling the `metrics` CLI.

---

## Extending the System

### Add a new special processing or tokenization function

1. Implement a function in `src/data/transforms.py`.
2. Decorate with `@register_func("<unique_name>")`.
3. Reference it in your YAML (`dataset.special_processing_funcs` or `dataset.tokenizer_funcs`).

### Add a new metric

1. Create `src/evaluation/metrics/<my_metric>.py` exposing a callable `my_metric(preds, refs) -> float`.
2. Register it in the `METRIC_REGISTRY` (see `run_metrics.py` / `metrics/__init__.py` pattern).
3. It will now appear in `--show_metrics` and the `metrics` CLI.

### Swap model / PEFT / quantization

* Change `model.hf_name` (or `model_dir` with `is_local: true`).
* Toggle LoRA (`peft.is_peft`) and target modules.
* Toggle bitsandbytes 4-bit (`bnb.is_bnb: true`) and set quant params.

---

## Multi-GPU / FSDP

Accelerate config example (`configs/accelerate/qwen.json`):

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16
num_processes: 8
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_use_orig_params: true
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer
downcast_bf16: "no"
```

Launch:

```bash
accelerate launch --config_file configs/accelerate/qwen.json \
  bin/run.py train -c configs/train.yaml
```

---

## Troubleshooting

* **CUDA OOM during training**
  Lower `per_device_train_batch_size`, increase `gradient_accumulation_steps`, enable `bnb.is_bnb: true`, or reduce `max_input_len`.

* **Tokenizer pad issues**
  Training uses **right** padding by default; inference uses **left** padding. Both set `pad_token = eos_token` if missing.

* **Mismatched tokenized lengths**
  `pipeline.py` will **re-tokenize** if counts don’t match expected shared indices. Check `need_tokenization: false/true` and your cached JSONs.

* **No metrics printed**
  Use `--show_metrics` for inline evaluation, or run the `metrics` subcommand with both `-o` (original/processed test with references) and `-p` (predictions JSON).

* **Compile/parse metrics**
  If your compile metric is enabled, ensure a JDK is available on `PATH`. For parse rate, verify that `extract_java_code(...)` returns a clean snippet.