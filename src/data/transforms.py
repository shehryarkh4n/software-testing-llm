from __future__ import annotations
from typing import Iterable, Optional
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm

SPECIAL_PROCESSING_FUNCS = {}

SYSTEM_INSTRUCTION = (
    "You are an expert coder, providing exactly the code requested by the user."
    "Output only the Java test code; no explanations, comments, or extra text."
)

USER_INSTRUCTION = (
    "Write a *single* JUnit-style unit test for **the focal method(s) shown below**."
    "Follow these rules:"
    "1. Exactly **one** public test method annotated with `@Test`."
    "2. Give it a **descriptive name** that reflects the behavior under test."
    "3. Include only **minimal setup** for the happy path."
    "4. Use `org.junit.Assert` (e.g., `assertEquals`, `assertTrue`, `assertThrows`)."
    "5. Make the code **compilable**: include `import` statements and wrap in a public class named `GeneratedTest`."
    "6. No extra helpers, comments, or unrelated code."
)


def register_func(name):
    def decorator(fn):
        SPECIAL_PROCESSING_FUNCS[name] = fn
        return fn
    return decorator


# ----------------------------
# Helpers
# ----------------------------

def _instr_len(tokenizer: PreTrainedTokenizerBase) -> int:
    return len(tokenizer(USER_INSTRUCTION, add_special_tokens=False).input_ids)


def _build_prompt(tokenizer: PreTrainedTokenizerBase, source_code: str, target: Optional[str], include_target: bool) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": USER_INSTRUCTION + source_code},
    ]
    if include_target and target is not None:
        messages.append({"role": "assistant", "content": target})
    # When include_target is False, we omit assistant so the generation starts from scratch.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

def _process_dataset(
    tokenizer: PreTrainedTokenizerBase,
    ds,
    *,
    include_target: bool,
    max_ctx: int = 4096,
    reserved_new: int = 512,
    src_field: str = "src_fm_fc_ms_ff",
    tgt_field: str = "target",
):
    """
    Build prompts for an HF Dataset and return a new Dataset with two columns:

      • prompt – chat‑formatted string (system + user + optional assistant)
      • idx    – original row index kept for fairness intersection later

    A sample is kept if the *prompt portion* (system + user + src) fits in
    `max_ctx - reserved_new` tokens with this tokenizer. Target length is not
    checked here (output tokens are generated or supervised separately).
    """
    def _instr_len(tok):
        return len(tok(USER_INSTRUCTION, add_special_tokens=False).input_ids)

    instr_len = _instr_len(tokenizer)
    max_prompt_tokens = max_ctx - reserved_new

    prompts: list[str] = []
    kept_indices: list[int] = []
    skipped = 0

    sources = ds[src_field]
    targets = ds[tgt_field] if tgt_field in ds.column_names else [None] * len(ds)

    for i, (src, tgt) in enumerate(tqdm(zip(sources, targets), total=len(ds))):
        src_len = len(tokenizer(src, add_special_tokens=False).input_ids)
        if instr_len + src_len > max_prompt_tokens:
            skipped += 1
            continue

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user",   "content": USER_INSTRUCTION + src},
        ]
        if include_target and tgt is not None:
            messages.append({"role": "assistant", "content": tgt})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
        kept_indices.append(i)

    print(f"Prompt‑build finished – kept {len(kept_indices)}, skipped {skipped}")
    return Dataset.from_dict({"prompt": prompts, "idx": kept_indices})



# ----------------------------
# Public registered functions
# ----------------------------

@register_func("dataset_preprocessing_with_target")
def dataset_preprocessing_with_target(
    tokenizer: PreTrainedTokenizerBase,
    ds,
    add_system_prompt: bool = False,  # kept for backward compat; unused
    MAX_CTX: int = 4096,
    RESERVED_NEW: int = 512,
):
    return _process_dataset(
        tokenizer,
        ds,
        include_target=True,
        max_ctx=MAX_CTX,
        reserved_new=RESERVED_NEW,
    )


@register_func("dataset_preprocessing_prompt_only")
def dataset_preprocessing_prompt_only(
    tokenizer: PreTrainedTokenizerBase,
    ds,
    add_system_prompt: bool = False,
    MAX_CTX: int = 4096,
    RESERVED_NEW: int = 512,
):
    return _process_dataset(
        tokenizer,
        ds,
        include_target=False,
        max_ctx=MAX_CTX,
        reserved_new=RESERVED_NEW,
    )


# ------- Tokenization -------

@register_func("dataset_tokenization")
def dataset_tokenization(tokenizer: PreTrainedTokenizerBase, example, max_input_len: int):
    tokens = tokenizer(example["prompt"], padding="max_length", truncation=True, max_length=max_input_len)
    # Supervised fine-tuning: use full-sequence labels (your original choice). Pad tokens ignored.
    tokens["labels"] = [
        -100 if token == tokenizer.pad_token_id else token for token in tokens["input_ids"]
    ]
    return tokens


@register_func("dataset_tokenization_infer")
def dataset_tokenization_infer(tokenizer: PreTrainedTokenizerBase, example, max_input_len: int):
    """Tokenization for inference-only prompts. No labels are produced."""
    tokens = tokenizer(example["prompt"], padding="max_length", truncation=True, max_length=max_input_len)
    # Do not include labels; generation code will call model.generate(...)
    return {k: tokens[k] for k in ("input_ids", "attention_mask")}