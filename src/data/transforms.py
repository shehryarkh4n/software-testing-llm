from datasets import Dataset

SPECIAL_PROCESSING_FUNCS = {}

def register_func(name):
    def decorator(fn):
        SPECIAL_PROCESSING_FUNCS[name] = fn
        return fn
    return decorator

@register_func("llama_3_2_1B_template")
def llama_3_2_1B_template(tokenizer, examples):
    user_instruction = (
        "You are an expert at writing Java test cases given a focal method. "
        "You will be given one now. ONLY provide the complete test case, no other text is necessary.\n"
    )
    prompts = []
    srcs = examples['src_fm_fc_ms']
    targets = examples['target']
    for src, target in zip(srcs, targets):
        messages = [
            {"role": "user", "content": user_instruction + src},
            {"role": "assistant", "content": target}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return Dataset.from_dict({"prompt": prompts})



@register_func("llama_3_2_1B_tokenize")
def llama_3_2_1B_tokenize(tokenizer, example, max_input_len):
    tokens = tokenizer(example['prompt'], padding="max_length", truncation=True, max_length=max_input_len)
    # Set padding token labels to -100 to ignore them in loss calculation
    tokens['labels'] = [
        -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
    ]
    return tokens
