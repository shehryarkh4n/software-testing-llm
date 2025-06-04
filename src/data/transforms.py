SPECIAL_PROCESSING_FUNCS = {}

def register_func(name):
    def decorator(fn):
        SPECIAL_PROCESSING_FUNCS[name] = fn
        return fn
    return decorator

@register_func("llama_3_2_1B_template")
def llama_3_2_1B_template(tokenizer, example):

    messages = [
        {"role": "user", "content": example['question']},
        {"role": "assistant", "content": example['answer']}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt}

@register_func("llama_3_2_1B_tokenize")
def llama_3_2_1B_tokenize(tokenizer, example):
    tokens = tokenizer(example['prompt'], padding="max_length", truncation=True, max_length=128)
    # Set padding token labels to -100 to ignore them in loss calculation
    tokens['labels'] = [
        -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
    ]
    return tokens
