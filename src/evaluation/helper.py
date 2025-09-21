import os
import re

ASSIST_PATTERNS = [
    r"<\|start_header_id\|>assistant<\|end_header_id\|>",   # Llama‑3
    r"<\|im_start\|>assistant",                             # Qwen
]

def extract_reference_from_prompt(prompt: str):
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
    matches = re.findall(pattern, prompt, flags=re.DOTALL)
    if matches:
        for ref in reversed(matches):
            clean = ref.strip()
            if clean:
                return clean
    return ""

def get_prompt_for_inference(prompt: str) -> str:
    """
    Works for both 
    Llama‑3 (`<|start_header_id|>…`)
    Qwen (`<|im_start|>assistant`) 
    chat templates.
    """
    # find all assistant headers (union of patterns)
    matches = []
    for pat in ASSIST_PATTERNS:
        matches.extend(re.finditer(pat, prompt))

    if not matches:                         # no assistant header at all
        return prompt.strip()

    # keep everything up to and including the *last* match
    last = max(matches, key=lambda m: m.end())
    return prompt[: last.end()].strip()

def extract_java_code(text: str) -> str:
    # quick check to save a regex run
    if not text.lstrip().startswith("```java"):
        return text

    # Strip the first line
    after_open = text.lstrip().split("\n", 1)
    if len(after_open) == 1:
        # There was no newline after the opening fence (unlikely but safe-guard :D)
        return text
    body = after_open[1]

    # … then look for the first closing fence
    closing_pos = body.find("```")
    if closing_pos != -1:
        body = body[:closing_pos]

    return body.rstrip()     # drop trailing whitespace from the extracted code