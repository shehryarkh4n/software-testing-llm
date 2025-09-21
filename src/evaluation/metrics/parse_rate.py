import javalang
import re
from tqdm import tqdm

def is_trivially_invalid(code):
    # Heuristics to skip obviously broken code
    code_stripped = code.strip()
    return (
        len(code_stripped) < 10 or        # too short
        code_stripped.lower() in {"type", "null", "undefined"} or  # known junk
        code.count('"') % 2 != 0 or      # unmatched string quotes
        code.count('{') != code.count('}') or  # unbalanced braces
        code.count('(') != code.count(')')     # unbalanced parentheses
    )

def compute_parse_rate(predictions, refs=None):
    """
    Try to parse each prediction with javalang.
    If prediction doesn't declare a class, wrap in a dummy class.
    """
    
    total = len(predictions)
    success = 0
    filtered_idx = []
    for idx, code in enumerate(tqdm(predictions, desc="Parsing", unit="sample")):
        if is_trivially_invalid(code):
            continue  # skip early
        # If there's a class declaration, don't wrap; else, wrap in dummy class
        code_to_parse = code if re.search(r"\bclass\b", code) else f"public class Dummy {{ {code} }}"
        try:
            javalang.parse.parse(code_to_parse)
            success += 1
            filtered_idx.append(idx)
        except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError, IndexError, TypeError):
            pass  # Parsing failed
    score = success / total if total > 0 else 0.0
    return score, filtered_idx
