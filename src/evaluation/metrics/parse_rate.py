import javalang

def compute_parse_rate(preds, refs=None, **kwargs):
    count = 0
    for code in preds:
        try:
            javalang.parse.parse(code)
            count += 1
        except:
            pass
    return count / len(preds) if preds else 0.0
