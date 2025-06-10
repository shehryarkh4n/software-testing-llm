from codebleu import calc_codebleu  

def compute_codebleu(preds, refs, lang='java', **kwargs):
    result = calc_codebleu(preds, refs, lang=lang)
    # result is a dict; usually return the main score
    return result