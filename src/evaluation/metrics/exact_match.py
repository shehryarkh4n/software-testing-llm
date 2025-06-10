def compute_exact_match(preds, refs, **kwargs):
    if not preds or not refs:
        return 0.0
    matches = [p.strip() == r.strip() for p, r in zip(preds, refs)]
    return sum(matches) / len(matches)
    