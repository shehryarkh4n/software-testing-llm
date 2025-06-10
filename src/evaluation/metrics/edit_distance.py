import Levenshtein

def compute_edit_distance(preds, refs, **kwargs):
    if not preds or not refs:
        return 0.0
    dists = [Levenshtein.distance(p, r) for p, r in zip(preds, refs)]
    return sum(dists) / len(dists)
