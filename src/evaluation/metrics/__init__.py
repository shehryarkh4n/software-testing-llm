from .bleu import compute_bleu
from .parse_rate import compute_parse_rate
from .edit_distance import compute_edit_distance
from .exact_match import compute_exact_match
from .codebleu import compute_codebleu

METRIC_REGISTRY = {
    "bleu": compute_bleu,
    "parse_rate": compute_parse_rate,
    "edit_distance": compute_edit_distance,
    "exact_match": compute_exact_match,
    "codebleu": compute_codebleu,
}
