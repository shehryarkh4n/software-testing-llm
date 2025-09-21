from .bleu import compute_bleu
from .parse_rate import compute_parse_rate
from .edit_distance import compute_edit_distance
from .exact_match import compute_exact_match
from .codebleu import compute_codebleu
from .compile_rate import compute_compile_rate
from .coverage import compute_coverage

METRIC_REGISTRY = {
    "parse_rate": compute_parse_rate,
    "compile_rate": compute_compile_rate,
    "coverage": compute_coverage,
    "bleu": compute_bleu,
    "edit_distance": compute_edit_distance,
    "exact_match": compute_exact_match,
    "codebleu": compute_codebleu,
}
