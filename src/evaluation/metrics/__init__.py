from .bleu import compute_bleu
from .parse_rate import compute_parse_rate
from .custom_metric import compute_custom_metric

METRIC_REGISTRY = {
    "bleu": compute_bleu,
    "parse_rate": compute_parse_rate,
    "custom_metric": compute_custom_metric,
}
