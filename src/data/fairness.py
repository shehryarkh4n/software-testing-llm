from datasets import load_dataset
from functools import reduce
from typing import Dict, Set
import yaml, os

def _load_set(path):  # tiny helper
    return set(load_dataset("json", data_files={"x": path})["x"]["idx"])

def split_intersections(yaml_path: str) -> Dict[str, Set[int]]:
    """Return {"train": idx_set, "eval": idx_set, "test": idx_set} of shared indices."""
    cfg = yaml.safe_load(open(yaml_path))
    shared: Dict[str, Set[int]] = {"train": None, "eval": None, "test": None}
    for model_cfg in cfg["models"].values():
        for split in shared:
            path = model_cfg[f"processed_{split}"]
            idx_set = _load_set(path)
            shared[split] = idx_set if shared[split] is None else shared[split] & idx_set
    # sanity: convert None → empty set (if yaml missing a split)
    for k, v in shared.items():
        shared[k] = v or set()
    return shared
