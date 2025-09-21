import logging
import os
import time
from pathlib import Path


def _is_main_process() -> bool:
    """Return True on the main (rank 0) process.
    Falls back to True when no distributed context is found.
    """
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    if rank is not None:
        try:
            return int(rank) == 0
        except ValueError:
            return True
    # Torch fallback (safe if torch.distributed is available)
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def setup_logging(output_dir: str, log_name: str = "train.log") -> logging.Logger:
    """Create a simple file-only logger.

    - Ensures directory exists
    - Unique timestamped filename: <stem>_YYYYmmdd_HHMMSS.log
    - Handlers cleared to avoid duplicates
    - File handler attached only on rank 0; others are silent
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = Path(log_name).stem
    file_path = Path(output_dir) / f"{stem}_{ts}.log"

    logger = logging.getLogger("software_testing_llm")
    logger.propagate = False
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if called multiple times
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if _is_main_process():
        # File handler
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.addHandler(logging.NullHandler())

    return logger