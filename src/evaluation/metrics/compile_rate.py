from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from threading import Lock
from typing import List, Tuple, Dict, Iterable, Any, Optional

from tqdm import tqdm  # <-- progress bar

# ------------------------------ Shared fixers --------------------------------
from helpers.java_fixers import (
    sanitize_test,
    ensure_outer_class,
    ensure_std_imports,
    collect_imports,
    make_stubs,
    build_classpath,
)

# ------------------------------ Config ---------------------------------------

HOME = os.environ.get("HOME", "")
JUNIT_JARS = [
    f"{HOME}/software-testing-llm/src/test-artifacts/junit-4.13.2.jar",
    f"{HOME}/software-testing-llm/src/test-artifacts/hamcrest-core-1.3.jar",
]
JUNIT_CP = build_classpath(*JUNIT_JARS)

ERROR_LOG = Path(f"{HOME}/software-testing-llm/logs/compile/{time.time()}_compile_errors.jsonl")
MAX_WORKERS = 96

# Only allow these import/package prefixes (keeps generations deterministic)
ALLOWED_PREFIXES = ("java.", "javax.", "org.junit.", "org.hamcrest.")

# Error classification patterns
FATAL_PATTERNS = [
    r"illegal start of (?:type|expression)",
    r"unclosed string literal",
    r"reached end of file while parsing",
    r"';' expected",
    r"class, interface, or enum expected",
    r"not a statement",
    r"invalid method declaration",
]
SOFT_PATTERNS = [
    r"package .* does not exist",
    r"cannot find symbol",
    r"class file for .* not found",
]

_log_lock = Lock()

# ------------------------------ Data types -----------------------------------

@dataclass
class CompileResult:
    idx: int
    status: str          # HARD_PASS | SOFT_PASS | HARD_FAIL
    msg: str             # first stderr line (or "")
    fixed_src: str       # sanitized test source used for compile

# ------------------------------ Helpers --------------------------------------

def disallowed_imports(src: str) -> List[str]:
    """Return any import FQCNs outside the allowlist fence."""
    bad = []
    for _is_static, fqcn, _wild in collect_imports(src):
        if not fqcn.startswith(ALLOWED_PREFIXES):
            bad.append(fqcn)
    return bad

def is_fatal_error(stderr: str) -> bool:
    for line in stderr.splitlines():
        if any(re.search(p, line) for p in FATAL_PATTERNS):
            return True
    return False

def only_soft_errors(stderr: str) -> bool:
    lines = stderr.splitlines()
    return bool(lines) and all(any(re.search(p, l) for p in SOFT_PATTERNS) for l in lines)

def trivially_bad(src: str) -> bool:
    if not src or src.strip() == "":
        return True
    if src.count("{") != src.count("}"):
        return True
    if "@Test" not in src:
        return True
    return False

# ------------------------------ Core worker ----------------------------------

def compile_one(idx: int, raw_src: str) -> CompileResult:
    """
    Normalize, fence imports, stub unknowns, and attempt to compile this test.
    """
    if trivially_bad(raw_src):
        return CompileResult(idx, "HARD_FAIL", "trivial reject", raw_src or "")

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)

        # This order works best when making it compile-compatible
        fixed = ensure_outer_class(raw_src)   # wrap if no top-level class
        fixed = sanitize_test(fixed)          # drop top-level 'public' from class/interface/enum
        fixed = ensure_std_imports(fixed)     # JUnit/time/util imports if needed

        # Early fence: disallow random external libraries (Mockito, etc.)
        bad = disallowed_imports(fixed)
        if bad:
            msg = f"disallowed import(s): {', '.join(sorted(set(bad)))}"
            return CompileResult(idx, "HARD_FAIL", msg, fixed)

        # Persist test source (single-file compile)
        (tmp / "GenTest.java").write_text(fixed)

        # Create stubs for unknown classes and imported non-std FQCNs (within fence)
        make_stubs(tmp, fixed)

        # Compile
        java_files = [str(p.relative_to(tmp)) for p in tmp.glob("**/*.java")]
        cmd = ["javac", "-cp", JUNIT_CP] + java_files
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)

        if proc.returncode == 0:
            return CompileResult(idx, "HARD_PASS", "", fixed)

        # Classify error
        stderr = proc.stderr or ""
        status = "HARD_FAIL"
        if not is_fatal_error(stderr) and only_soft_errors(stderr):
            status = "SOFT_PASS"
        msg = stderr.splitlines()[0] if stderr else "unknown error"

        return CompileResult(idx, status, msg, fixed)

def _worker(args: Tuple[int, str]) -> CompileResult:
    idx, pred = args
    res = compile_one(idx, pred)
    if res.status == "HARD_FAIL":
        with _log_lock:
            ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
            with ERROR_LOG.open("a") as f:
                f.write(json.dumps({"idx": res.idx, "error": res.msg}) + "\n")
    return res

# ------------------------------ Public API -----------------------------------

def compute_compile_rate(predictions: List[str],
                         max_workers: int = MAX_WORKERS
) -> Tuple[float, List[int], List[int], List[int], List[CompileResult]]:
    """
    Returns:
        score     : (hard + soft passes) / total
        pass_ids  : idx compiled cleanly (HARD_PASS)
        soft_ids  : idx with only soft errors (SOFT_PASS) – allow onward
        fail_ids  : idx with fatal errors (HARD_FAIL)
        results   : list[CompileResult] for all inputs (no guaranteed order)
    """
    if ERROR_LOG.exists():
        ERROR_LOG.unlink()

    pass_ids: List[int] = []
    soft_ids: List[int] = []
    fail_ids: List[int] = []
    results: List[CompileResult] = []

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("fork")) as ex:
        futures = [ex.submit(_worker, (i, p)) for i, p in enumerate(predictions)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Compiling", unit="test"):
            res: CompileResult = fut.result()
            results.append(res)
            if res.status == "HARD_PASS":
                pass_ids.append(res.idx)
            elif res.status == "SOFT_PASS":
                soft_ids.append(res.idx)
            else:
                fail_ids.append(res.idx)

    score = (len(pass_ids) + len(soft_ids)) / len(predictions) if predictions else 0.0
    return score, pass_ids, soft_ids, fail_ids, results

def emit_corrected_tests(results: List[CompileResult],
                         jsonl_path: Optional[Path] = None,
                         java_dir: Optional[Path] = None) -> None:
    """
    Artifact C:
      JSONL lines: {idx, status, fixed_src, log}
      Optional: write GenTest{idx}.java to java_dir
    """
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w") as f:
            for r in sorted(results, key=lambda x: x.idx):
                f.write(json.dumps({
                    "idx": r.idx,
                    "status": r.status,
                    "fixed_src": r.fixed_src,
                    "log": r.msg,
                }) + "\n")
    if java_dir:
        java_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            (java_dir / f"GenTest{r.idx}.java").write_text(r.fixed_src)

# ------------------------------ I/O utils ------------------------------------

def _load_json_any(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open() as f:
        return json.load(f)

def load_predictions(path: Path) -> Tuple[List[str], List[int]]:
    """
    Accepts:
      - dict-of-lists: {"prediction": [...], "original_idx": [...]}
      - list-of-objs:  [{"prediction": str, "original_idx": int}, ...]
      - jsonl with same shapes
    Returns (predictions_in_order, idxs_in_order)
    """
    data = _load_json_any(path)

    # dict-of-lists
    if isinstance(data, dict) and "prediction" in data:
        preds = list(data["prediction"])
        if "original_idx" in data and isinstance(data["original_idx"], list):
            idxs = [int(x) for x in data["original_idx"]]
        else:
            idxs = list(range(len(preds)))
        return preds, idxs

    # list-of-objects
    if isinstance(data, list):
        preds = []
        idxs = []
        for i, row in enumerate(data):
            if isinstance(row, dict) and "prediction" in row:
                preds.append(row["prediction"])
                idxs.append(int(row.get("original_idx", i)))
        if preds:
            return preds, idxs

    # Fallback: treat as a single prediction string
    if isinstance(data, str):
        return [data], [0]

    raise ValueError(f"Unrecognized test JSON format at {path}")

# ------------------------------ CLI ------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compile gate for model-generated Java tests (with progress).")
    parser.add_argument("--tests_json", "-t", required=True,
                        help="Path to JSON/JSONL with 'prediction' and optional 'original_idx'.")
    parser.add_argument("--emit_fixed", "-o", default=None,
                        help="Path to write corrected tests JSONL (artifact C).")
    parser.add_argument("--emit_java_dir", "-j", default=None,
                        help="Directory to write corrected .java files (optional).")
    parser.add_argument("--max_workers", "-w", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    tests_path = Path(args.tests_json)
    predictions, idxs = load_predictions(tests_path)

    score, hard, soft, fail, results = compute_compile_rate(predictions, args.max_workers)

    # Simple summary to stdout
    summary = {
        "compile_rate": score,
        "hard": len(hard),
        "soft": len(soft),
        "fail": len(fail),
    }
    print(json.dumps(summary))

    # Emit artifacts
    jsonl_path = Path(args.emit_fixed) if args.emit_fixed else None
    java_dir = Path(args.emit_java_dir) if args.emit_java_dir else None
    emit_corrected_tests(results, jsonl_path, java_dir)

if __name__ == "__main__":
    main()
