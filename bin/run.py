#!/usr/bin/env python3
import sys
from pathlib import Path
import typer
import logging

# project root on path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.training.trainer import train_main
from src.evaluation.infer import infer_main
from src.evaluation.run_metrics import run_metrics
import os, torch

app = typer.Typer(add_completion=False)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", help="Path to training config YAML.")
):
    """Train or fine-tune a model (launched under `accelerate launch`)."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    train_main(str(config))

@app.command()
def infer(
    config: Path = typer.Option(..., "--config", "-c", help="Path to inference config YAML."),
    num_examples: int = typer.Option(None, "--num_examples", "-n", help="Number of records to run. Default: all."),
    save_predictions: bool = typer.Option(False, "--save_predictions", help="Save predictions_<timestamp>.json"),
    show_metrics: bool = typer.Option(False, "--show_metrics", help="Show metrics against the predictions")
):
    """Run inference on a model (you may also launch this under accelerate)."""
    infer_main(str(config), num_examples, save_predictions, show_metrics)

@app.command()
def metrics(
    original: Path = typer.Option(..., "--original", "-o", help="Path to the original dataset"),
    predictions: Path = typer.Option(..., "--predictions", "-p", help="Path to the predictions from a model. This should be a simple list of examples"),
):
    """Get metrics for a set of original/predictions."""
    logger.info("Running Metrics")
    run_metrics(original, predictions)

if __name__ == "__main__":
    app()
