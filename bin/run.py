#!/usr/bin/env python3
import os
import typer
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.training.trainer import train_main
from src.evaluation.infer import infer  

app = typer.Typer(add_completion=False)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", help="Path to training config YAML.")
):
    """Train or fine-tune a model."""
    logger.info(f"Starting training using config: {config}")
    train_main(str(config))


@app.command()
def infer_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to inference config YAML."),
    num_examples: int = typer.Option(None, "--num-examples", "-n", help="Number of records to run. Default: all."),
    save_predictions: bool = typer.Option(False, "--save-predictions", help="Save predictions_<timestamp>.json"),
):
    """Run inference on a model."""
    logger.info(
        f"Running inference using config: {config} | num_examples: {num_examples} | save_predictions: {save_predictions}"
    )
    infer(str(config), num_examples, save_predictions)


if __name__ == "__main__":
    app()
