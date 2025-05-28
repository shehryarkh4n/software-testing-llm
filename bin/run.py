#!/usr/bin/env python3
import typer
from pathlib import Path
import logging
from src.training.trainer import train_main
from src.evaluation.evaluator import evaluate_main

app = typer.Typer(add_completion=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", help="Path to training config YAML.")
):
    """Train or fine-tune a model."""
    logger.info(f"Starting training using config: {config}")
    train_main(config_path=config)


@app.command()
def evaluate(
    config: Path = typer.Option(..., "--config", "-c", help="Path to evaluation config YAML.")
):
    """Evaluate a model on a dataset."""
    logger.info(f"Starting evaluation using config: {config}")
    evaluate_main(config_path=config)

@app.command()
def test(
    config: Path = typer.Option(..., "--config", "-c", help="Path to testing config YAML.")
):
    """Test a model on a dataset."""
    logger.info(f"Starting testing using config: {config}")
    test_main(config_path=config)

if __name__ == "__main__":
    app()
