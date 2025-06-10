import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.training.trainer import train_main

if __name__ == "__main__":
    # Expects: torchrun bin/launch.py path/to/config.yaml
    config_path = sys.argv[1]
    train_main(config_path=config_path)
