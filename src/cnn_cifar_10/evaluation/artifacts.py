import json
from pathlib import Path

import torch


def save_model_state(model: torch.nn.Module, save_path: Path) -> None:
    """Save the model state dictionary to a file.
    Args:
        model (torch.nn.Module): The model whose state dictionary is to be saved.
        save_path (Path): The path where the model state will be saved.
    """
    torch.save(model.state_dict(), save_path)


def save_metrics(metrics_dict: dict, save_path: Path) -> None:
    """Save the metrics dictionary to a JSON file.
    Args:
        metrics_dict (dict): A dictionary containing metric names and their corresponding values.
        save_path (Path): The path where the metrics will be saved.
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)
