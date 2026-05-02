from __future__ import annotations
import torch
from torch import nn


def evaluate(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> float:
    """Evaluate the model on the test set.
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader for the test data.
        device (torch.device): The device to run the evaluation on.
    Returns:
        float: The accuracy of the model on the test set.
    """

    model.eval()
    correct = 0
    total = len(dataloader.dataset)
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            pred = torch.argmax(logits, dim=1)

            correct += (pred == labels).sum().item()  # For int type and not tensor type

    accuracy = correct / total

    return accuracy
