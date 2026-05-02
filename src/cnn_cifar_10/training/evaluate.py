from __future__ import annotations

import torch
from torch import nn


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Evaluate the model on the test set.
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader for the test data.
        device (torch.device): The device to run the evaluation on.
    Returns:
        tuple[float, float]: The accuracy and loss of the model on the test set.
    """

    model.eval()
    correct = 0
    total_loss = 0.0
    total = len(dataloader.dataset)
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)  # Calculate the loss per batch
            total_loss += loss.item()  # For int type and not tensor type

            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()  # For int type and not tensor type

    accuracy = correct / total  # total is total of datasets complete
    average_loss = total_loss / len(dataloader)  # Average loss per batch

    return (
        average_loss,
        accuracy,
    )
