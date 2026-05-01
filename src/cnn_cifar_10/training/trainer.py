from __future__ import annotations
import torch
from torch import nn


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.CrossEntropyLoss,
) -> float:
    """Train the model for one epoch.
    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The dataloader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        device (torch.device): The device to run the training on.
        criterion (nn.Module): The loss function to use for training.
    Returns:
        float: The average loss for the epoch.
    """

    model.train()  # Set the model to training mode
    total_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)  # Move images to the specified device
        labels = labels.to(device)  # Same

        optimizer.zero_grad()

        logits = model(images)  # Forward pass
        loss = criterion(logits, labels)  # Calculate the loss

        loss.backward()  # backward pass
        optimizer.step()  # Update the weights of each parameter

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    return avg_loss
