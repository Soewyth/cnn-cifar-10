from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random

from torchvision import transforms

from cnn_cifar_10.config.paths import get_root_dir, get_paths
from cnn_cifar_10.config.read_yaml import read_config_yaml
from cnn_cifar_10.data.load_data import load_train_data, load_test_data
from cnn_cifar_10.data.preprocessing import get_preprocessing_transforms
from cnn_cifar_10.data.augmentation import get_augmentation_transforms
from cnn_cifar_10.evaluation.artifacts import save_model_state, save_metrics
from cnn_cifar_10.evaluation.plots import plot_curves
from cnn_cifar_10.io.run_id import get_run_id
from cnn_cifar_10.models.resnet import get_resnet18
from cnn_cifar_10.training.trainer import train_one_epoch
from cnn_cifar_10.training.evaluate import evaluate


def main():
    # Load config and get paths
    root_dir = get_root_dir()
    paths = get_paths(root_dir=root_dir)
    config = read_config_yaml(root_dir / "config.yaml")
    path_raw_data = paths["data_raw"]
    path_figure = paths["figures"]
    path_model = paths["models"]

    # config to compare models with different hyperparameters
    image_size = config["model"]["input_size"]
    pretrained = config["model"]["pretrained"]
    freeze_backbone = config["model"]["freeze_backbone"]

    # Get run ID
    run_id = get_run_id(tag="run3_finetune_inputsize_224_freezefalse_512features")

    # Get same seed
    seed = config["training"]["random_seed"]
    torch.manual_seed(seed=seed)  # for torch
    np.random.seed(seed)  # for numpy
    random.seed(seed)  # for random

    # Get Transforms
    preprocessing = get_preprocessing_transforms(image_size=image_size)
    augmentation = get_augmentation_transforms()
    combined = transforms.Compose([*augmentation.transforms, *preprocessing.transforms])

    # Load data
    train_data = load_train_data(data_dir=path_raw_data, transform=combined)
    test_data = load_test_data(data_dir=path_raw_data, transform=preprocessing)

    # DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get model
    model = get_resnet18(pretrained=pretrained, freeze_backbone=freeze_backbone)
    model.to(device)

    optimizer_name = config["training"]["optimizer"]
    if optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    criterion = nn.CrossEntropyLoss()

    # Prepare Datas for loop
    num_epochs = config["training"]["num_epochs"]
    average_train_loss = 0.0
    average_test_loss = 0.0
    average_test_accuracies = 0.0

    # Lists to store train losses and test accuracies for each epoch
    train_losses = []
    test_losses = []
    test_accuracies = []
    # Early stopping parameters
    delta = config["training"]["delta"]
    patience = config["training"]["patience"]
    no_improvement_epochs = 0
    best_accuracy = 0.0

    # Time the training process
    start_time = time()

    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
        )

        train_losses.append(train_loss)
        print(f"Epoch : {epoch+1} / {num_epochs} : Train Loss : {train_loss:.4f}")

        # Accuracy on test set
        test_loss, test_accuracy = evaluate(
            model=model, dataloader=test_dataloader, device=device, criterion=criterion
        )
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)
        print(f"Epoch : {epoch+1} / {num_epochs} : Test Accuracy : {test_accuracy:.4f}")

        # EArly stopping for convergence
        if test_accuracy > best_accuracy + delta:
            best_accuracy = test_accuracy
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(
                f"Early stopping at epoch {epoch+1} due to no improvement in test accuracy."
            )
            break

    # End time
    training_time_seconds = time() - start_time

    # Calculate the mean of the train losses and test accuracies over all epochs
    average_train_loss = np.mean(train_losses)
    average_test_accuracies = np.mean(test_accuracies)
    average_test_loss = np.mean(test_losses)

    # === Save model and the artifacts ===
    # Plots curves
    path_plot = path_figure / f"training_curves_{run_id}.png"
    plot_curves(
        train_loss=train_losses,
        test_loss=test_losses,
        test_accuracies=test_accuracies,
        save_path=path_plot,
    )

    # Save model state
    model_destination = path_model / f"model_{run_id}.pth"
    save_model_state(model=model, save_path=model_destination)

    # Metrics
    metrics = {
        "datetime": str(run_id),
        "config": {
            "config_path": str(root_dir / "config.yaml"),
            "training": config["training"],
            "paths": config["paths"],
            "model": config["model"],
        },
        "results": {
            "training_time_seconds": round(training_time_seconds, 2),
            "best_accuracy": best_accuracy,
            "epoch_runs": len(train_losses),
            "average_train_loss": float(average_train_loss),
            "average_test_loss": float(average_test_loss),
            "average_test_accuracies": float(average_test_accuracies),
        },
        "per_epoch": {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "test_accuracies": test_accuracies,
        },
    }

    metrics_dest = path_model / f"metrics_{run_id}.json"
    save_metrics(metrics_dict=metrics, save_path=metrics_dest)

    print(f"\n === Training Values : ===\n")
    print(f" Average Train Loss over {num_epochs} epochs: {average_train_loss:.4f}\n")
    print(f" Average Test Loss over {num_epochs} epochs: {average_test_loss:.4f}\n")
    print(
        f" Average Test Accuracy over {num_epochs} epochs: {average_test_accuracies:.4f}"
    )
    print(f"\n === Metrics and Artifacts Saved : ===\n")
    print(f"\n Models saved at : {model_destination}\n")
    print(f"\n Metrics saved at : {metrics_dest}\n")
    print(f"\n Plots saved at : {path_plot}\n")


if __name__ == "__main__":
    main()
