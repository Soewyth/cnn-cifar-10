from pathlib import Path

import matplotlib.pyplot as plt


def plot_curves(
    train_loss: list[float],
    test_loss: list[float],
    test_accuracies: list[float],
    save_path: Path,
) -> None:
    """Generate and save training curves for loss and accuracy.
    One plots with two subplots :
        - Left : train loss and test loss per epoch
        - Right : test accuracy per epoch
    Args:
        train_loss (list[float]): List of training loss values per epoch.
        test_loss (list[float]): List of test loss values per epoch.
        test_accuracies (list[float]): List of test accuracy values per epoch.
        save_path (Path): Path to save the generated plot.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left subplot : Train loss and test loss

    axes[0].plot(train_loss, label="Train Loss", marker="o")
    axes[0].plot(test_loss, label="Test Loss", marker="o")
    axes[0].set_title("Training and Test Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Right subplot : Test accuracy
    axes[1].plot(test_accuracies, label="Test Accuracy", marker="o")
    axes[1].set_title("Test Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    # prevent overlap
    fig.tight_layout()
    # SAve the figure
    fig.savefig(save_path)
    plt.close(fig)
