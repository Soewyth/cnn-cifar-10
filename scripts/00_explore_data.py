import torch

from cnn_cifar_10.data.load_data import load_train_data
from cnn_cifar_10.config.paths import get_paths, get_root_dir


def main():
    # get paths
    root_dir = get_root_dir()
    paths = get_paths(root_dir=root_dir)
    path_data_raw = paths["data_raw"]

    # Load the data
    train_data = load_train_data(path_data_raw)

    # check shapes and types of the data
    print("\n === Exploring the training data ===")
    # check number of classes
    print(f"Classes in the training data : {train_data.classes}")
    print(f"Train data shape : {train_data.data.shape} and type : {type(train_data)}")
    # calculate the mean and std of the training data per channel
    transform = torch.tensor(train_data.data).float() / 255.0
    print(
        f"Shape of the training data after transformation for position of channels : {transform.shape}\n"
    )
    # images shape is [N, H, W, C]
    mean = torch.mean(transform, dim=[0, 1, 2])
    std = torch.std(transform, dim=[0, 1, 2])
    print("=== Mean and Std of the training data per channel ===")
    print(f"Mean of the training data per channel : {mean}")
    print(f"Std of the training data per channel : {std}")


if __name__ == "__main__":
    main()
