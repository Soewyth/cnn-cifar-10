from torchvision import datasets


def load_train_data(data_dir, transform=None):
    """Loads the training data from the specified directory."""
    return datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)


def load_test_data(data_dir, transform=None):
    """Loads the test data from the specified directory."""
    return datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
