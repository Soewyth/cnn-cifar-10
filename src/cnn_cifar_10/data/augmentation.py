from torchvision import transforms


def get_augmentation_transforms():
    """Get the augmentation transforms for the training data."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                32, padding=4
            ),  # Randomly crop the image to 32x32 with a padding of 4 pixels on each side
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    )
