from torchvision import transforms


def get_preprocessing_transforms(image_size: int = 224):
    """Get the preprocessing transforms for the training and test data."""
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),  # resize the image to the input size specified in the config
            transforms.ToTensor(),  # convert the image to a tensor and scale the pixel values to [0, 1]
            transforms.Normalize(
                mean=mean, std=std
            ),  # Center and reduce variance of the data
        ]
    )
