import torchvision.models as models
from torch import nn


def get_resnet18():
    """Get the ResNet-18 model."""
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )  # Load the pre-trained modnel with default weights
    in_features = model.fc.in_features  # Get nb of input features of the last fc layer
    model.fc = nn.Linear(
        in_features, 10
    )  # take input feat and output 10 classes (for cifar-10)
    return model
