import torchvision.models as models
from torch import nn


def get_resnet18(pretrained: bool = True, freeze_backbone: bool = True):
    """Get the ResNet-18 model."""

    # Load the pre-trained model with default weights
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    #  Freeze the pre-trained weights

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the last fc layer with a new one that has 10 output classes (for CIFAR-10)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)

    # Unfreeze the last fc layer (already unfrozen by default, but its explicitly set)
    if freeze_backbone:
        for param in model.fc.parameters():
            param.requires_grad = True

    return model
