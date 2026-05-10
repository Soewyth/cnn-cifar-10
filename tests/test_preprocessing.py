import numpy as np
from PIL import Image
from torchvision import transforms as T

from cnn_cifar_10.data.preprocessing import get_preprocessing_transforms


def test_output_type_compose():
    transform = get_preprocessing_transforms()
    assert isinstance(transform, T.Compose), f"Expected Compose, got {type(transform)}"


def test_output_shape_tensor():
    transform = get_preprocessing_transforms()
    # Create a random image with shape(224, 224, 3) * 255 to scale pixel values from [0, 1] to [0, 255]
    x = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    result = transform(x)
    assert result.shape == (3, 224, 224), f"Expected (3, 224, 224), got {result.shape}"


def test_output_normalized_and_centered():
    transform = get_preprocessing_transforms()
    # Create a random image with shape(224, 224, 3) * 255 to scale pixel values from [0, 1] to [0, 255]
    x = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    result = transform(x)
    assert result.min() < 0, "Normalization should produce negative values"
    assert result.max() > 0, "Normalization should produce positive values"
