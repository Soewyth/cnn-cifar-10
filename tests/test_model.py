import torch

from cnn_cifar_10.models.resnet import get_resnet18


def test_output_shape():
    model = get_resnet18()
    x = torch.rand(4, 3, 224, 224)  # Batch of 4 images, 3 channels, 224x224 pixels
    output = model(x)
    assert output.shape == (4, 10), f"Expected output shape (4, 10), but got {output.shape}"


def test_logits_not_softmax():
    model = get_resnet18()
    x = torch.rand(4, 3, 224, 224)  # Batch of 4 images, 3 channels, 224x224 pixels
    output = model(x)
    row_sums = output.sum(dim=1)
    is_softmax = torch.allclose(row_sums, torch.ones_like(row_sums))
    assert not is_softmax, "Output should not be softmax probabilities"
