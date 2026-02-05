import torch.nn as nn
import torchvision as tv


def get_resnet18(in_channels: int = 3, image_size=None):
    # two outputs for mean and variance
    network = tv.models.resnet18(weights=None, num_classes=2)
    return _adapt_for_small_images(network, in_channels, image_size=image_size)


def _adapt_for_small_images(network: nn.Module, in_channels: int, image_size=32):
    # adapt network for 32x32 images
    # TODO: use the image size variable on the first linear layer!
    network.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    network.maxpool = nn.Identity()
    return network
