from enum import Enum


class DatasetName(Enum):
    BLOBS = "blobs"
    MOONS = "moons"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    BLURRED_CIFAR10 = "blurred_cifar10"
    BLURRED_CIFAR100 = "blurred_cifar100"
    SVHN = "svhn"
    TINY_IMAGENET = "tiny_imagenet"
