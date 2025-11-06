from enum import Enum


class DatasetName(Enum):
    BLOBS = "blobs"
    MOONS = "moons"
    CIFAR10 = "cifar10"
    CIFAR10C_1 = "cifar10c_1"
    CIFAR10C_2 = "cifar10c_2"
    CIFAR10C_3 = "cifar10c_3"
    CIFAR10C_4 = "cifar10c_4"
    CIFAR10C_5 = "cifar10c_5"
    CIFAR100 = "cifar100"
    CIFAR10_NOISY = "cifar10_noisy_label"
    CIFAR100_NOISY = "cifar100_noisy_label"
    BLURRED_CIFAR10 = "blurred_cifar10"
    BLURRED_CIFAR100 = "blurred_cifar100"
    SVHN = "svhn"
    TINY_IMAGENET = "tiny_imagenet"
    IMAGENET_A = "imagenet_a"
    IMAGENET_O = "imagenet_o"
    IMAGENET_R = "imagenet_r"
