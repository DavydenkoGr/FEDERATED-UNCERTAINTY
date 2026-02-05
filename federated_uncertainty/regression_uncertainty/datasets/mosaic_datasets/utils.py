import torch
from torchvision import transforms
from typing import Sequence, Optional, List
from torch.utils.data import ConcatDataset
from torchvision import datasets


def make_gray_tensor_transform(
    tile_size: int,
    normalize: bool = True,
    mean: List[float] = [0.0],
    std: List[float] = [1.0],
) -> transforms.Compose:
    """
    Standardize all sources to 1xHxW tensors with values in [0,1] and optionally normalize.

    Args:
        tile_size: Size to resize images to
        normalize: Whether to apply normalization
        mean: List of means per channel for normalization (only used if normalize=True)
        std: List of standard deviations per channel for normalization (only used if normalize=True)
    """
    transform_list = [
        transforms.Resize((tile_size, tile_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


def mosaic_2x2(imgs: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    imgs: four tensors [C,H,W] (top-left, top-right, bottom-left, bottom-right)
    return: single tensor [C, 2H, 2W]
    """
    assert len(imgs) == 4
    tl, tr, bl, br = imgs

    C, H, W = tl.shape
    for t in (tr, bl, br):
        assert t.shape == (C, H, W), f"Tile shapes differ: {t.shape} vs {(C, H, W)}"

    top = torch.cat([tl, tr], dim=2)
    bottom = torch.cat([bl, br], dim=2)
    return torch.cat([top, bottom], dim=1)


def number_from_digits(d: Sequence[int]) -> int:
    """
    Convert 4 digits to a 4-digit number: d0 d1 d2 d3 -> d0*1000 + d1*100 + d2*10 + d3
    """
    return int(d[0]) * 1000 + int(d[1]) * 100 + int(d[2]) * 10 + int(d[3])
