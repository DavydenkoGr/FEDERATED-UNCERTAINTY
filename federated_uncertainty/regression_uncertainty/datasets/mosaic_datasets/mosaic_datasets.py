import random
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets
from .utils import (
    make_gray_tensor_transform,
    mosaic_2x2,
    number_from_digits,
)


def compute_mnist_normalization_stats(
    root: str, tile_size: int = 32, download: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Compute per-channel mean and std of MNIST pixels after transformation to the specified tile size.
    Returns (mean_list, std_list) where each list has one element per channel.
    For grayscale images, this will be single-element lists.
    """
    # Load MNIST training set
    transform_basic = make_gray_tensor_transform(tile_size, normalize=False)
    mnist_train = datasets.MNIST(
        root=root, train=True, download=download, transform=transform_basic
    )

    # Sample a subset for statistics computation (to avoid memory issues)
    n_samples = min(10000, len(mnist_train))
    indices = torch.randperm(len(mnist_train))[:n_samples]

    # Collect all images to compute channel-wise statistics
    all_images = []
    for idx in indices:
        img, _ = mnist_train[idx]
        all_images.append(img)

    # Stack all images: [n_samples, channels, height, width]
    stacked_images = torch.stack(all_images)

    # Compute mean and std per channel
    # Shape: [channels] after computing over batch and spatial dimensions
    mean_per_channel = stacked_images.mean(
        dim=(0, 2, 3)
    )  # Mean over batch, height, width
    std_per_channel = stacked_images.std(dim=(0, 2, 3))  # Std over batch, height, width

    return mean_per_channel.tolist(), std_per_channel.tolist()


class QuadDigitRegressionMNIST(Dataset):
    """
    Creates 2x2 MNIST mosaics with regression label equal to the 4-digit number
    formed by (TL, TR, BL, BR). Ensures TL != 0 by construction (resampling).
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        n_samples: Optional[int] = None,
        tile_size: int = 32,
        seed: int = 0,
        first_digit_nonzero: bool = True,
        normalize_target_to_unit: bool = True,
        download: bool = True,
        normalize_images: bool = True,
    ):
        super().__init__()
        self.rng = random.Random(seed)
        self.tile_size = tile_size
        self.normalize_images = normalize_images

        # Compute normalization statistics if needed
        if normalize_images:
            self.norm_mean, self.norm_std = compute_mnist_normalization_stats(
                root, tile_size, download
            )
            self.transform = make_gray_tensor_transform(
                tile_size, normalize=True, mean=self.norm_mean, std=self.norm_std
            )
        else:
            self.norm_mean, self.norm_std = None, None
            self.transform = make_gray_tensor_transform(tile_size, normalize=False)

        self.mnist = datasets.MNIST(
            root=root, train=train, download=download, transform=None
        )

        self.n_samples = n_samples if n_samples is not None else len(self.mnist)

        if hasattr(self.mnist, "targets"):
            labels = self.mnist.targets
            self.labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        else:
            self.labels = [self.mnist[i][1] for i in range(len(self.mnist))]

        self.idx_nonzero = [i for i, y in enumerate(self.labels) if y != 0]
        self.idx_all = list(range(len(self.mnist)))

        if first_digit_nonzero and len(self.idx_nonzero) == 0:
            raise RuntimeError("No nonzero-digit samples found in MNIST?")

        self.quads: List[Tuple[int, int, int, int]] = []
        for _ in range(self.n_samples):
            tl = (
                self.rng.choice(self.idx_nonzero)
                if first_digit_nonzero
                else self.rng.choice(self.idx_all)
            )
            tr = self.rng.choice(self.idx_all)
            bl = self.rng.choice(self.idx_all)
            br = self.rng.choice(self.idx_all)
            self.quads.append((tl, tr, bl, br))

        self.normalize_target_to_unit = normalize_target_to_unit

    def __len__(self) -> int:
        return self.n_samples

    def _get_img_label(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, y = self.mnist[idx]
        img_t = self.transform(img)
        return img_t, int(y)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tl, tr, bl, br = self.quads[i]
        (img_tl, y_tl) = self._get_img_label(tl)
        (img_tr, y_tr) = self._get_img_label(tr)
        (img_bl, y_bl) = self._get_img_label(bl)
        (img_br, y_br) = self._get_img_label(br)

        x = mosaic_2x2([img_tl, img_tr, img_bl, img_br])

        digits = (y_tl, y_tr, y_bl, y_br)
        target_val = float(number_from_digits(digits))
        if self.normalize_target_to_unit:
            target_val = target_val / 9999.0

        y_reg = torch.tensor(target_val, dtype=torch.float32)
        return x, y_reg

    def get_normalization_stats(
        self,
    ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Returns the normalization statistics (mean_list, std_list) used for this dataset.
        Each list contains per-channel statistics.
        Returns (None, None) if normalization is disabled.
        """
        return self.norm_mean, self.norm_std


class QuadOODDataset(Dataset):
    """
    Builds 2x2 mosaics from ANY torchvision dataset of images (ignores labels).
    Targets are NaN by default (OOD for a regression model trained on MNIST quads).
    """

    def __init__(
        self,
        base_ds: Dataset,
        n_samples: Optional[int] = None,
        tile_size: int = 32,
        seed: int = 0,
        target_value: Optional[float] = None,
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
    ):
        super().__init__()
        self.rng = random.Random(seed)
        self.base = base_ds
        self.n_samples = n_samples if n_samples is not None else len(base_ds)

        # Apply same normalization as the ID dataset if provided
        if normalize_mean is not None and normalize_std is not None:
            self.transform = make_gray_tensor_transform(
                tile_size, normalize=True, mean=normalize_mean, std=normalize_std
            )
        else:
            self.transform = make_gray_tensor_transform(tile_size, normalize=False)

        self.target_val = target_value

        self.idx_all = list(range(len(self.base)))
        self.quads: List[Tuple[int, int, int, int]] = []
        for _ in range(self.n_samples):
            tl = self.rng.choice(self.idx_all)
            tr = self.rng.choice(self.idx_all)
            bl = self.rng.choice(self.idx_all)
            br = self.rng.choice(self.idx_all)
            self.quads.append((tl, tr, bl, br))

    def __len__(self) -> int:
        return self.n_samples

    def _get_img(self, idx: int) -> torch.Tensor:
        item = self.base[idx]
        img = item[0] if isinstance(item, (tuple, list)) else item
        return self.transform(img)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tl, tr, bl, br = self.quads[i]
        x = mosaic_2x2(
            [self._get_img(tl), self._get_img(tr), self._get_img(bl), self._get_img(br)]
        )
        if self.target_val is None:
            y = torch.tensor(float("nan"), dtype=torch.float32)
        else:
            y = torch.tensor(float(self.target_val), dtype=torch.float32)
        return x, y


        y_reg = torch.tensor(target_val, dtype=torch.float32)
        return x, y_reg

    def get_normalization_stats(
        self,
    ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Returns the normalization statistics (mean_list, std_list) used for this dataset.
        Each list contains per-channel statistics.
        Returns (None, None) if normalization is disabled.
        """
        return self.norm_mean, self.norm_std


class QuadIDOODMixedDataset(QuadOODDataset):
    """
    Builds 2x2 mosaics from ANY torchvision dataset of images (ignores labels).
    Targets are NaN by default (OOD for a regression model trained on MNIST quads).
    """

    def __init__(
        self,
        id_ds: Dataset,
        ood_ds: Dataset,
        ood_positions: List[int] = [],
        n_samples: Optional[int] = None,
        tile_size: int = 32,
        seed: int = 0,
        target_value: Optional[float] = None,
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
    ):
        # super().__init__()
        self.rng = random.Random(seed)
        self.base_id = id_ds
        self.base_ood = ood_ds
        self.ood_positions = ood_positions

        self.n_samples = n_samples if n_samples is not None else len(id_ds)

        # Apply same normalization as the ID dataset if provided
        if normalize_mean is not None and normalize_std is not None:
            self.transform = make_gray_tensor_transform(
                tile_size, normalize=True, mean=normalize_mean, std=normalize_std
            )
        else:
            self.transform = make_gray_tensor_transform(tile_size, normalize=False)

        self.target_val = target_value

        # get the indexes for both
        self.id_idxs = list(range(len(self.base_id)))
        self.ood_idxs = list(range(len(self.base_ood)))

        self.quads: List[Tuple[int, int, int, int]] = []
        for _ in range(self.n_samples):
            tl = self.rng.choice(self.ood_idxs if 0 in self.ood_positions else self.id_idxs)
            tr = self.rng.choice(self.ood_idxs if 1 in self.ood_positions else self.id_idxs)
            bl = self.rng.choice(self.ood_idxs if 2 in self.ood_positions else self.id_idxs)
            br = self.rng.choice(self.ood_idxs if 3 in self.ood_positions else self.id_idxs)
            self.quads.append((tl, tr, bl, br))

    def _get_img(self, idx: int, pos: int) -> torch.Tensor:
        item = self.base_ood[idx] if pos in self.ood_positions else self.base_id[idx]
        img = item[0] if isinstance(item, (tuple, list)) else item
        return self.transform(img)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tl, tr, bl, br = self.quads[i]
        x = mosaic_2x2(
            [self._get_img(tl, pos=0), self._get_img(tr, pos=1), self._get_img(bl, pos=2), self._get_img(br, pos=3)]
        )
        if self.target_val is None:
            y = torch.tensor(float("nan"), dtype=torch.float32)
        else:
            y = torch.tensor(float(self.target_val), dtype=torch.float32)
        return x, y



def build_id_and_ood(
    root: str = "./data",
    tile_size: int = 32,
    seed: int = 0,
    n_id_train: Optional[int] = None,
    n_id_test: Optional[int] = None,
    n_ood_each: Optional[int] = None,
    download: bool = True,
    normalize_images: bool = True,
):
    """
    Returns:
      id_train, id_test, ood_fashion, ood_cifar10, ood_svhn, ood_mixture
    All datasets output tensors of shape [1, 2*tile_size, 2*tile_size].

    Args:
        normalize_images: Whether to normalize images to have mean 0 and std 1
    """
    # ID: MNIST (train/test)
    id_train = QuadDigitRegressionMNIST(
        root=root,
        train=True,
        n_samples=n_id_train,
        tile_size=tile_size,
        seed=seed,
        first_digit_nonzero=True,
        normalize_target_to_unit=True,
        download=download,
        normalize_images=normalize_images,
    )
    id_test = QuadDigitRegressionMNIST(
        root=root,
        train=False,
        n_samples=n_id_test,
        tile_size=tile_size,
        seed=seed + 1,
        first_digit_nonzero=True,
        normalize_target_to_unit=True,
        download=download,
        normalize_images=normalize_images,
    )

    # Get normalization statistics from the training set
    norm_mean, norm_std = id_train.get_normalization_stats()

    # OOD sources (raw)
    tform = None  # we handle transforms in the wrapper to ensure consistency
    fashion = datasets.FashionMNIST(
        root=root, train=False, download=download, transform=tform
    )
    cifar10 = datasets.CIFAR10(
        root=root, train=False, download=download, transform=tform
    )
    svhn = datasets.SVHN(root=root, split="test", download=download, transform=tform)

    # Wrap as OOD mosaics with same normalization as ID data
    ood_fashion = QuadOODDataset(
        fashion,
        n_samples=n_ood_each,
        tile_size=tile_size,
        seed=seed + 2,
        normalize_mean=norm_mean,
        normalize_std=norm_std,
    )
    ood_cifar10 = QuadOODDataset(
        cifar10,
        n_samples=n_ood_each,
        tile_size=tile_size,
        seed=seed + 3,
        normalize_mean=norm_mean,
        normalize_std=norm_std,
    )
    ood_svhn = QuadOODDataset(
        svhn,
        n_samples=n_ood_each,
        tile_size=tile_size,
        seed=seed + 4,
        normalize_mean=norm_mean,
        normalize_std=norm_std,
    )

    # Optional mixture (uniform concatenation of sources, sampled by PyTorch ConcatDataset order)
    # If you want a *single* dataset that mixes sources, you can:
    ood_mixture = ConcatDataset([ood_fashion, ood_cifar10, ood_svhn])

    return id_train, id_test, ood_fashion, ood_cifar10, ood_svhn, ood_mixture



def build_id_and_ood_emnist_one_at_a_time(
    root: str = "./data",
    tile_size: int = 32,
    seed: int = 0,
    n_id_train: Optional[int] = None,
    n_id_test: Optional[int] = None,
    n_ood_each: Optional[int] = None,
    download: bool = True,
    normalize_images: bool = True,
    ood_positions: List[int] = [],
):
    # create an id train datasets
    id_train = QuadDigitRegressionMNIST(
        root=root,
        train=True,
        n_samples=n_id_train,
        tile_size=tile_size,
        seed=seed,
        first_digit_nonzero=True,
        normalize_target_to_unit=False,
        download=download,
        normalize_images=normalize_images,
    )

    # load the two datasets
    # Get normalization statistics from the training set
    norm_mean, norm_std = id_train.get_normalization_stats()


    # load the mnist and th
    mnist = datasets.MNIST(root=root, train=False, download=download, transform=None)
    emnist = datasets.EMNIST(root=root, split="letters", download=download, transform=None)

    # return a mixed quadrant emnist ood dataset
    return QuadIDOODMixedDataset(
        mnist,
        emnist,
        ood_positions=ood_positions,
        n_samples=n_ood_each,
        tile_size=tile_size,
        seed=seed + 1,
        normalize_mean=norm_mean,
        normalize_std=norm_std,
    )
