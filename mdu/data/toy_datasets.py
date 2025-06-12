from sklearn.datasets import make_moons, make_blobs
import numpy as np


def load_toy_dataset(
    toy_dataset: str, n_classes: int, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    if toy_dataset == "moons" and n_classes == 2:
        X, y = make_moons(
            n_samples=kwargs["n_samples"], noise=kwargs["noise"]
        )
    elif toy_dataset == "blobs" and n_classes > 1:
        X, y = make_blobs(
            n_samples=kwargs["n_samples"],
            centers=n_classes,
            cluster_std=kwargs["cluster_std"],
        )
    else:
        raise ValueError(
            f"Invalid toy dataset: {toy_dataset} with n_classes: {n_classes}"
        )
    return X, y
