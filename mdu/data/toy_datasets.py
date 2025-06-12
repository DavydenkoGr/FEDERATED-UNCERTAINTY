from sklearn.datasets import make_moons, make_blobs
import numpy as np


def load_toy_dataset(
    toy_dataset: str, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    if toy_dataset == "moons":
        X, y = make_moons(
            n_samples=kwargs["n_samples"], noise=kwargs["noise"]
        )
    elif toy_dataset == "blobs":
        X, y = make_blobs(
            n_samples=kwargs["n_samples"],
            centers=kwargs["centers"],
            cluster_std=kwargs["cluster_std"],
        )
    else:
        raise ValueError(
            f"Invalid toy dataset: {toy_dataset}"
        )
    return X, y
