from .toy_datasets import load_toy_dataset
import numpy as np
from .constants import DatasetName


def get_dataset(dataset: DatasetName, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    if dataset in [DatasetName.BLOBS, DatasetName.MOONS]:
        return load_toy_dataset(dataset.value, **kwargs)
    else:
        raise ValueError(f"Invalid dataset: {dataset.value}")
