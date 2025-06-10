import torch
from mdu.randomness import set_all_seeds
import numpy as np
from mdu.nn.load_models import get_model
from mdu.nn.constants import ModelName
from mdu.optim.train import train_ensembles
import torch.nn as nn
from mdu.vis.toy_plots import plot_decision_boundaries, plot_uncertainty_measures
from mdu.unc.constants import UncertaintyType
from mdu.unc.risk_metrics.constants import GName, RiskType, ApproximationType
from mdu.unc.multidimensional_uncertainty import MultiDimensionalUncertainty
from mdu.eval.eval_utils import get_ensemble_predictions
from mdu.data.load_dataset import get_dataset
from mdu.data.constants import DatasetName


torch.manual_seed(0)
np.random.seed(0)

set_all_seeds(42)

from sklearn.model_selection import train_test_split

dataset_name = DatasetName.BLOBS
n_classes = 10
device = torch.device("cuda:0")
n_members = 2
input_dim = 2
hidden_dim = 32
n_epochs = 50
batch_size = 64
lambda_ = 1.0
criterion = nn.CrossEntropyLoss()

UNCERTAINTY_MEASURES = [
    {
        "type": UncertaintyType.RISK,
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.BAYES_RISK,
            "gt_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "pred_approx": ApproximationType.OUTER,
            "gt_approx": ApproximationType.INNER,
            "T": 1.0,
        },
    },
]

if dataset_name == DatasetName.BLOBS:
    dataset_params = {
        "n_samples": 4000,
        "cluster_std": 1.0,
    }
elif dataset_name == DatasetName.MOONS:
    dataset_params = {
        "n_samples": 4000,
        "noise": 0.1,
    }
else:
    raise ValueError(f"Invalid dataset: {dataset_name}")

X, y = get_dataset(dataset_name, n_classes, **dataset_params)


mean_point = np.mean(X, axis=0)
