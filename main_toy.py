import torch
from mdu.randomness import set_all_seeds
import numpy as np
from mdu.nn.load_models import get_model
from mdu.nn.constants import ModelName
from mdu.optim.train import train_ensembles
import torch.nn as nn
from mdu.vis.toy_plots import plot_decision_boundaries, plot_uncertainty_measures
from mdu.unc.constants import VectorQuantileModel
from mdu.unc.multidimensional_uncertainty import MultiDimensionalUncertainty
from mdu.eval.eval_utils import get_ensemble_predictions
from mdu.data.load_dataset import get_dataset
from mdu.data.data_utils import split_dataset
from mdu.data.constants import DatasetName
from configs.uncertainty_measures_configs import (
    MAHALANOBIS_AND_BAYES_RISK,
    GMM_AND_BAYES_RISK,
    EXCESSES_DIFFERENT_INSTANTIATIONS,
    EXCESSES_DIFFERENT_APPROXIMATIONS,
    BAYES_DIFFERENT_APPROXIMATIONS,
    BAYES_DIFFERENT_INSTANTIATIONS,
    BAYES_RISK_AND_BAYES_RISK,
)

UNCERTAINTY_MEASURES = GMM_AND_BAYES_RISK

seed = 1
set_all_seeds(seed)

dataset_name = DatasetName.BLOBS

n_classes = 10
device = torch.device("cuda:0")
n_members = 1
input_dim = 2
hidden_dim = 32
n_epochs = 50
batch_size = 64
lambda_ = 1.0
lr = 1e-3
criterion = nn.CrossEntropyLoss()

hidden_dim_vqm = 10
n_epochs_vqm = 10
lr_vqm = 1e-4

# MULTIDIM_MODEL = VectorQuantileModel.CPFLOW
# MULTIDIM_MODEL = VectorQuantileModel.OTCP
MULTIDIM_MODEL = VectorQuantileModel.ENTROPIC_OT

if MULTIDIM_MODEL == VectorQuantileModel.CPFLOW:
    train_kwargs = {
        "lr": lr_vqm,
        "num_epochs": n_epochs_vqm,
        "batch_size": batch_size,
        "device": device,
    }
    multidim_params = {
        "feature_dimension": len(UNCERTAINTY_MEASURES),
        "hidden_dim": hidden_dim_vqm,
        "num_hidden_layers": 10,
        "nblocks": 4,
        "zero_softplus": False,
        "softplus_type": "softplus",
        "symm_act_first": False,
    }

elif MULTIDIM_MODEL == VectorQuantileModel.OTCP:
    train_kwargs = {
        "batch_size": batch_size,
        "device": device,
    }
    multidim_params = {
        "positive": True,
    }
elif MULTIDIM_MODEL == VectorQuantileModel.ENTROPIC_OT:
    train_kwargs = {
        "batch_size": batch_size,
        "device": device,
    }
    multidim_params = {
        "target": "exp",
        "standardize": False,
        "fit_mse_params": False,
        "eps": 0.1,
        "max_iters": 100,
        "tol": 1e-6,
        "random_state": seed,
    }
else:
    raise ValueError(f"Invalid multidim model: {MULTIDIM_MODEL}")


if dataset_name == DatasetName.BLOBS:
    # Generate n_classes centers uniformly on a circle
    radius = 8.0
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    dataset_params = {
        "n_samples": 4000,
        "cluster_std": 1.0,
        "centers": centers,
    }
elif dataset_name == DatasetName.MOONS:
    dataset_params = {
        "n_samples": 4000,
        "noise": 0.1,
    }
else:
    raise ValueError(f"Invalid dataset: {dataset_name}")


X, y = get_dataset(dataset_name, **dataset_params)

(
    X_train_main,
    X_train_cond,
    X_calib,
    X_test,
    y_train_main,
    y_train_cond,
    y_calib,
    y_test,
) = split_dataset(X, y)

X_tensor = torch.tensor(X_train_main, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y_train_main, dtype=torch.long, device=device)

ensemble = [
    get_model(
        ModelName.SHALLOWNET,
        n_classes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    for _ in range(n_members)
]

ensemble = train_ensembles(
    ensemble,
    X_tensor,
    y_tensor,
    batch_size,
    n_epochs,
    lambda_=lambda_,
    criterion=criterion,
    lr=lr,
)

accuracies = []
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

X_calib_tensor = torch.tensor(X_calib, dtype=torch.float32, device=device)

X_test_logits = get_ensemble_predictions(ensemble, X_test_tensor, return_logits=True)
X_calib_logits = get_ensemble_predictions(ensemble, X_calib_tensor, return_logits=True)

for i, model in enumerate(ensemble):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == y_test_tensor).sum().item()
        acc = correct / len(y_test_tensor)
        accuracies.append(acc)
        print(f"Model {i + 1} accuracy: {acc:.4f}")

grid_tensor, xx, yy = plot_decision_boundaries(
    ensemble, X_test, y_test, accuracies, device, n_classes, return_grid=True
)

multi_dim_uncertainty = MultiDimensionalUncertainty(
    UNCERTAINTY_MEASURES,
    multidim_model=MULTIDIM_MODEL,
    multidim_params=multidim_params,
    if_add_maximal_elements=True,
)

multi_dim_uncertainty.fit(
    logits_train=X_calib_logits,
    y_train=y_calib,
    logits_calib=X_calib_logits,
    train_kwargs=train_kwargs,
)

grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

X_grid_logits = get_ensemble_predictions(
    ensemble,
    torch.from_numpy(grid_points).to(torch.float32).to(device),
    return_logits=True,
)

print(X_grid_logits.shape)
if MULTIDIM_MODEL == VectorQuantileModel.CPFLOW:
    X_grid_logits = torch.from_numpy(X_grid_logits).to(torch.float32).to(device)

ordering_indices, uncertainty_scores = multi_dim_uncertainty.predict(X_grid_logits)

print(uncertainty_scores['multidim_scores'].std())

plot_uncertainty_measures(
    xx=xx,
    yy=yy,
    uncertainty_measures_dict=uncertainty_scores,
    X_test=X_test,
)
