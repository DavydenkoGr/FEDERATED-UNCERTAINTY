import sys
from pathlib import Path

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdu.unc.constants import OTTarget, SamplingMethod, ScalingType
import torch
from mdu.randomness import set_all_seeds
import numpy as np
from mdu.nn.load_models import get_model
from mdu.nn.constants import ModelName
from mdu.optim.train import train_ensembles
import torch.nn as nn
from mdu.vis.toy_plots import plot_decision_boundaries, plot_uncertainty_measures
from mdu.unc.multidimensional_uncertainty import (
    fit_transform_uncertainty_estimators,
    pretty_compute_all_uncertainties,
)
from mdu.unc.entropic_ot import EntropicOTOrdering
from mdu.eval.eval_utils import get_ensemble_predictions
from mdu.data.load_dataset import get_dataset
from mdu.data.data_utils import split_dataset
from mdu.data.constants import DatasetName
from configs.uncertainty_measures_configs import (
    MAHALANOBIS_AND_BAYES_RISK,
    GMM_AND_BAYES_RISK,
    EXCESSES_DIFFERENT_INSTANTIATIONS,
    EXCESSES_DIFFERENT_APPROXIMATIONS_LOGSCORE,
    BAYES_DIFFERENT_APPROXIMATIONS_LOGSCORE,
    BAYES_DIFFERENT_INSTANTIATIONS,
    BAYES_RISK_AND_BAYES_RISK,
)


UNCERTAINTY_MEASURES = MAHALANOBIS_AND_BAYES_RISK

seed = 1
set_all_seeds(seed)

dataset_name = DatasetName.BLOBS

n_classes = 10
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
n_members = 1
input_dim = 2
hidden_dim = 32
n_epochs = 50
batch_size = 64
lambda_ = 1.0
lr = 1e-3
criterion = nn.CrossEntropyLoss()


target = OTTarget.EXP
sampling_method = SamplingMethod.GRID
scaling_type = ScalingType.GLOBAL
grid_size = 5
n_targets_multiplier = 2
eps = 0.5
max_iters = 1000
tol = 1e-6
random_state = seed


radius = 8.0
angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

dataset_params = {
    "n_samples": 4000,
    "cluster_std": 1.0,
    "centers": centers,
}

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

multi_dim_uncertainty = EntropicOTOrdering(
    target=target,
    sampling_method=sampling_method,
    scaling_type=scaling_type,
    grid_size=grid_size,
    target_params={},
    eps=eps,
    n_targets_multiplier=n_targets_multiplier,
    max_iters=max_iters,
    random_state=random_state,
    tol=tol,
)


####
pretty_uncertainty_scores_calib, fitted_uncertainty_estimators = (
    fit_transform_uncertainty_estimators(
        uncertainty_configs=UNCERTAINTY_MEASURES,
        X_calib_logits=X_calib_logits,
        y_calib=y_calib,
        X_test_logits=X_calib_logits,
    )
)

###
scores_calib = np.column_stack(
    [scores for _, scores in pretty_uncertainty_scores_calib]
)

multi_dim_uncertainty.fit(
    scores_cal=scores_calib,
)

grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

X_grid_logits = get_ensemble_predictions(
    ensemble,
    torch.from_numpy(grid_points).to(torch.float32).to(device),
    return_logits=True,
)

pretty_uncertainty_scores_test = pretty_compute_all_uncertainties(
    uncertainty_estimators=fitted_uncertainty_estimators,
    logits_test=X_grid_logits,
)
scores_test = np.column_stack([scores for _, scores in pretty_uncertainty_scores_test])

uncertainty_scores, ordering_indices = multi_dim_uncertainty.predict(scores_test)


uncertainty_measures_dict = {k: v for k, v in pretty_uncertainty_scores_test}
uncertainty_measures_dict.update({"multidim_scores": uncertainty_scores})

plot_uncertainty_measures(
    xx=xx,
    yy=yy,
    uncertainty_measures_dict=uncertainty_measures_dict,
    X_test=X_test,
)
