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


torch.manual_seed(0)
np.random.seed(0)

set_all_seeds(42)

from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split

toy_dataset = "moons"
n_classes = 2
device = torch.device("cuda:0")
n_members = 20
input_dim = 2
hidden_dim = 32
n_epochs = 50
batch_size = 64
lambda_ = 1.0
criterion = nn.CrossEntropyLoss()

UNCERTAINTY_MEASURES = [
    # {
    #     "type": UncertaintyType.RISK,
    #     "kwargs": {
    #         "g_name": GName.LOG_SCORE,
    #         "risk_type": RiskType.BAYES_RISK,
    #         "gt_approx": ApproximationType.OUTER,
    #         "T": 1.0,
    #     },
    # },
    # {
    #     "type": UncertaintyType.RISK,
    #     "kwargs": {
    #         "g_name": GName.BRIER_SCORE,
    #         "risk_type": RiskType.BAYES_RISK,
    #         "gt_approx": ApproximationType.OUTER,
    #         "T": 1.0,
    #     },
    # },
    #     {
    #     "type": UncertaintyType.RISK,
    #     "kwargs": {
    #         "g_name": GName.SPHERICAL_SCORE,
    #         "risk_type": RiskType.BAYES_RISK,
    #         "gt_approx": ApproximationType.OUTER,
    #         "T": 1.0,
    #     },
    # },
    #     {
    #     "type": UncertaintyType.RISK,
    #     "kwargs": {
    #         "g_name": GName.ZERO_ONE_SCORE,
    #         "risk_type": RiskType.BAYES_RISK,
    #         "gt_approx": ApproximationType.OUTER,
    #         "T": 1.0,
    #     },
    # },
    {
        "type": UncertaintyType.RISK,
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.OUTER,
            "pred_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.INNER,
            "pred_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.OUTER,
            "pred_approx": ApproximationType.INNER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.OUTER,
            "pred_approx": ApproximationType.CENTRAL,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.INNER,
            "pred_approx": ApproximationType.CENTRAL,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.CENTRAL,
            "pred_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.CENTRAL,
            "pred_approx": ApproximationType.INNER,
            "T": 1.0,
        },
    },
    # {
    #     "type": UncertaintyType.MAHALANOBIS,
    #     "kwargs": {},
    # },
]


if toy_dataset == "moons":
    if n_classes != 2:
        raise ValueError("n_classes must be 2 for moons dataset")
    X, y = make_moons(n_samples=4000, noise=0.1, random_state=42)
elif toy_dataset == "blobs":
    X, y = make_blobs(
        n_samples=4000, centers=n_classes, cluster_std=1.0, random_state=42
    )
else:
    raise ValueError(f"Invalid toy dataset: {toy_dataset}")


X_train_main, X_temp, y_train_main, y_temp = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)
X_train_cond, X_temp2, y_train_cond, y_temp2 = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
X_calib, X_test, y_calib, y_test = train_test_split(
    X_temp2, y_temp2, test_size=0.5, random_state=42, stratify=y_temp2
)

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
)

accuracies = []
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)
X_calib_tensor = torch.tensor(X_calib, dtype=torch.float32, device=device)

X_test_logits = get_ensemble_predictions(
    ensemble, X_test_tensor, device, return_logits=True
)
X_calib_logits = get_ensemble_predictions(
    ensemble, X_calib_tensor, device, return_logits=True
)

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

multi_dim_uncertainty = MultiDimensionalUncertainty(UNCERTAINTY_MEASURES)
multi_dim_uncertainty.fit(X_calib_logits, X_calib_logits)

grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

X_grid_logits = get_ensemble_predictions(
    ensemble,
    torch.from_numpy(grid_points).to(torch.float32).to(device),
    device,
    return_logits=True,
)

ordering_indices, uncertainty_scores = multi_dim_uncertainty.predict(X_grid_logits)

plot_uncertainty_measures(
    xx=xx,
    yy=yy,
    uncertainty_measures_dict=uncertainty_scores,
    X_test=X_test,
)
