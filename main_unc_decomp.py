import torch
from mdu.randomness import set_all_seeds
import numpy as np
import torch.nn as nn
from mdu.unc.constants import UncertaintyType
from mdu.unc.risk_metrics.constants import GName, RiskType, ApproximationType
from mdu.data.load_dataset import get_dataset
from mdu.data.constants import DatasetName
from mdu.eval.toy_exp import eval_unc_decomp
import pandas as pd
from mdu.vis.toy_plots import plot_data_and_test_point

set_all_seeds(42)

dataset_name = DatasetName.BLOBS
n_classes = 2
device = torch.device("cuda:0")
n_members = 50
input_dim = 2
n_epochs = 1
batch_size = 64
lambda_ = 0.0
calib_ratio = 0.2
val_ratio = 0.2
lr = 1e-2
criterion = nn.CrossEntropyLoss()

UNCERTAINTY_MEASURES = [
    {
        "type": UncertaintyType.RISK,
        "kwargs": {
            "g_name": GName.BRIER_SCORE,
            "risk_type": RiskType.BAYES_RISK,
            "gt_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "kwargs": {
            "g_name": GName.BRIER_SCORE,
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


res = eval_unc_decomp(
    X=X,
    y=y,
    test_point=mean_point,
    device=device,
    uncertainty_measures=UNCERTAINTY_MEASURES,
    n_epochs=n_epochs,
    input_dim=input_dim,
    n_members=n_members,
    batch_size=batch_size,
    lambda_=lambda_,
    criterion=criterion,
    calib_ratio=calib_ratio,
    val_ratio=val_ratio,
    lr=lr,
)

uncertainty_keys = set()
for r in res:
    for k in r.keys():
        if any(
            prefix in k
            for prefix in ["aleatoric_", "epistemic_", "additive_total", "ot_scores"]
        ):
            uncertainty_keys.add(k)
uncertainty_keys = sorted(uncertainty_keys)

df_results = pd.DataFrame(
    [
        {
            "samples_per_class": r["n_samples_per_class"],
            "total_samples": r["total_samples"],
            "avg_val_acc": r["avg_val_acc"],
            "val_size": r["val_size"],
            "calib_size": r["calib_size"],
            **{k: r.get(k, float("nan")) for k in uncertainty_keys},
        }
        for r in res
    ]
)

print("Summary of Results at Midpoint (all available uncertainty metrics):")
with pd.option_context("display.max_columns", None):
    print(df_results.to_string(index=False, float_format="%.4f"))

plot_data_and_test_point(X, y, mean_point)
