from mdu.eval.eval_utils import load_pickle
import numpy as np
from collections import defaultdict
from mdu.data.constants import DatasetName
from mdu.unc.constants import UncertaintyType
from mdu.unc.risk_metrics.constants import GName, RiskType, ApproximationType
from sklearn.metrics import roc_auc_score
from mdu.data.data_utils import split_dataset_indices
from mdu.unc.multidimensional_uncertainty import MultiDimensionalUncertainty
from mdu.unc.constants import VectorQuantileModel
import torch
import pandas as pd
from configs.uncertainty_measures_configs import (
    MAHALANOBIS_AND_BAYES_RISK,
    EXCESSES_DIFFERENT_INSTANTIATIONS,
    EXCESSES_DIFFERENT_APPROXIMATIONS,
    BAYES_DIFFERENT_APPROXIMATIONS,
    BAYES_DIFFERENT_INSTANTIATIONS,
)

UNCERTAINTY_MEASURES = BAYES_DIFFERENT_INSTANTIATIONS

MULTIDIM_MODEL = VectorQuantileModel.ENTROPIC_OT

device = torch.device("cuda:0")

if MULTIDIM_MODEL == VectorQuantileModel.CPFLOW:
    train_kwargs = {
        "lr": 1e-4,
        "num_epochs": 10,
        "batch_size": 64,
        "device": device,
    }
    multidim_params = {
        "feature_dimension": len(UNCERTAINTY_MEASURES),
        "hidden_dim": 8,
        "num_hidden_layers": 5,
        "nblocks": 4,
        "zero_softplus": False,
        "softplus_type": "softplus",
        "symm_act_first": False,
    }

elif MULTIDIM_MODEL == VectorQuantileModel.OTCP:
    train_kwargs = {
        "batch_size": 64,
        "device": device,
    }
    multidim_params = {
        "positive": True,
    }
elif MULTIDIM_MODEL == VectorQuantileModel.ENTROPIC_OT:
    train_kwargs = {
        "batch_size": 64,
        "device": device,
    }
    multidim_params = {
        "target": "exp",
        "standardize": True,
        "fit_mse_params": False,
        "eps": 0.15,
    }
else:
    raise ValueError(f"Invalid multidim model: {MULTIDIM_MODEL}")


ENSEMBLE_GROUPS = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
]

train_dataset = DatasetName.CIFAR10_NOISY.value
eval_dataset = DatasetName.CIFAR10.value

results = defaultdict(list)

for group in ENSEMBLE_GROUPS:
    all_ind_logits = []
    for model_id in group:
        eval_res = load_pickle(
            f"./model_weights/{train_dataset}/checkpoints/resnet18/CrossEntropy/{model_id}/{eval_dataset}.pkl"
        )

        logits_ind = eval_res["embeddings"]
        all_ind_logits.append(eval_res["embeddings"][None])

    y_true = eval_res["labels"]

    _, train_cond_idx, calib_idx, test_idx = split_dataset_indices(
        logits_ind,
        y_true,
        train_ratio=0.0,
        calib_ratio=0.1,
        test_ratio=0.8,
    )

    y_train_cond = y_true[train_cond_idx]
    y_calib = y_true[calib_idx]
    y_test = y_true[test_idx]

    X_train_cond = np.vstack(all_ind_logits)[:, train_cond_idx, :]
    X_calib = np.vstack(all_ind_logits)[:, calib_idx, :]
    X_test = np.vstack(all_ind_logits)[:, test_idx, :]

    y_pred = np.argmax(np.mean(X_test, axis=0), axis=-1)

    print(f"Ensemble accuracy: {np.mean(y_pred == y_test)}")

    multi_dim_uncertainty = MultiDimensionalUncertainty(
        UNCERTAINTY_MEASURES,
        multidim_model=MULTIDIM_MODEL,
        multidim_params=multidim_params,
    )
    multi_dim_uncertainty.fit(
        logits_train=X_train_cond,
        y_train=y_train_cond,
        logits_calib=X_calib,
        train_kwargs=train_kwargs,
    )

    _, uncertainty_scores = multi_dim_uncertainty.predict(X_test)

    # Compute ROC AUC between in-distribution (class 0) and OOD (class 1) using sklearn
    for k in uncertainty_scores.keys():
        correct_scores = uncertainty_scores[k][y_test == y_pred]
        incorrect_scores = uncertainty_scores[k][y_test != y_pred]

        # Concatenate scores and labels
        all_scores = np.concatenate([correct_scores, incorrect_scores])
        all_labels = np.concatenate(
            [
                np.zeros_like(correct_scores),  # class 0: correct scores
                np.ones_like(incorrect_scores),  # class 1: incorrect score
            ]
        )

        auc = roc_auc_score(all_labels, all_scores)
        results[k].append(auc)

print(f"train_cond_idx.shape: {train_cond_idx.shape}")
print(f"calib_idx.shape: {calib_idx.shape}")
print(f"test_idx.shape: {test_idx.shape}")

print(f"For metrics: {[m['print_name'] for m in UNCERTAINTY_MEASURES]}")


rows = []
for metric, aucs in results.items():
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    row = {
        "Train dataset": train_dataset,
        "Eval dataset": eval_dataset,
        "Metric": metric,
        "ROC AUC Scores": aucs,
        "Mean ROC AUC": mean_auc,
        "Std ROC AUC": std_auc,
    }
    rows.append(row)

df = pd.DataFrame(
    rows,
    columns=[
        "Train dataset",
        "Eval dataset",
        "Metric",
        "ROC AUC Scores",
        "Mean ROC AUC",
        "Std ROC AUC",
    ],
)

print("\nSummary of ROC AUCs across groups:")
print(df)
