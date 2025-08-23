from mdu.eval.eval_utils import load_pickle
import numpy as np
import torch
from collections import defaultdict
from mdu.data.constants import DatasetName
from sklearn.metrics import roc_auc_score
from mdu.data.data_utils import split_dataset_indices
from mdu.unc.constants import VectorQuantileModel
from mdu.unc.multidimensional_uncertainty import MultiDimensionalUncertainty
import pandas as pd
from configs.uncertainty_measures_configs import (
    MAHALANOBIS_AND_BAYES_RISK,
    EXCESSES_DIFFERENT_INSTANTIATIONS,
    EXCESSES_DIFFERENT_APPROXIMATIONS,
    BAYES_DIFFERENT_APPROXIMATIONS,
    BAYES_DIFFERENT_INSTANTIATIONS,
    BAYES_RISK_AND_BAYES_RISK,
    SINGLE_MEASURE,
)

UNCERTAINTY_MEASURES = MAHALANOBIS_AND_BAYES_RISK # + BAYES_RISK_AND_BAYES_RISK + EXCESSES_DIFFERENT_INSTANTIATIONS

# MULTIDIM_MODEL = VectorQuantileModel.CPFLOW
# MULTIDIM_MODEL = VectorQuantileModel.OTCP
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
        "eps": 0.25,
    }
else:
    raise ValueError(f"Invalid multidim model: {MULTIDIM_MODEL}")


ENSEMBLE_GROUPS = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
]

ind_dataset = DatasetName.CIFAR10.value
ood_dataset = DatasetName.TINY_IMAGENET.value

results = defaultdict(list)

for group in ENSEMBLE_GROUPS:
    all_ind_logits = []
    all_ood_logits = []
    for model_id in group:
        ind_res = load_pickle(
            f"./model_weights/{ind_dataset}/checkpoints/resnet18/CrossEntropy/{model_id}/{ind_dataset}.pkl"
        )
        ood_res = load_pickle(
            f"./model_weights/{ind_dataset}/checkpoints/resnet18/CrossEntropy/{model_id}/{ood_dataset}.pkl"
        )

        logits_ind = ind_res["embeddings"]
        all_ind_logits.append(ind_res["embeddings"][None])
        all_ood_logits.append(ood_res["embeddings"][None])

    y_ind = ind_res["labels"]
    y_ood = ood_res["labels"]

    _, train_cond_idx, calib_idx, test_idx = split_dataset_indices(
        logits_ind,
        y_ind,
        train_ratio=0.0,
        calib_ratio=0.1,
        test_ratio=0.8,
    )

    y_train_cond = y_ind[train_cond_idx]
    y_calib = y_ind[calib_idx]

    X_train_cond = np.vstack(all_ind_logits)[:, train_cond_idx, :]
    X_calib = np.vstack(all_ind_logits)[:, calib_idx, :]
    X_test = np.vstack(all_ind_logits)[:, test_idx, :]

    X_ood = np.vstack(all_ood_logits)

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
    if MULTIDIM_MODEL == VectorQuantileModel.CPFLOW:
        X_test = torch.from_numpy(X_test).to(torch.float32).to(train_kwargs["device"])
        X_ood = torch.from_numpy(X_ood).to(torch.float32).to(train_kwargs["device"])

    _, uncertainty_scores_ind = multi_dim_uncertainty.predict(X_test)
    _, uncertainty_scores_ood = multi_dim_uncertainty.predict(X_ood)

    # Compute ROC AUC between in-distribution (class 0) and OOD (class 1) using sklearn
    for k in uncertainty_scores_ind.keys():
        ind_scores = uncertainty_scores_ind[k]
        ood_scores = uncertainty_scores_ood[k]

        # Concatenate scores and labels
        all_scores = np.concatenate([ind_scores, ood_scores])
        all_labels = np.concatenate(
            [
                np.zeros_like(ind_scores),  # class 0: in-distribution
                np.ones_like(ood_scores),  # class 1: OOD
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
        "In-distribution": ind_dataset,
        "Out-of-distribution": ood_dataset,
        "Metric": metric,
        "ROC AUC Scores": aucs,
        "Mean ROC AUC": mean_auc,
        "Std ROC AUC": std_auc,
    }
    rows.append(row)

df = pd.DataFrame(
    rows,
    columns=[
        "In-distribution",
        "Out-of-distribution",
        "Metric",
        "ROC AUC Scores",
        "Mean ROC AUC",
        "Std ROC AUC",
    ],
)

print("\nSummary of ROC AUCs across groups:")
print(df)
