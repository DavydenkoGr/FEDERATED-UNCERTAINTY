import sys
from pathlib import Path

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from configs.uncertainty_measures_configs import (
    ADDITIVE_TOTALS, BAYES_DIFFERENT_APPROXIMATIONS_LOGSCORE,
    BAYES_DIFFERENT_APPROXIMATIONS_SPHERICALSCORE,
    BAYES_DIFFERENT_INSTANTIATIONS,
    BAYES_RISK_AND_BAYES_RISK,
    CORRESPONDING_COMPONENTS_TO_ADDITIVE_TOTALS, EAT_M,
    EXCESSES_DIFFERENT_APPROXIMATIONS_BRIERSCORE,
    EXCESSES_DIFFERENT_APPROXIMATIONS_LOGSCORE,
    EXCESSES_DIFFERENT_APPROXIMATIONS_SPHERICALSCORE,
    EXCESSES_DIFFERENT_INSTANTIATIONS, MAHALANOBIS_AND_BAYES_RISK,
    MULTIPLE_SAME_MEASURES,
    SINGLE_AND_FAKE,
    SINGLE_MEASURE
    )
from mdu.data.constants import DatasetName
from mdu.data.data_utils import split_dataset_indices
from mdu.eval.eval_utils import load_pickle
from mdu.randomness import set_all_seeds
from mdu.unc.constants import OTTarget, SamplingMethod, ScalingType
from mdu.unc.entropic_ot import EntropicOTOrdering
from mdu.unc.multidimensional_uncertainty import (
    fit_and_apply_uncertainty_estimators, pretty_compute_all_uncertainties)


def main(
    ensemble_groups,
    ind_dataset,
    ood_dataset,
    uncertainty_measures,
    weights_root,
    seed,
    target,
    sampling_method,
    scaling_type,
    grid_size,
    n_targets_multiplier,
    eps,
    max_iters,
    tol,
):
    set_all_seeds(seed)
    results = defaultdict(list)

    for group in ensemble_groups:
        all_ind_logits = []
        all_ood_logits = []
        for model_id in group:
            ind_res = load_pickle(
                f"{weights_root}/{ind_dataset}/checkpoints/resnet18/CrossEntropy/{model_id}/{ind_dataset}.pkl"
            )
            ood_res = load_pickle(
                f"{weights_root}/{ind_dataset}/checkpoints/resnet18/CrossEntropy/{model_id}/{ood_dataset}.pkl"
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
            random_state=seed,
        )

        y_train_cond = y_ind[train_cond_idx]

        X_train_cond = np.vstack(all_ind_logits)[:, train_cond_idx, :]
        X_calib = np.vstack(all_ind_logits)[:, calib_idx, :]
        X_test = np.vstack(all_ind_logits)[:, test_idx, :]

        X_ood = np.vstack(all_ood_logits)

        multi_dim_uncertainty = EntropicOTOrdering(
            target=target,
            sampling_method=sampling_method,
            scaling_type=scaling_type,
            grid_size=grid_size,
            target_params={},
            eps=eps,
            n_targets_multiplier=n_targets_multiplier,
            max_iters=max_iters,
            random_state=seed,
            tol=tol,
        )

        uncertainty_scores_calib, fitted_uncertainty_estimators = (
            fit_and_apply_uncertainty_estimators(
                uncertainty_configs=UNCERTAINTY_MEASURES,
                X_calib_logits=X_train_cond,
                y_calib=y_train_cond,
                X_test_logits=X_calib,
            )
        )

        uncertainty_scores_list_ind = pretty_compute_all_uncertainties(
            uncertainty_estimators=fitted_uncertainty_estimators,
            logits_test=X_test,
        )
        uncertainty_scores_list_ood = pretty_compute_all_uncertainties(
            uncertainty_estimators=fitted_uncertainty_estimators,
            logits_test=X_ood,
        )

        ###
        scores_calib = np.column_stack(
            [scores for _, scores in uncertainty_scores_calib]
        )
        scores_ind = np.column_stack(
            [scores for _, scores in uncertainty_scores_list_ind]
        )
        scores_ood = np.column_stack(
            [scores for _, scores in uncertainty_scores_list_ood]
        )

        multi_dim_uncertainty.fit(
            scores_cal=scores_calib,
        )

        uncertainty_scores_list_ind.append(
            ("multidim_scores", multi_dim_uncertainty.predict(scores_ind))
        )
        uncertainty_scores_list_ood.append(
            ("multidim_scores", multi_dim_uncertainty.predict(scores_ood))
        )

        # Compute ROC AUC between in-distribution (class 0) and OOD (class 1) using sklearn
        for ind_ in range(len(uncertainty_scores_list_ind)):
            k = uncertainty_scores_list_ind[ind_][0]
            ind_scores = uncertainty_scores_list_ind[ind_][1]
            ood_scores = uncertainty_scores_list_ood[ind_][1]

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

    print(f"For metrics: {[m['print_name'] for m in uncertainty_measures]}")

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

    return df


if __name__ == "__main__":
    seed = 42
    # UNCERTAINTY_MEASURES = MAHALANOBIS_AND_BAYES_RISK # + BAYES_RISK_AND_BAYES_RISK + EXCESSES_DIFFERENT_INSTANTIATIONS
    UNCERTAINTY_MEASURES = SINGLE_AND_FAKE
    print(UNCERTAINTY_MEASURES)
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    ENSEMBLE_GROUPS = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
    ]

    ind_dataset = DatasetName.CIFAR10.value
    ood_dataset = DatasetName.CIFAR100.value
    weights_root = "./resources/model_weights"

    target = OTTarget.EXP
    sampling_method = SamplingMethod.GRID
    scaling_type = ScalingType.FEATURE_WISE
    grid_size = 5
    n_targets_multiplier = 1
    eps = 0.5
    max_iters = 1000
    tol = 1e-6
    random_state = seed

    df = main(
        ensemble_groups=ENSEMBLE_GROUPS,
        ind_dataset=ind_dataset,
        ood_dataset=ood_dataset,
        uncertainty_measures=UNCERTAINTY_MEASURES,
        weights_root=weights_root,
        seed=seed,
        target=target,
        sampling_method=sampling_method,
        scaling_type=scaling_type,
        grid_size=grid_size,
        n_targets_multiplier=n_targets_multiplier,
        eps=eps,
        max_iters=max_iters,
        tol=tol,
    )
    print(df)
    df = df.drop(columns=['ROC AUC Scores']).round(4)
    df.to_csv("./ood_results.csv", index=False)
    print(df.to_latex())
