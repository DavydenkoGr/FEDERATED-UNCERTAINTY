import sys
from pathlib import Path

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdu.eval.eval_utils import load_pickle
import numpy as np
import torch
from collections import defaultdict
from mdu.data.constants import DatasetName
from sklearn.metrics import roc_auc_score
from mdu.data.data_utils import split_dataset_indices
import pandas as pd
from mdu.unc.constants import OTTarget, SamplingMethod, ScalingType
from mdu.unc.entropic_ot import EntropicOTOrdering
from mdu.unc.multidimensional_uncertainty import (
    fit_transform_uncertainty_estimators,
    pretty_compute_all_uncertainties,
)
from configs.uncertainty_measures_configs import (
    MAHALANOBIS_AND_BAYES_RISK,
    EXCESSES_DIFFERENT_INSTANTIATIONS,
    EXCESSES_DIFFERENT_APPROXIMATIONS_LOGSCORE,
    BAYES_DIFFERENT_APPROXIMATIONS_LOGSCORE,
    BAYES_DIFFERENT_APPROXIMATIONS_SPHERICALSCORE,
    BAYES_DIFFERENT_INSTANTIATIONS,
)
from mdu.randomness import set_all_seeds


def main(
    ensemble_groups,
    train_dataset,
    eval_dataset,
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
        for model_id in group:
            eval_res = load_pickle(
                f"{weights_root}/{train_dataset}/checkpoints/resnet18/CrossEntropy/{model_id}/{eval_dataset}.pkl"
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

        uncertainty_scores_calib, fitted_uncertainty_estimators = (
            fit_transform_uncertainty_estimators(
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

        scores_calib = np.column_stack(
            [scores for _, scores in uncertainty_scores_calib]
        )
        scores_ind = np.column_stack(
            [scores for _, scores in uncertainty_scores_list_ind]
        )

        multi_dim_uncertainty.fit(
            scores_cal=scores_calib,
        )

        uncertainty_scores_list_ind.append(
            ("multidim_scores", multi_dim_uncertainty.predict(scores_ind))
        )

        # Compute ROC AUC between in-distribution (class 0) and OOD (class 1) using sklearn
        for ind_ in range(len(uncertainty_scores_list_ind)):
            k = uncertainty_scores_list_ind[ind_][0]
            ind_scores = uncertainty_scores_list_ind[ind_][1]

            correct_scores = ind_scores[y_test == y_pred]
            incorrect_scores = ind_scores[y_test != y_pred]

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

    print(f"For metrics: {[m['print_name'] for m in uncertainty_measures]}")

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

    return df


if __name__ == "__main__":
    seed = 42
    UNCERTAINTY_MEASURES = EXCESSES_DIFFERENT_APPROXIMATIONS_LOGSCORE
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
    eval_dataset = DatasetName.CIFAR10.value
    weights_root = "./resources/model_weights"

    target = OTTarget.EXP
    sampling_method = SamplingMethod.GRID
    scaling_type = ScalingType.FEATURE_WISE
    grid_size = 5
    n_targets_multiplier = 10
    eps = 0.5
    max_iters = 1000
    tol = 1e-6
    random_state = seed

    df = main(
        ensemble_groups=ENSEMBLE_GROUPS,
        train_dataset=ind_dataset,
        eval_dataset=eval_dataset,
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
