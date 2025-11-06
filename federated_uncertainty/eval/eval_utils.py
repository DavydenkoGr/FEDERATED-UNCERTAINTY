import numpy as np
import torch.nn.functional as F
import torch
import pickle
from typing import Any
from mdu.data.constants import DatasetName
from mdu.data.data_utils import split_dataset_indices

def load_pickle(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def get_ensemble_predictions(
    ensemble: list[torch.nn.Module],
    input_tensor: torch.Tensor,
    return_logits: bool = True,
) -> np.ndarray:
    """
    Evaluates the ensemble on the input_tensor and returns the softmax probabilities.
    Returns: numpy array of shape (n_models, num_points, n_classes)
    """
    pred_list = []
    for model in ensemble:
        model.eval()
        with torch.no_grad():
            logits: torch.Tensor = model(input_tensor)
            if return_logits:
                pred_list.append(logits.cpu().numpy())
            else:
                probs = F.softmax(logits, dim=1)
                pred_list.append(probs.cpu().numpy())
    pred_stack = np.stack(pred_list, axis=0)  # shape: (n_models, num_points, n_classes)
    return pred_stack


def load_predictions_and_split(
    ind_dataset_: DatasetName,
    weights_root: str = "./resources/model_weights",
    ensemble_groups: list = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
    ],
    train_ratio: float = 0.0,
    calib_ratio: float = 0.1,
    test_ratio: float = 0.8,
    random_state: int = 42,
):
    """Load predictions from pickle files and split data"""
    ind_dataset = ind_dataset_.value

    results_by_group = {}

    for group_idx, group in enumerate(ensemble_groups):
        # Load logits from ensemble models in this group
        all_ind_logits = []
        for model_id in group:
            if ind_dataset == DatasetName.TINY_IMAGENET.value:
                eval_res = load_pickle(
                    f"{weights_root}/{ind_dataset}/{model_id}/{ind_dataset}.pkl"
                )
            else:
                eval_res = load_pickle(
                    f"{weights_root}/{ind_dataset}/checkpoints/resnet18/CrossEntropy/{model_id}/{ind_dataset}.pkl"
                )
            all_ind_logits.append(eval_res["embeddings"][None])

        y_true = eval_res["labels"]
        logits_ind = eval_res["embeddings"]  # Use last model's logits for splitting

        # Split dataset indices (same split for all groups)
        _, train_cond_idx, calib_idx, test_idx = split_dataset_indices(
            logits_ind,
            y_true,
            train_ratio=train_ratio,
            calib_ratio=calib_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )

        # Split labels
        y_train_cond = y_true[train_cond_idx]
        y_calib = y_true[calib_idx]
        y_test = y_true[test_idx]

        # Split features (logits from ensemble)
        ensemble_logits = np.vstack(
            all_ind_logits
        )  # Shape: (n_models_in_group, n_samples, n_classes)
        X_train_cond = ensemble_logits[:, train_cond_idx, :]
        X_calib = ensemble_logits[:, calib_idx, :]
        X_test = ensemble_logits[:, test_idx, :]

        # Compute ensemble predictions for this group
        y_pred = np.argmax(np.mean(X_test, axis=0), axis=-1)
        ensemble_accuracy = np.mean(y_pred == y_test)

        results_by_group[f"group_{group_idx}"] = {
            "group_models": group,
            "X_train_cond": X_train_cond,
            "X_calib": X_calib,
            "X_test": X_test,
            "y_train_cond": y_train_cond,
            "y_calib": y_calib,
            "y_test": y_test,
            "y_pred": y_pred,
            "ensemble_accuracy": ensemble_accuracy,
        }

    return results_by_group