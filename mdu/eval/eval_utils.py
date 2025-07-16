import numpy as np
import torch.nn.functional as F
import torch
import pickle
from typing import Any


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
