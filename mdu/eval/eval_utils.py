import numpy as np
import torch.nn.functional as F
import torch


def get_ensemble_probabilities(ensemble, grid_tensor):
    """
    Evaluates the ensemble on the grid_tensor and returns the softmax probabilities.
    Returns: numpy array of shape (n_models, num_points, n_classes)
    """
    probs_list = []
    for model in ensemble:
        model.eval()
        with torch.no_grad():
            logits = model(grid_tensor)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())
    probs_stack = np.stack(
        probs_list, axis=0
    )  # shape: (n_models, num_points, n_classes)
    return probs_stack
