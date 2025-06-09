import numpy as np
import torch.nn.functional as F
import torch


def compute_expected_entropy(probs):
    """
    Computes the expected entropy for a set of probability distributions.
    probs: numpy array of shape (n_models, num_points, n_classes)
    Returns: numpy array of shape (num_points,)
    """
    # Entropy for each model and point: shape (n_models, num_points)
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=2)
    # Average over models: shape (num_points,)
    expected_ent = np.mean(entropy, axis=0)
    return expected_ent


def compute_entropy_of_expectation(probs):
    """
    Computes the entropy of the mean probability vector over models.
    probs: numpy array of shape (n_models, num_points, n_classes)
    Returns: numpy array of shape (num_points,)
    """
    mean_probs = np.mean(probs, axis=0)  # shape: (num_points, n_classes)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
    return entropy


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


def evaluate_uncertainty_measures(ensemble, grid_tensor):
    """
    Evaluates expected entropy and entropy of expectation for the ensemble over the grid_tensor.
    Returns: (expected_entropy, mutual information), both numpy arrays of shape (num_points,)
    """
    probs_stack = get_ensemble_probabilities(ensemble, grid_tensor)
    exp_ent = compute_expected_entropy(probs_stack)
    ent_of_exp = compute_entropy_of_expectation(probs_stack)
    mi = ent_of_exp - exp_ent
    return exp_ent, mi
