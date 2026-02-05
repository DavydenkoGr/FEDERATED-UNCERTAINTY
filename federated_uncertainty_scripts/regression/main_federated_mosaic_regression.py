#!/usr/bin/env python3
"""
Federated training script for mosaic regression task.

This script implements federated ensemble training for regression models on 2x2 mosaic datasets.
Each client has access to mosaics built from a subset of digits (0-9), and selects models
from a shared model pool based on various strategies.

Example usage:
    python main_federated_mosaic_regression.py --n_models 20 --n_clients 5 --ensemble_size 3
"""

import sys
import argparse
from pathlib import Path
import datetime
import random
import copy
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets

# Project root = parent of "federated_uncertainty_scripts"
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated_uncertainty.regression_uncertainty.datasets.mosaic_datasets import (
    build_id_and_ood,
    QuadDigitRegressionMNIST,
)
from federated_uncertainty.regression_uncertainty.datasets.mosaic_datasets.utils import (
    number_from_digits,
)
from federated_uncertainty.regression_uncertainty.source.models.cnn import get_cnn
from federated_uncertainty.regression_uncertainty.source.models.resnet import get_resnet18
from federated_uncertainty.regression_uncertainty.source.trainer import fit
from federated_uncertainty.regression_uncertainty.source.utils.seeding import fix_seeds
from federated_uncertainty.regression_uncertainty.source.models.utils import (
    variance_link,
    natural_link,
)
from federated_uncertainty.regression_uncertainty.source.utils.objectives import (
    GaussianNLL,
    NaturalGaussianNLL,
    compute_anti_regularization,
)


class FilteredMosaicDataset(Dataset):
    """
    Wrapper around QuadDigitRegressionMNIST that filters samples by allowed digits.
    Only includes mosaics where all 4 digits are from the allowed set.
    """
    
    def __init__(self, base_dataset: QuadDigitRegressionMNIST, allowed_digits: list):
        self.base_dataset = base_dataset
        self.allowed_digits = set(allowed_digits)
        
        # Filter indices: keep only samples where all digits are in allowed_digits
        self.valid_indices = []
        for i in range(len(base_dataset)):
            # Get the quad indices
            tl, tr, bl, br = base_dataset.quads[i]
            # Get the digit labels
            y_tl = base_dataset.labels[tl]
            y_tr = base_dataset.labels[tr]
            y_bl = base_dataset.labels[bl]
            y_br = base_dataset.labels[br]
            
            # Check if all digits are in allowed set
            if all(d in self.allowed_digits for d in [y_tl, y_tr, y_bl, y_br]):
                self.valid_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.valid_indices[idx]]


def digits_from_number(normalized_value: float) -> tuple:
    """
    Extract digits from normalized regression target.
    Assumes target was normalized by dividing by 9999.0
    """
    # Denormalize
    number = int(normalized_value * 9999.0)
    # Extract digits
    d0 = number // 1000
    d1 = (number // 100) % 10
    d2 = (number // 10) % 10
    d3 = number % 10
    return (d0, d1, d2, d3)


def train_ensembles_w_local_data_regression(
    models: list[nn.Module],
    train_loaders: list[DataLoader],
    device: str,
    n_epochs: int,
    lambda_antireg: float,
    lr: float,
    use_natural: bool = False,
):
    """
    Train ensemble models for regression, each on its own data loader.
    Adapted from classification version for regression task.
    """
    n_members = len(models)
    
    if use_natural:
        loss_fn = NaturalGaussianNLL()
        link_fn = natural_link
    else:
        loss_fn = GaussianNLL()
        link_fn = variance_link
    
    for i, model in enumerate(models):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        print(f"\nTraining model {i + 1}/{n_members}")
        for epoch in range(n_epochs):
            epoch_losses = []
            for xb, yb in train_loaders[i]:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                output = model(xb)
                mean, var = link_fn(output)
                loss = loss_fn(mean, var, yb)
                
                # Anti-regularization
                if lambda_antireg > 0:
                    anti_reg = compute_anti_regularization(model)
                    loss -= lambda_antireg * anti_reg
                
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch + 1}/{n_epochs} - Mean Loss: {mean_loss:.4f}")
    
    return models


def evaluate_single_model_regression(model, data_loader, device, use_natural=False):
    """Evaluate single regression model (returns MSE)."""
    model.eval()
    total_mse = 0.0
    total_samples = 0
    
    if use_natural:
        link_fn = natural_link
    else:
        link_fn = variance_link
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            mean, _ = link_fn(output)
            mse = ((mean - targets) ** 2).sum()
            total_mse += mse.item()
            total_samples += targets.size(0)
    
    return total_mse / total_samples if total_samples > 0 else float('inf')


def evaluate_ensemble_regression(models, data_loader, device, use_natural=False):
    """Evaluate ensemble of regression models (returns MSE)."""
    for model in models:
        model.eval()
    
    total_mse = 0.0
    total_samples = 0
    
    if use_natural:
        link_fn = natural_link
    else:
        link_fn = variance_link
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get predictions from all models
            means = []
            for model in models:
                output = model(inputs)
                mean, _ = link_fn(output)
                means.append(mean)
            
            # Average predictions
            ensemble_mean = torch.stack(means, dim=0).mean(dim=0)
            mse = ((ensemble_mean - targets) ** 2).sum()
            total_mse += mse.item()
            total_samples += targets.size(0)
    
    return total_mse / total_samples if total_samples > 0 else float('inf')


# Strategy selection functions (stubs for now)
def select_random_models(models, num_to_select):
    """Random selection strategy."""
    return random.sample(range(len(models)), num_to_select)


def select_accuracy_only_models_regression(models, num_to_select, client_test_loader, device, use_natural=False):
    """Select models with lowest MSE on client test data."""
    mses = []
    for i, model in enumerate(models):
        mse = evaluate_single_model_regression(model, client_test_loader, device, use_natural)
        mses.append((mse, i))
    
    mses.sort(key=lambda x: x[0])
    return [idx for _, idx in mses[:num_to_select]]


def calculate_local_risk_regression(model, data_loader, device, use_natural=False):
    """Calculate local risk (MSE) for a model."""
    model.eval()
    total_mse = 0.0
    num_batches = 0
    
    if use_natural:
        link_fn = natural_link
    else:
        link_fn = variance_link
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            mean, _ = link_fn(output)
            mse = ((mean - targets) ** 2).mean()
            total_mse += mse.item()
            num_batches += 1
    
    return total_mse / num_batches if num_batches > 0 else float('inf')


def calculate_disagreement_regression(model_f, model_g, ood_loader, device, use_natural=False):
    """Calculate disagreement between two regression models using KL divergence between Gaussians."""
    model_f.eval()
    model_g.eval()
    total_disagreement = 0.0
    num_batches = 0
    
    if use_natural:
        link_fn = natural_link
    else:
        link_fn = variance_link
    
    with torch.no_grad():
        for inputs, _ in ood_loader:
            inputs = inputs.to(device)
            
            # Get predictions from both models
            output_f = model_f(inputs)
            mean_f, var_f = link_fn(output_f)
            output_g = model_g(inputs)
            mean_g, var_g = link_fn(output_g)
            
            # KL divergence between two Gaussians: KL(N(μ1,σ1²) || N(μ2,σ2²))
            # = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 0.5
            var_f = var_f.clamp(min=1e-7)
            var_g = var_g.clamp(min=1e-7)
            
            kl_fg = torch.log(var_g / var_f) + (var_f + (mean_f - mean_g) ** 2) / (2 * var_g) - 0.5

            total_disagreement += kl_fg.mean().item()
            num_batches += 1
    
    return total_disagreement / num_batches if num_batches > 0 else 0.0


def select_uncertainty_aware_models_regression(
    models,
    num_to_select,
    client_test_loader,
    ood_loader,
    lambda_disagreement,
    device,
    use_natural=False,
):
    """Uncertainty-aware selection: balance between local risk and disagreement."""
    candidate_indices = list(range(len(models)))
    
    # Calculate local risk for each model
    model_risks = {
        idx: calculate_local_risk_regression(models[idx], client_test_loader, device, use_natural)
        for idx in candidate_indices
    }
    
    selected_indices = []
    
    for step in range(num_to_select):
        best_idx = None
        best_score = float('inf')
        
        if not selected_indices:
            # First model: select one with lowest risk
            best_idx = min(model_risks, key=model_risks.get)
        else:
            # Select model that minimizes: risk - lambda * avg_disagreement
            for k_idx in candidate_indices:
                if k_idx in selected_indices:
                    continue
                
                total_disagreement = 0.0
                for f_idx in selected_indices:
                    total_disagreement += calculate_disagreement_regression(
                        models[f_idx],
                        models[k_idx],
                        ood_loader,
                        device,
                        use_natural,
                    )
                
                avg_disagreement = total_disagreement / len(selected_indices)
                risk = model_risks[k_idx]
                score = risk - lambda_disagreement * avg_disagreement
                
                if score < best_score:
                    best_score = score
                    best_idx = k_idx
        
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)
    
    return selected_indices


def select_market_models_regression(models, num_to_select, client_loader, device, args, use_natural=False):
    """Market-based selection using weight optimization."""
    n_m = len(models)
    
    if use_natural:
        link_fn = natural_link
        loss_fn = NaturalGaussianNLL()
    else:
        link_fn = variance_link
        loss_fn = GaussianNLL()
    
    # Collect all inputs and targets
    all_inputs = []
    all_targets = []
    for inp, tar in client_loader:
        all_inputs.append(inp)
        all_targets.append(tar)
    
    inputs_tensor = torch.cat(all_inputs, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0).to(device)
    n_samples = inputs_tensor.shape[0]
    
    # Use smaller batch size for memory efficiency
    eval_batch_size = min(256, n_samples)
    
    print("    [Market] Pre-computing pairwise disagreement matrix (batched)...")
    
    # Compute disagreement matrix in batches
    disagreement_matrix = torch.zeros(n_m, n_m, device=device)
    total_samples_processed = 0
    
    for batch_start in range(0, n_samples, eval_batch_size):
        batch_end = min(batch_start + eval_batch_size, n_samples)
        batch_inputs = inputs_tensor[batch_start:batch_end].to(device)
        batch_size_actual = batch_end - batch_start
        
        # Get predictions for all models
        batch_means = []
        batch_vars = []
        with torch.no_grad():
            for model in models:
                model.eval()
                output = model(batch_inputs)
                mean, var = link_fn(output)
                batch_means.append(mean)
                batch_vars.append(var)
        
        batch_means = torch.stack(batch_means, dim=0)  # (n_m, batch, 1)
        batch_vars = torch.stack(batch_vars, dim=0)
        
        # Compute pairwise KL divergence
        for i in range(n_m):
            for j in range(i + 1, n_m):
                var_i = batch_vars[i].clamp(min=1e-7)
                var_j = batch_vars[j].clamp(min=1e-7)
                
                kl_ij = torch.log(var_j / var_i) + (var_i + (batch_means[i] - batch_means[j]) ** 2) / (2 * var_j) - 0.5
                kl_ji = torch.log(var_i / var_j) + (var_j + (batch_means[j] - batch_means[i]) ** 2) / (2 * var_i) - 0.5
                
                disagreement = (kl_ij + kl_ji) / 2
                kl_mean = disagreement.mean()
                
                disagreement_matrix[i, j] += kl_mean * batch_size_actual
                disagreement_matrix[j, i] += kl_mean * batch_size_actual
        
        total_samples_processed += batch_size_actual
        del batch_means, batch_vars
    
    # Normalize
    if total_samples_processed > 0:
        disagreement_matrix /= total_samples_processed
    disagreement_matrix.fill_diagonal_(0)
    
    # Initialize weights
    w = torch.ones(n_m, device=device) / n_m
    w = w.detach().requires_grad_(True)
    
    print(f"    [Market] Optimizing weights ({args.market_epochs} steps, batched)...")
    
    # Optimize weights
    for epoch in range(args.market_epochs):
        total_loss_nll = 0.0
        
        # Compute loss in batches
        for batch_start in range(0, n_samples, eval_batch_size):
            batch_end = min(batch_start + eval_batch_size, n_samples)
            batch_inputs = inputs_tensor[batch_start:batch_end].to(device)
            batch_targets = targets_tensor[batch_start:batch_end]
            
            # Get predictions
            batch_means = []
            batch_vars = []
            with torch.no_grad():
                for model in models:
                    model.eval()
                    output = model(batch_inputs)
                    mean, var = link_fn(output)
                    batch_means.append(mean)
                    batch_vars.append(var)
            
            batch_means = torch.stack(batch_means, dim=0)  # (n_m, batch, 1)
            batch_vars = torch.stack(batch_vars, dim=0)
            
            # Weighted ensemble predictions
            ensemble_mean = torch.einsum('n, nbc -> bc', w, batch_means)
            ensemble_var = torch.einsum('n, nbc -> bc', w, batch_vars)
            
            # NLL loss
            batch_loss_nll = loss_fn(ensemble_mean, ensemble_var, batch_targets) * batch_size_actual
            total_loss_nll += batch_loss_nll
        
        loss_nll = total_loss_nll / n_samples
        
        # Diversity term
        w_D = torch.matmul(w, disagreement_matrix)
        diversity = torch.dot(w_D, w)
        
        loss = loss_nll - args.lambda_disagreement * diversity
        
        if w.grad is not None:
            w.grad.zero_()
        loss.backward()
        
        with torch.no_grad():
            w_new = w * torch.exp(-args.market_lr * w.grad)
            w_new /= w_new.sum()
            w.copy_(w_new)
            w.grad.zero_()
    
    weights_np = w.detach().cpu().numpy()
    selected_indices = weights_np.argsort()[-num_to_select:][::-1]
    
    print(f"    [Market] Final Weights Top: {weights_np[selected_indices]}")
    
    return selected_indices.tolist()


def select_greedy_ensemble_accuracy_regression(models, num_to_select, client_loader, device, use_natural=False):
    """Greedy forward selection: at each step add model that minimizes ensemble MSE."""
    selected_indices = []
    pool_indices = list(range(len(models)))
    
    if use_natural:
        link_fn = natural_link
    else:
        link_fn = variance_link
    
    # Get all predictions and targets
    all_means = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in client_loader:
            inputs = inputs.to(device)
            batch_means = []
            
            for model in models:
                model.eval()
                output = model(inputs)
                mean, _ = link_fn(output)
                batch_means.append(mean.cpu())
            
            all_means.append(torch.stack(batch_means, dim=0))  # (n_models, batch, 1)
            all_targets.append(targets.cpu())
    
    means_tensor = torch.cat(all_means, dim=1)  # (n_models, total_samples, 1)
    targets_tensor = torch.cat(all_targets, dim=0)  # (total_samples,)
    
    for _ in range(num_to_select):
        best_idx = -1
        best_mse = float('inf')
        
        for candidate_idx in pool_indices:
            current_indices = selected_indices + [candidate_idx]
            # Average predictions
            ensemble_mean = means_tensor[current_indices].mean(dim=0).squeeze(-1)  # (total_samples,)
            
            mse = ((ensemble_mean - targets_tensor) ** 2).mean().item()
            
            if mse < best_mse:
                best_mse = mse
                best_idx = candidate_idx
        
        if best_idx != -1:
            selected_indices.append(best_idx)
            pool_indices.remove(best_idx)
        else:
            break
    
    return selected_indices


def get_logits_and_labels_regression(models, data_loader, device, use_natural=False):
    """Get predictions (means and variances) from all models."""
    all_means = []
    all_vars = []
    all_labels = []
    
    if use_natural:
        link_fn = natural_link
    else:
        link_fn = variance_link
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            batch_means = []
            batch_vars = []
            
            for model in models:
                model.eval()
                output = model(inputs)
                mean, var = link_fn(output)
                batch_means.append(mean.cpu())
                batch_vars.append(var.cpu())
            
            all_means.append(torch.stack(batch_means, dim=0))  # (n_models, batch, 1)
            all_vars.append(torch.stack(batch_vars, dim=0))
            all_labels.append(labels.cpu())
    
    # Concatenate: (n_models, total_samples, 1)
    means = torch.cat(all_means, dim=1)
    vars = torch.cat(all_vars, dim=1)
    labels = torch.cat(all_labels, dim=0)
    
    return means, vars, labels


def save_predictions_and_labels(selected_models, ind_loader, ood_loader, client_num, strategy, device, run_dir, use_natural=False):
    """Save predictions and labels for later analysis."""
    ind_means, ind_vars, y_ind = get_logits_and_labels_regression(selected_models, ind_loader, device, use_natural)
    ood_means, ood_vars, y_ood = get_logits_and_labels_regression(selected_models, ood_loader, device, use_natural)
    
    # Save predictions and labels
    preds_ind_path = run_dir / "predictions" / f"client_{client_num}" / f"{strategy}_ind.pt"
    preds_ood_path = run_dir / "predictions" / f"client_{client_num}" / f"{strategy}_ood.pt"
    labels_ind_path = run_dir / "labels" / f"client_{client_num}" / f"{strategy}_ind.pt"
    labels_ood_path = run_dir / "labels" / f"client_{client_num}" / f"{strategy}_ood.pt"
    
    preds_ind_path.parent.mkdir(parents=True, exist_ok=True)
    labels_ind_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Stack means and vars: (n_models, n_samples, 2) where last dim is [mean, var]
    ind_preds = torch.stack([ind_means.squeeze(-1), ind_vars.squeeze(-1)], dim=-1)
    ood_preds = torch.stack([ood_means.squeeze(-1), ood_vars.squeeze(-1)], dim=-1)
    
    torch.save(ind_preds, preds_ind_path)
    torch.save(ood_preds, preds_ood_path)
    torch.save(y_ind, labels_ind_path)
    torch.save(y_ood, labels_ood_path)
    
    print(f"Saved client {client_num} predictions and labels to: {run_dir}")


def select_and_evaluate_models_regression(
    strategy,
    ensemble,
    spoilers,
    client_ind_train_loader,
    client_ood_test_loader,
    client_ind_test_loader,
    client_num,
    device,
    use_natural,
    run_dir,
    ensemble_size,
    lambda_disagreement=0.1,
):
    """Select models using specified strategy and evaluate."""
    num_to_select = min(ensemble_size, len(spoilers))
    
    if strategy == "random":
        print("  --- Strategy: Random Selection ---")
        ensemble_indices = select_random_models(spoilers, num_to_select)
    elif strategy == "accuracy":
        print("\n  --- Strategy: Accuracy-only Selection ---")
        ensemble_indices = select_accuracy_only_models_regression(
            spoilers, num_to_select, client_ind_test_loader, device, use_natural
        )
    elif strategy == "uncertainty":
        print(f"\n  --- Strategy: Uncertainty-Aware ---")
        ensemble_indices = select_uncertainty_aware_models_regression(
            spoilers,
            num_to_select,
            client_ind_test_loader,
            client_ood_test_loader,
            lambda_disagreement,
            device,
            use_natural,
        )
    elif strategy == "market":
        print(f"\n  --- Strategy: Market (selection by weights) ---")
        # Create a minimal args object for market selection
        class Args:
            pass
        args = Args()
        args.market_epochs = 2
        args.market_lr = 1.0
        args.lambda_disagreement = 0.1
        ensemble_indices = select_market_models_regression(
            spoilers, num_to_select, client_ind_train_loader, device, args, use_natural
        )
    elif strategy == "greedy_accuracy":
        print("\n  --- Strategy: Greedy Ensemble Accuracy ---")
        ensemble_indices = select_greedy_ensemble_accuracy_regression(
            spoilers, num_to_select, client_ind_train_loader, device, use_natural
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'.")
    
    selected_ensemble = [ensemble[i] for i in ensemble_indices]
    print(f"  -> Selected models: {ensemble_indices}")
    
    ensemble_mse = evaluate_ensemble_regression(selected_ensemble, client_ind_test_loader, device, use_natural)
    for i, model in enumerate(selected_ensemble):
        model_mse = evaluate_single_model_regression(model, client_ind_test_loader, device, use_natural)
        print(f"  -> Model[{i + 1}] MSE: {model_mse:.6f}")
    print(f"    -> Ensemble MSE: {ensemble_mse:.6f}")
    
    save_predictions_and_labels(
        selected_ensemble, client_ind_test_loader, client_ood_test_loader, client_num, strategy, device, run_dir, use_natural
    )
    
    return selected_ensemble, ensemble_indices


def main():
    parser = argparse.ArgumentParser(description='PyTorch FEDERATED UNCERTAINTY Training for Regression')
    
    # Hyperparameters from main_federated_ensembles.py
    parser.add_argument('--n_models', default=20, type=int, help='number of models')
    parser.add_argument('--n_clients', default=5, type=int, help='number of clients')
    parser.add_argument('--ensemble_size', default=3, type=int, help='number of models for single client')
    parser.add_argument('--lambda_disagreement', default=0.1, type=float, help='disagreement importance')
    parser.add_argument('--lambda_antireg', default=0.01, type=float, help='antiregularization coefficient')
    parser.add_argument('--fraction', default=0.25, type=float, help='client and model part of train data')
    parser.add_argument('--n_epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for models')
    parser.add_argument('--model_pool_split_ratio', default=0.6, type=float, help='model/client data split ratio')
    parser.add_argument('--model_min_digits', default=5, type=int, help='min digits for model pool')
    parser.add_argument('--model_max_digits', default=8, type=int, help='max digits for model pool')
    parser.add_argument('--client_min_digits', default=2, type=int, help='min digits for clients')
    parser.add_argument('--client_max_digits', default=5, type=int, help='max digits for clients')
    parser.add_argument('--market_lr', default=1.0, type=float, help='learning rate for mirror descent')
    parser.add_argument('--market_epochs', default=2, type=int, help='optimization steps for market weighting')
    parser.add_argument('--save_dir', 
                        default=f"./data/saved_models/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                        type=str, help='Path to save/load ensemble models')
    parser.add_argument('--seed', default=0, type=int, help='seed for random number generator')
    
    # Regression-specific arguments
    parser.add_argument("--network", type=str, default="cnn", choices=["cnn", "resnet"],
                        help="Network architecture: cnn or resnet")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--tile_size", type=int, default=32,
                        help="Size of each tile in the mosaic")
    parser.add_argument("--n_id_train", type=int, default=100000,
                        help="Number of training samples")
    parser.add_argument("--n_id_test", type=int, default=1000,
                        help="Number of test samples")
    parser.add_argument("--use_natural", action="store_true",
                        help="Use natural parametrization")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of data loading workers")
    
    args = parser.parse_args()
    
    seed = args.seed
    fix_seeds(seed)
    
    # Use CUDA if available
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # Extract arguments
    n_models = args.n_models
    n_clients = args.n_clients
    ensemble_size = args.ensemble_size
    lambda_disagreement = args.lambda_disagreement
    lambda_antireg = args.lambda_antireg
    fraction = args.fraction
    n_epochs = args.n_epochs
    lr = args.lr
    batch_size = args.batch_size
    model_pool_split_ratio = args.model_pool_split_ratio
    model_min_digits = args.model_min_digits
    model_max_digits = args.model_max_digits
    client_min_digits = args.client_min_digits
    client_max_digits = args.client_max_digits
    save_dir = args.save_dir
    
    # All possible digits (0-9)
    all_digits = list(range(10))
    
    print('==> Loading base mosaic dataset...')
    id_train_full, id_test_full, _, _, _, _ = build_id_and_ood(
        root=args.data_root,
        tile_size=args.tile_size,
        seed=seed,
        n_id_train=args.n_id_train,
        n_id_test=args.n_id_test,
        n_ood_each=1000,
        download=True,
        normalize_images=True,
    )
    
    # Create validation set
    _, id_val_full, _, _, _, _ = build_id_and_ood(
        root=args.data_root,
        tile_size=args.tile_size,
        seed=seed + 1,
        n_id_train=args.n_id_train,
        n_id_test=args.n_id_test,
        n_ood_each=1000,
        download=True,
        normalize_images=True,
    )
    
    print(f"  Total train data: {len(id_train_full)}")
    print(f"  Total test data: {len(id_test_full)}")
    
    print('==> Splitting trainset into model_pool and client_data (non-overlapping)...')
    
    # Split by sample indices to avoid leakage between model_pool and client_data
    all_train_indices = list(range(len(id_train_full)))
    random.shuffle(all_train_indices)
    split_point = int(len(all_train_indices) * model_pool_split_ratio)
    model_pool_indices = set(all_train_indices[:split_point])
    client_data_indices = set(all_train_indices[split_point:])
    
    # Build digit-based indices for each split (used later for sampling)
    model_pool_digit_indices = {d: [] for d in all_digits}
    client_data_digit_indices = {d: [] for d in all_digits}
    
    for i in model_pool_indices:
        tl, tr, bl, br = id_train_full.quads[i]
        digits_in_sample = set([
            id_train_full.labels[tl],
            id_train_full.labels[tr],
            id_train_full.labels[bl],
            id_train_full.labels[br]
        ])
        for d in digits_in_sample:
            model_pool_digit_indices[d].append(i)
    
    for i in client_data_indices:
        tl, tr, bl, br = id_train_full.quads[i]
        digits_in_sample = set([
            id_train_full.labels[tl],
            id_train_full.labels[tr],
            id_train_full.labels[bl],
            id_train_full.labels[br]
        ])
        for d in digits_in_sample:
            client_data_digit_indices[d].append(i)
    
    print(f"  Model pool data size: {len(model_pool_indices)} ({(len(model_pool_indices)/len(id_train_full)*100):.2f}%)")
    print(f"  Client data size: {len(client_data_indices)} ({(len(client_data_indices)/len(id_train_full)*100):.2f}%)")
    
    def sample_indices_by_digits(selected_digits, digit_indices_dict, total_samples):
        """Sample indices for samples containing selected digits."""
        # Get all indices that contain at least one of the selected digits
        candidate_indices = set()
        for d in selected_digits:
            candidate_indices.update(digit_indices_dict[d])
        
        candidate_indices = list(candidate_indices)
        random.shuffle(candidate_indices)
        
        # Sample up to total_samples
        n_samples = min(total_samples, len(candidate_indices))
        return random.sample(candidate_indices, n_samples) if n_samples > 0 else []

    def sample_indices_all_digits(dataset, selected_digits, candidate_indices, total_samples):
        """
        Sample indices where ALL digits in the mosaic are from selected_digits.
        candidate_indices should be a set/list of eligible indices to consider.
        """
        allowed_digits = set(selected_digits)
        valid_indices = []
        for i in candidate_indices:
            tl, tr, bl, br = dataset.quads[i]
            digits_in_sample = [
                dataset.labels[tl],
                dataset.labels[tr],
                dataset.labels[bl],
                dataset.labels[br],
            ]
            if all(d in allowed_digits for d in digits_in_sample):
                valid_indices.append(i)

        random.shuffle(valid_indices)
        n_samples = min(total_samples, len(valid_indices))
        return valid_indices[:n_samples]
    
    samples_per_model = int(len(id_train_full) * fraction)
    samples_per_client = int(len(id_train_full) * fraction)
    
    print(f"\n==> Generating {n_models} datasets for model training...")
    
    model_train_loaders = []
    model_digit_subsets = []
    
    for i in range(n_models):
        model_max_digits = min(model_max_digits, 10)
        model_min_digits = min(model_min_digits, model_max_digits)
        
        n_model_digits = random.randint(model_min_digits, model_max_digits)
        selected_digits = random.sample(all_digits, n_model_digits)
        model_digit_subsets.append(selected_digits)
        
        # Sample indices from model_pool
        train_indices = sample_indices_by_digits(selected_digits, model_pool_digit_indices, samples_per_model)
        
        if len(train_indices) > 0:
            train_subset = Subset(id_train_full, train_indices)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
            model_train_loaders.append(train_loader)
            print(f"  Model {i + 1} will be trained on digits {selected_digits} with {len(train_indices)} samples.")
        else:
            print(f"  Warning: Model {i + 1} has no samples for digits {selected_digits}")
            # Create empty loader
            model_train_loaders.append(DataLoader(Subset(id_train_full, []), batch_size=batch_size))
    
    print(f"\n==> Generating {n_clients} datasets for client evaluation...")
    
    client_digit_subsets = []
    client_ind_train_loaders = []
    client_ood_test_loaders = []
    client_ind_test_loaders = []
    
    for i in range(n_clients):
        n_client_digits = random.randint(client_min_digits, client_max_digits)
        selected_digits = random.sample(all_digits, n_client_digits)
        client_digit_subsets.append(selected_digits)
        
        # Client IND train data - only mosaics composed exclusively of client's digits
        candidate_indices = set()
        for d in selected_digits:
            candidate_indices.update(client_data_digit_indices[d])
        train_ind_indices = sample_indices_all_digits(
            id_train_full, selected_digits, candidate_indices, samples_per_client
        )
        train_size = len(train_ind_indices)
        if train_size > 0:
            train_ind_subset = Subset(id_train_full, train_ind_indices)
            train_ind_loader = DataLoader(train_ind_subset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
            client_ind_train_loaders.append(train_ind_loader)
        else:
            client_ind_train_loaders.append(DataLoader(Subset(id_train_full, []), batch_size=batch_size))
        
        # Client IND test data
        filtered_test = FilteredMosaicDataset(id_test_full, selected_digits)
        test_indices = list(range(min(len(filtered_test), args.n_id_test)))
        test_subset = Subset(filtered_test, test_indices)
        test_size = len(test_subset)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
        client_ind_test_loaders.append(test_loader)
        
        # Client OOD test data (digits not in client's subset)
        ood_digits = [d for d in all_digits if d not in selected_digits]
        ood_size = 0
        if len(ood_digits) > 0:
            filtered_ood = FilteredMosaicDataset(id_test_full, ood_digits)
            ood_indices = list(range(min(len(filtered_ood), args.n_id_test)))
            ood_subset = Subset(filtered_ood, ood_indices)
            ood_size = len(ood_subset)
            ood_loader = DataLoader(ood_subset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
            client_ood_test_loaders.append(ood_loader)
        else:
            # No OOD digits available, use empty loader
            print(f"  Warning: Model {i + 1} has no OoD TEST samples for digits {selected_digits}")
            client_ood_test_loaders.append(DataLoader(Subset(filtered_test, []), batch_size=batch_size))
        
        print(f"  Client {i + 1} has data for digits {selected_digits}: "
              f"{train_size} train samples, {test_size} ind test samples, {ood_size} ood test samples.")
    
    print('\n==> Preparing ensemble path and checking for existing models...')
    
    run_dir = Path(save_dir)
    model_file_path = run_dir / 'ensemble.pt'
    
    model_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save hyperparameters to YAML config
    config_path = run_dir / 'config.yaml'
    config_dict = vars(args)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    print(f"  Hyperparameters saved to {config_path}.")
    
    # Model configuration
    model_kwargs = {
        'in_channels': 1,
        'image_size': 2 * args.tile_size,
    }
    
    # Create ensemble
    if args.network == "cnn":
        ensemble = [get_cnn(**model_kwargs).to(device) for _ in range(n_models)]
    elif args.network == "resnet":
        ensemble = [get_resnet18(**model_kwargs).to(device) for _ in range(n_models)]
    else:
        raise ValueError(f"Unknown network: {args.network}")
    
    ensemble_state_dicts = None
    
    if model_file_path.exists():
        try:
            ensemble_state_dicts = torch.load(model_file_path, map_location=device)
            print(f"  Ensemble loaded from {model_file_path}.")
        except Exception as e:
            print(f"  Warning: Could not load models from {model_file_path}. Error: {e}")
            ensemble_state_dicts = None
    
    if ensemble_state_dicts is not None:
        if len(ensemble_state_dicts) == n_models:
            for model, state_dict in zip(ensemble, ensemble_state_dicts):
                model.load_state_dict(state_dict)
        else:
            print(f"  Warning: Expected {n_models} models, but loaded {len(ensemble_state_dicts)}. Retraining.")
            ensemble_state_dicts = None
    
    if ensemble_state_dicts is None:
        print("  Training ensemble from scratch...")
        
        # Create criterion for regression (we'll use GaussianNLL internally)
        ensemble = train_ensembles_w_local_data_regression(
            ensemble,
            model_train_loaders,
            device,
            n_epochs,
            lambda_antireg,
            lr,
            args.use_natural,
        )
        
        state_dicts_to_save = [model.state_dict() for model in ensemble]
        torch.save(state_dicts_to_save, model_file_path)
        print(f"  Ensemble successfully trained and saved to {model_file_path}.")
    
    if ensemble_size > n_models:
        raise ValueError(f"ensemble_size ({ensemble_size}) cannot be larger than n_models ({n_models})")
    
    # For now, spoilers are just copies of the ensemble (no noise added)
    # TODO: Add noise if needed
    spoilers = [copy.deepcopy(model) for model in ensemble]
    
    print("\nMODEL SELECTION STRATEGIES COMPARISON")
    print(f"The model pool (`ensemble`) consists of {n_models} models.")
    print(f"For each of the {n_clients} clients, an ensemble of {ensemble_size} models will be selected.")
    
    # Final selection loop
    for i in range(n_clients):
        print(f"\n[Client {i + 1}] (Digits: {client_digit_subsets[i]})")
        
        selected_ensemble_random, indices_random = select_and_evaluate_models_regression(
            "random",
            ensemble,
            spoilers,
            client_ind_train_loaders[i],
            client_ood_test_loaders[i],
            client_ind_test_loaders[i],
            i + 1,
            device,
            args.use_natural,
            run_dir,
            ensemble_size,
            lambda_disagreement,
        )
        
        selected_ensemble_acc, indices_acc = select_and_evaluate_models_regression(
            "accuracy",
            ensemble,
            spoilers,
            client_ind_train_loaders[i],
            client_ood_test_loaders[i],
            client_ind_test_loaders[i],
            i + 1,
            device,
            args.use_natural,
            run_dir,
            ensemble_size,
            lambda_disagreement,
        )
        
        selected_ensemble_unc, indices_unc = select_and_evaluate_models_regression(
            "uncertainty",
            ensemble,
            spoilers,
            client_ind_train_loaders[i],
            client_ood_test_loaders[i],
            client_ind_test_loaders[i],
            i + 1,
            device,
            args.use_natural,
            run_dir,
            ensemble_size,
            lambda_disagreement,
        )
        
        # selected_ensemble_mkt, indices_mkt = select_and_evaluate_models_regression(
        #     "market",
        #     ensemble,
        #     spoilers,
        #     client_ind_train_loaders[i],
        #     client_ood_test_loaders[i],
        #     client_ind_test_loaders[i],
        #     i + 1,
        #     device,
        #     args.use_natural,
        #     run_dir,
        #     ensemble_size,
        #     lambda_disagreement,
        # )
        
        selected_ensemble_greedy, indices_greedy = select_and_evaluate_models_regression(
            "greedy_accuracy",
            ensemble,
            spoilers,
            client_ind_train_loaders[i],
            client_ood_test_loaders[i],
            client_ind_test_loaders[i],
            i + 1,
            device,
            args.use_natural,
            run_dir,
            ensemble_size,
            lambda_disagreement,
        )
    
    print("\nTraining and evaluation completed!")


if __name__ == "__main__":
    main()
