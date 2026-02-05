import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# project root = parent of "federated_uncertainty_scripts"
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated_uncertainty.regression_uncertainty.source.utils.uncertainty_measures import (
    calculate_uncertainties_crps,
    calculate_uncertainties_log,
    calculate_uncertainties_quadratic,
    calculate_uncertainties_mse,
)
from federated_uncertainty.randomness import set_all_seeds

set_all_seeds(0)


def _load_predictions(preds_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns means and variances with shape (n_samples, n_models).
    """
    preds = torch.load(preds_path)
    means = preds[..., 0].transpose(0, 1).contiguous()
    variances = preds[..., 1].transpose(0, 1).contiguous()
    return means, variances


def compute_metrics(ind_means, ind_variances, ood_means, ood_variances, ind_labels, weights=None):
    """
    Computes AUC for multiple uncertainty metrics
    
    ind_means: torch.Tensor (N_ind, n_models)
    ind_variances: torch.Tensor (N_ind, n_models)
    ood_means: torch.Tensor (N_ood, n_models)
    ood_variances: torch.Tensor (N_ood, n_models)
    ind_labels: torch.Tensor (N_ind)
    weights: torch.Tensor (n_models,) or None - optional weights for weighted ensemble
    """
    
    # --- 1. Prepare labels for ROC-AUC ---
    # 0 = In-Distribution, 1 = Out-of-Distribution
    y_true = np.concatenate([
        np.zeros(ind_means.shape[0]),  # N_ind
        np.ones(ood_means.shape[0])    # N_ood
    ])
    
    # --- 2. Concatenate ind and ood data ---
    all_means = torch.cat([ind_means, ood_means], dim=0)
    all_variances = torch.cat([ind_variances, ood_variances], dim=0)
    
    # Clamp variances to avoid numerical issues
    all_variances = all_variances.clamp_min(1e-9)
    
    metrics_results = {}
    
    # --- METRIC 1: CRPS ---
    crps_results = calculate_uncertainties_crps(all_means, all_variances)
    crps = crps_results["excess_1_1"].numpy()
    metrics_results['CRPS'] = roc_auc_score(y_true, crps)
    
    # --- METRIC 2: Log ---
    log_results = calculate_uncertainties_log(all_means, all_variances)
    log_score = log_results["excess_1_1"].numpy()
    metrics_results['Log'] = roc_auc_score(y_true, log_score)
    
    # --- METRIC 3: Quadratic ---
    quad_results = calculate_uncertainties_quadratic(all_means, all_variances)
    quad_score = quad_results["excess_1_1"].numpy()
    metrics_results['Quadratic'] = roc_auc_score(y_true, quad_score)
    
    # --- METRIC 4: Squared Error (MSE) ---
    se_results = calculate_uncertainties_mse(all_means, all_variances)
    se_score = se_results["excess_1_1"].numpy()
    metrics_results['SquaredError'] = roc_auc_score(y_true, se_score)
    
    # --- METRIC 5: MSE (using ensemble mean) ---
    if weights is not None:
        w_expanded = weights[None, :]
        ensemble_mean = (ind_means * w_expanded).sum(dim=-1)
    else:
        ensemble_mean = ind_means.mean(dim=-1)
    
    mse_value = torch.mean((ensemble_mean - ind_labels) ** 2).item()
    metrics_results['MSE'] = mse_value
    
    return metrics_results


def _get_default_strategies(data_path: Path, n_clients: int) -> list[str]:
    client_dir = data_path / "predictions" / "client_1"
    if not client_dir.exists():
        return ["random", "accuracy", "greedy_accuracy", "uncertainty", "market"]

    strategies = set()
    for path in client_dir.glob("*_ind.pt"):
        strategies.add(path.name.replace("_ind.pt", ""))
    return sorted(strategies)


parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', default=5, type=int, help='number of clients')
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument(
    "--strategies",
    type=str,
    default=None,
    help="comma-separated list of strategies (defaults to inferred)",
)

args = parser.parse_args()
data_path = Path(args.data_path)
n_clients = args.n_clients

if args.strategies:
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
else:
    strategies = _get_default_strategies(data_path, n_clients)

metrics_names = ["MSE", "CRPS", "Log", "Quadratic", "SquaredError"]

print(f"--- Starting Regression Metrics ---")
print(f"{'Client':<8} | {'Strategy':<16} | {'MSE':<10} | {'CRPS':<10} | {'Log':<10} | {'Quadratic':<10} | {'SquaredError':<12}")
print("-" * 100)

# Collect all results for xlsx export
results_dict = {}

for strategy in strategies:
    for client_id in range(1, n_clients + 1):
        ind_preds_path = data_path / "predictions" / f"client_{client_id}" / f"{strategy}_ind.pt"
        ood_preds_path = data_path / "predictions" / f"client_{client_id}" / f"{strategy}_ood.pt"
        ind_labels_path = data_path / "labels" / f"client_{client_id}" / f"{strategy}_ind.pt"
        
        if not ind_preds_path.exists() or not ood_preds_path.exists() or not ind_labels_path.exists():
            print(
                f"{client_id:02d}       | {strategy:<16} | {'missing':<10} | "
                f"{'-':<10} | {'-':<10} | {'-':<10} | {'-':<12}"
            )
            continue
        
        ind_means, ind_variances = _load_predictions(ind_preds_path)
        ood_means, ood_variances = _load_predictions(ood_preds_path)
        ind_labels = torch.load(ind_labels_path)
        
        weights = None
        if strategy == "hybrid":
            weights_path = data_path / "weights" / f"client_{client_id}" / f"{strategy}_weights.pt"
            try:
                weights = torch.load(weights_path)
                if isinstance(weights, torch.Tensor):
                    weights = weights.detach().cpu()
            except FileNotFoundError:
                pass
        
        results = compute_metrics(ind_means, ind_variances, ood_means, ood_variances, ind_labels, weights=weights)
        
        # Store results for xlsx
        if client_id not in results_dict:
            results_dict[client_id] = {}
        for metric_name in metrics_names:
            col_name = f"{metric_name}_{strategy}"
            results_dict[client_id][col_name] = results[metric_name]
        
        print(f"{client_id:02d}       | {strategy:<16} | {results['MSE']:<10.6f} | {results['CRPS']:<10.6f} | {results['Log']:<10.6f} | {results['Quadratic']:<10.6f} | {results['SquaredError']:<12.6f}")
    print("-" * 100)

print(f"--- Calculation Complete ---")

# Create DataFrame for xlsx export
# Columns: Metric1_random, Metric1_accuracy, Metric1_uncertainty, Metric1_hybrid, Metric2_random, ...
column_order = []
for metric_name in metrics_names:
    for strategy in strategies:
        column_order.append(f"{metric_name}_{strategy}")

# Create DataFrame with clients as index
df_data = []
client_ids = []
for client_id in sorted(results_dict.keys()):
    client_ids.append(client_id)
    row_data = [results_dict[client_id].get(col, np.nan) for col in column_order]
    df_data.append(row_data)

df = pd.DataFrame(df_data, index=client_ids, columns=column_order)
df.index.name = "Client"

# Save to xlsx
output_xlsx_path = data_path / "regression_metrics_results.xlsx"
try:
    df.to_excel(output_xlsx_path, engine='openpyxl')
    print(f"\n--- Results saved to {output_xlsx_path} ---")
except ImportError:
    print(f"\n--- Warning: openpyxl is not installed. Cannot save to xlsx format. ---")
    print(f"--- Install openpyxl with: pip install openpyxl ---")
except Exception as e:
    print(f"\n--- Error saving to xlsx: {e} ---")
