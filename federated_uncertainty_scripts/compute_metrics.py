import sys
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import pandas as pd

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated_uncertainty.unc.risk_metrics import *
from federated_uncertainty.unc.calibration_metrics import *
from federated_uncertainty.randomness import set_all_seeds

set_all_seeds(0)

parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', default=5, type=int, help='number of clients')
parser.add_argument("--data_path", type=str, required=True)

args = parser.parse_args()
data_path = args.data_path
n_clients = args.n_clients

def compute_metrics(ind_logits, ood_logits, ind_labels, weights=None):
    """
    Computes AUC for multiple uncertainty metrics
    
    ind_logits: np.array (n_models, N_ind, n_classes)
    ood_logits: np.array (n_models, N_ood, n_classes)
    ind_labels: np.array (N_ind)
    weights: np.array (n_models,) or None - optional weights for weighted ensemble
    """
    
    # --- 1. Prepare labels for ROC-AUC ---
    # 0 = In-Distribution, 1 = Out-of-Distribution
    y_true = np.concatenate([
        np.zeros(ind_logits.shape[1]), # N_ind
        np.ones(ood_logits.shape[1])   # N_ood
    ])
    
    metrics_results = {}

    # --- METRIC 1: LogScore ---
    logscore = get_risk_approximation(
        GName.LOG_SCORE,
        RiskType.EXCESS_RISK,
        np.concatenate([ind_logits, ood_logits], axis=1),
        ApproximationType.OUTER,
        pred_approx=ApproximationType.OUTER,
        weights=weights
    )
    metrics_results['LogScore'] = roc_auc_score(y_true, logscore)

    # --- METRIC 2: Brier ---
    brier = get_risk_approximation(
        GName.BRIER_SCORE,
        RiskType.EXCESS_RISK,
        np.concatenate([ind_logits, ood_logits], axis=1),
        ApproximationType.OUTER,
        pred_approx=ApproximationType.OUTER,
        weights=weights
    )
    metrics_results['Brier'] = roc_auc_score(y_true, brier)

    # --- METRIC 3: Spherical ---
    spherical = get_risk_approximation(
        GName.SPHERICAL_SCORE,
        RiskType.EXCESS_RISK,
        np.concatenate([ind_logits, ood_logits], axis=1),
        ApproximationType.OUTER,
        pred_approx=ApproximationType.OUTER,
        weights=weights
    )
    metrics_results['Spherical'] = roc_auc_score(y_true, spherical)

    ind_logits_tensor = torch.from_numpy(ind_logits)
    ind_probs_tensor = F.softmax(ind_logits_tensor, dim=2)
    ind_probs = ind_probs_tensor.numpy()
    
    if weights is not None:
        w_expanded = weights[:, None, None]
        avg_ind_probs = np.sum(ind_probs * w_expanded, axis=0)
    else:
        avg_ind_probs = np.mean(ind_probs, axis=0)

    # --- METRIC 4: ECE ---
    ece = get_metric("ece")
    metrics_results['ECE'] = ece(probs=avg_ind_probs, y_true=ind_labels)

    # --- METRIC 5: MCE ---
    mce = get_metric("mce")
    metrics_results['MCE'] = mce(probs=avg_ind_probs, y_true=ind_labels)

    # --- METRIC 6: CW-ECE ---
    cw_ece = get_metric("cw-ece")
    metrics_results['CW-ECE'] = cw_ece(probs=avg_ind_probs, y_true=ind_labels)

    # For Accuracy: average logits first
    if weights is not None:
        w_expanded = weights[:, None, None]
        avg_ind_logits = np.sum(ind_logits * w_expanded, axis=0)
    else:
        avg_ind_logits = np.mean(ind_logits, axis=0)
    
    # --- METRIC 7: Accuracy (using averaged logits, same as main script) ---
    predictions = np.argmax(avg_ind_logits, axis=1)
    accuracy = np.mean(predictions == ind_labels)
    metrics_results['Accuracy'] = accuracy

    return metrics_results

print(f"--- Starting Evaluation ---")
print(f"{'Client':<8} | {'Strategy':<12} | {'LogScore':<10} | {'Brier':<10} | {'Spherical':<10} | {'ECE':<10} | {'MCE':<10} | {'CW-ECE':<10} | {'Accuracy':<10}")
print("-" * 110)

strategies = [
    "random", 
    "accuracy", 
    "greedy_accuracy",
    "uncertainty", 
    "market",
]
metrics_names = ["LogScore", "Brier", "Spherical", "ECE", "MCE", "CW-ECE", "Accuracy"]

# Collect all results for xlsx export
results_dict = {}

for strategy in strategies:
    for client_id in range(1, n_clients + 1):
        ind_logits_path = f"{data_path}/logits/client_{client_id}/{strategy}_ind.pt"
        ood_logits_path = f"{data_path}/logits/client_{client_id}/{strategy}_ood.pt"
        ind_labels_path = f"{data_path}/labels/client_{client_id}/{strategy}_ind.pt"
        
        all_ind_logits = torch.load(ind_logits_path).numpy()
        all_ood_logits = torch.load(ood_logits_path).numpy()
        all_ind_labels = torch.load(ind_labels_path).numpy()
        
        weights = None
        if strategy == "hybrid":
            weights_path = f"{data_path}/weights/client_{client_id}/{strategy}_weights.pt"
            try:
                weights = torch.load(weights_path)
                if isinstance(weights, torch.Tensor):
                    weights = weights.detach().cpu().numpy()     
            except FileNotFoundError:
                pass
        
        results = compute_metrics(all_ind_logits, all_ood_logits, all_ind_labels, weights=weights)
        
        # Store results for xlsx
        if client_id not in results_dict:
            results_dict[client_id] = {}
        for metric_name in metrics_names:
            col_name = f"{metric_name}_{strategy}"
            results_dict[client_id][col_name] = results[metric_name]
        
        print(f"{client_id:02d}       | {strategy:<12} | {results['LogScore']:.4f}     | {results['Brier']:.4f}     | {results['Spherical']:.4f}     | {results['ECE']:.4f}     | {results['MCE']:.4f}     | {results['CW-ECE']:.4f}     | {results['Accuracy']:.4f}")
    print("-" * 110)

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
output_xlsx_path = Path(data_path) / "metrics_results.xlsx"
try:
    df.to_excel(output_xlsx_path, engine='openpyxl')
    print(f"\n--- Results saved to {output_xlsx_path} ---")
except ImportError:
    print(f"\n--- Warning: openpyxl is not installed. Cannot save to xlsx format. ---")
    print(f"--- Install openpyxl with: pip install openpyxl ---")
except Exception as e:
    print(f"\n--- Error saving to xlsx: {e} ---")
