import sys
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated_uncertainty.unc.risk_metrics import *
from federated_uncertainty.unc.calibration_metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', default=5, type=int, help='number of clients')
parser.add_argument("--data_path", type=str, required=True)

args = parser.parse_args()
data_path = args.data_path
n_clients = args.n_clients

def compute_ood_metrics(ind_logits, ood_logits, ind_labels):
    """
    Computes AUC for multiple uncertainty metrics
    
    ind_logits: np.array (n_models, N_ind, n_classes)
    ood_logits: np.array (n_models, N_ood, n_classes)
    ind_labels: np.array (N_ind)
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
        ApproximationType.CENTRAL,
        pred_approx=ApproximationType.OUTER
    )
    metrics_results['LogScore'] = roc_auc_score(y_true, logscore)

    # --- METRIC 2: Brier ---
    brier = get_risk_approximation(
        GName.BRIER_SCORE,
        RiskType.EXCESS_RISK,
        np.concatenate([ind_logits, ood_logits], axis=1),
        ApproximationType.CENTRAL,
        pred_approx=ApproximationType.OUTER
    )
    metrics_results['Brier'] = roc_auc_score(y_true, brier)

    # --- METRIC 3: Spherical ---
    spherical = get_risk_approximation(
        GName.SPHERICAL_SCORE,
        RiskType.EXCESS_RISK,
        np.concatenate([ind_logits, ood_logits], axis=1),
        ApproximationType.CENTRAL,
        pred_approx=ApproximationType.OUTER
    )
    metrics_results['Spherical'] = roc_auc_score(y_true, spherical)

    ind_logits_tensor = torch.from_numpy(ind_logits)
    ind_probs_tensor = F.softmax(ind_logits_tensor, dim=2)
    ind_probs = ind_probs_tensor.numpy()
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

    return metrics_results

print(f"--- Starting OOD Uncertainty Evaluation ---")
print(f"{'Client':<8} | {'Strategy':<12} | {'LogScore':<10} | {'Brier':<10} | {'Spherical':<10} | {'ECE':<10} | {'MCE':<10} | {'CW-ECE':<10}")
print("-" * 100)

strategies = ["random", "accuracy", "uncertainty"]

for strategy in strategies:
    for client_id in range(1, n_clients + 1):
        ind_logits_path = f"{data_path}/logits/client_{client_id}/{strategy}_ind.pt"
        ood_logits_path = f"{data_path}/logits/client_{client_id}/{strategy}_ood.pt"
        ind_labels_path = f"{data_path}/labels/client_{client_id}/{strategy}_ind.pt"
        
        all_ind_logits = torch.load(ind_logits_path).numpy()
        all_ood_logits = torch.load(ood_logits_path).numpy()
        all_ind_labels = torch.load(ind_labels_path).numpy()
        
        results = compute_ood_metrics(all_ind_logits, all_ood_logits, all_ind_labels)
        
        print(f"{client_id:02d}       | {strategy:<12} | {results['LogScore']:.4f}     | {results['Brier']:.4f}     | {results['Spherical']:.4f}     | {results['ECE']:.4f}     | {results['MCE']:.4f}     | {results['CW-ECE']:.4f}")
    print("-" * 100)

print(f"--- Calculation Complete ---")