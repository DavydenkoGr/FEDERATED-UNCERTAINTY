import sys
import argparse
from pathlib import Path

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

from federated_uncertainty.unc_from_scoring_rules.uncertainty_scores import mv_logscore, mv_brier, mv_spherical

parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', default=5, type=int, help='number of clients')
parser.add_argument("--data_path", type=str, required=True)

args = parser.parse_args()
data_path = args.data_path
n_clients = args.n_clients

def compute_ood_metrics(ind_logits, ood_logits):
    """
    Computes AUC for multiple uncertainty metrics
    
    ind_logits: np.array (n_models, N_ind, n_classes)
    ood_logits: np.array (n_models, N_ood, n_classes)
    """
    
    # --- 1. Prepare labels for ROC-AUC ---
    # 0 = In-Distribution, 1 = Out-of-Distribution
    y_true = np.concatenate([
        np.zeros(ind_logits.shape[1]), # N_ind
        np.ones(ood_logits.shape[1])   # N_ood
    ])
    
    metrics_results = {}

    # --- METRIC 1: MSP (Baseline) ---
    ind_logits_mean = ind_logits.mean(axis=0)
    ood_logits_mean = ood_logits.mean(axis=0)

    ind_probs = F.softmax(torch.tensor(ind_logits_mean), dim=1).numpy()
    ood_probs = F.softmax(torch.tensor(ood_logits_mean), dim=1).numpy()

    ind_msp_score = 1 - ind_probs.max(axis=1)
    ood_msp_score = 1 - ood_probs.max(axis=1)
    
    y_scores_msp = np.concatenate([ind_msp_score, ood_msp_score])
    metrics_results['MSP'] = roc_auc_score(y_true, y_scores_msp)

    # --- METRIC 2: Model Variance (LogScore) ---
    ind_mv_log = mv_logscore(logits_pred=ind_logits, logits_gt=ind_logits)
    ood_mv_log = mv_logscore(logits_pred=ood_logits, logits_gt=ood_logits)

    y_scores_mv_log = np.concatenate([ind_mv_log, ood_mv_log])
    metrics_results['MV_LogScore'] = roc_auc_score(y_true, y_scores_mv_log)

    # --- METRIC 3: Model Variance (Brier) ---
    ind_mv_brier = mv_brier(logits_pred=ind_logits, logits_gt=ind_logits)
    ood_mv_brier = mv_brier(logits_pred=ood_logits, logits_gt=ood_logits)
    
    y_scores_mv_brier = np.concatenate([ind_mv_brier, ood_mv_brier])
    metrics_results['MV_Brier'] = roc_auc_score(y_true, y_scores_mv_brier)

    # --- METRIC 4: Model Variance (Spherical) ---
    ind_mv_spherical = mv_spherical(logits_pred=ind_logits, logits_gt=ind_logits)
    ood_mv_spherical = mv_spherical(logits_pred=ood_logits, logits_gt=ood_logits)
    
    y_scores_mv_spherical = np.concatenate([ind_mv_spherical, ood_mv_spherical])
    metrics_results['MV_Spherical'] = roc_auc_score(y_true, y_scores_mv_spherical)

    return metrics_results

print(f"--- Starting OOD Uncertainty Evaluation ---")
print(f"{'Client':<8} | {'Strategy':<12} | {'MSP':<10} | {'MV LogScore':<10} | {'MV Brier':<10} | {'MV Spherical':<12}")
print("-" * 80)

strategies = ["random", "accuracy", "uncertainty"]

for strategy in strategies:
    for client_id in range(1, n_clients + 1):
        ind_logits_path = f"{data_path}/logits/client_{client_id}/{strategy}_ind.pt"
        ood_logits_path = f"{data_path}/logits/client_{client_id}/{strategy}_ood.pt"
        
        all_ind_logits = torch.load(ind_logits_path).numpy()
        all_ood_logits = torch.load(ood_logits_path).numpy()
        
        results = compute_ood_metrics(all_ind_logits, all_ood_logits)
        
        print(f"{client_id:02d}       | {strategy:<12} | {results['MSP']:.4f}     | {results['MV_LogScore']:.4f}       | {results['MV_Brier']:.4f}    | {results['MV_Spherical']:.4f}")
    print("-" * 80)

print(f"--- Calculation Complete ---")