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

parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', default=5, type=int, help='number of clients')
parser.add_argument("--data_path", type=str, required=True)

args = parser.parse_args()

data_path = args.data_path
n_clients = args.n_clients

def compute_msp_auc(ind_logits, ood_logits):
    """
    ind_logits: np.array (n_models, N_ind, n_classes)
    ood_logits: np.array (n_models, N_ood, n_classes)
    """

    # Mean ensemble logits
    ind_logits_mean = ind_logits.mean(axis=0)
    ood_logits_mean = ood_logits.mean(axis=0)

    # Softmax
    ind_probs = F.softmax(torch.tensor(ind_logits_mean), dim=1).numpy()
    ood_probs = F.softmax(torch.tensor(ood_logits_mean), dim=1).numpy()

    # Compute MSP = 1 - max softmax prob
    ind_msp = 1 - ind_probs.max(axis=1)
    ood_msp = 1 - ood_probs.max(axis=1)

    # Binary labels: 0 = IND, 1 = OOD
    y_true = np.concatenate([
        np.zeros(len(ind_msp)),
        np.ones(len(ood_msp))
    ])

    y_scores = np.concatenate([ind_msp, ood_msp])

    auc = roc_auc_score(y_true, y_scores)
    return auc

print(f"--- Starting MSP ROC-AUC Calculation ---")
print("-" * 60)

strategies = ["random", "accuracy", "uncertainty"]

for client_id in range(1, n_clients + 1):
    for strategy in strategies:
        try:
            # Define file paths
            ind_logits_path = f"{data_path}/logits/client_{client_id}/{strategy}_ind.pt"
            ood_logits_path = f"{data_path}/logits/client_{client_id}/{strategy}_ood.pt"
            
            # Load logits
            all_ind_logits = torch.load(ind_logits_path).numpy()
            all_ood_logits = torch.load(ood_logits_path).numpy()
            
            # Compute MSP AUC
            msp_auc = compute_msp_auc(all_ind_logits, all_ood_logits)
            
            # Print formatted results
            print(f"Client {client_id:02d} | Strategy: {strategy:<12} | MSP ROC-AUC: {msp_auc:.4f}")
        
        except FileNotFoundError:
            print(f"Client {client_id:02d} | Strategy: {strategy:<12} | Data not found, skipping.")
        except Exception as e:
            print(f"Client {client_id:02d} | Strategy: {strategy:<12} | ERROR: {e}")
    print("-" * 60)

print("-" * 60)
print(f"--- Calculation Complete ---")