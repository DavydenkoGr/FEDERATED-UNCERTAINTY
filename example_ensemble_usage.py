import torch
from mdu.randomness import set_all_seeds
import numpy as np
from mdu.nn.architectures import ShallowNet
from mdu.optim.train import train_emsembles
import torch.nn as nn
from mdu.vis.toy_plots import plot_decision_boundaries
from mdu.unc.constants import UncertaintyType
from mdu.unc.risk_metrics.constants import GName, RiskType, ApproximationType
from mdu.unc.multidimensional_uncertainty import UncertaintyEnsemble

torch.manual_seed(0)
np.random.seed(0)
set_all_seeds(42)

from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split

# Dataset configuration
toy_dataset = "moons"
n_classes = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_members = 5
input_dim = 2
hidden_dim = 32
n_epochs = 50
batch_size = 64
lambda_ = 1.0
criterion = nn.CrossEntropyLoss()

# Define uncertainty measures using the new ensemble approach
UNCERTAINTY_MEASURES = [
    {
        "type": UncertaintyType.RISK,
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.BAYES_RISK,
            "gt_approx": ApproximationType.INNER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.INNER,
            "pred_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.MAHALANOBIS,
        "kwargs": {},
    },
]

# Create dataset
if toy_dataset == "moons":
    if n_classes != 2:
        raise ValueError("n_classes must be 2 for moons dataset")
    X, y = make_moons(n_samples=4000, noise=0.1, random_state=42)
elif toy_dataset == "blobs":
    X, y = make_blobs(
        n_samples=4000, centers=n_classes, cluster_std=1.0, random_state=42
    )
else:
    raise ValueError(f"Invalid toy dataset: {toy_dataset}")

# Split data
X_train_main, X_temp, y_train_main, y_temp = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)
X_train_cond, X_temp2, y_train_cond, y_temp2 = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
X_calib, X_test, y_calib, y_test = train_test_split(
    X_temp2, y_temp2, test_size=0.5, random_state=42, stratify=y_temp2
)

# Train ensemble models
X_tensor = torch.tensor(X_train_main, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y_train_main, dtype=torch.long, device=device)

ensemble = [
    ShallowNet(input_dim, hidden_dim, n_classes).to(device) for _ in range(n_members)
]

ensemble = train_emsembles(
    ensemble,
    X_tensor,
    y_tensor,
    batch_size,
    n_epochs,
    lambda_=lambda_,
    criterion=criterion,
)


# Get logits for uncertainty estimation
def get_ensemble_logits(models, X_data):
    """Get ensemble logits for uncertainty estimation"""
    X_tensor = torch.tensor(X_data, dtype=torch.float32, device=device)
    all_logits = []

    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor)
            all_logits.append(logits.cpu().numpy())

    # Stack logits: shape (n_models, n_samples, n_classes)
    stacked_logits = np.stack(all_logits, axis=0)
    # Reshape to (n_samples, n_models * n_classes) for uncertainty estimation
    return stacked_logits.transpose(1, 0, 2).reshape(len(X_data), -1)


# Get logits for training and calibration
X_calib_logits = get_ensemble_logits(ensemble, X_calib)
X_test_logits = get_ensemble_logits(ensemble, X_test)

# OLD WAY: Manual uncertainty measure handling (commented out)
"""
measure_results = {}
for uncertainty_measure in UNCERTAINTY_MEASURES:
    uncertainty_wrapper = UncertaintyWrapper(
        uncertainty_measure["type"], **uncertainty_measure["kwargs"]
    )
    uncertainty_wrapper.fit(X_calib_logits)

    # Compute uncertainty measures on the grid
    measure_results_ = uncertainty_wrapper.predict(grid_tensor)
    # Reshape for plotting
    measure_results_[uncertainty_wrapper.name] = measure_results_.reshape(xx.shape)
    measure_results[uncertainty_wrapper.name] = measure_results_

ot_scorer = OTCPOrdering(positive=True)
ot_scorer.fit(measure_results)
grid_l2_norms, _ = ot_scorer.predict(grid_uq_labels)
"""

# NEW WAY: Using UncertaintyEnsemble
print("Creating UncertaintyEnsemble...")
uncertainty_ensemble = UncertaintyEnsemble(
    uncertainty_configs=UNCERTAINTY_MEASURES,
    positive=True,  # Use positive reference distribution for OT
)

print("Fitting uncertainty ensemble...")
# Fit the ensemble using calibration data for both training and calibration
# (In practice, you might want to use different splits)
uncertainty_ensemble.fit(X_train=X_calib_logits, X_calib=X_calib_logits)

print("Predicting uncertainties on test data...")
# Get ensemble uncertainty predictions
test_l2_norms, test_ordering = uncertainty_ensemble.predict(X_test_logits)

print(
    f"Ensemble contains {uncertainty_ensemble.n_uncertainty_measures} uncertainty measures:"
)
print(f"Measures: {uncertainty_ensemble.estimator_names}")

# Get individual predictions for analysis
individual_preds = uncertainty_ensemble.get_individual_predictions(X_test_logits)
print("\nIndividual uncertainty measures on test data:")
for name, scores in individual_preds.items():
    print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

# Get combined uncertainty matrix
combined_matrix = uncertainty_ensemble.get_combined_uncertainty_matrix(X_test_logits)
print(f"\nCombined uncertainty matrix shape: {combined_matrix.shape}")
print(
    f"Ensemble L2 norms: mean={test_l2_norms.mean():.4f}, std={test_l2_norms.std():.4f}"
)

# Example: Get most uncertain samples
most_uncertain_indices = test_ordering[-10:]  # Top 10 most uncertain
print(f"\nIndices of 10 most uncertain test samples: {most_uncertain_indices}")
print(f"Their uncertainty scores: {test_l2_norms[most_uncertain_indices]}")

# For visualization (if you have a grid)
# grid_tensor, xx, yy = plot_decision_boundaries(ensemble, X_test, y_test, accuracies, device, n_classes, return_grid=True)
# grid_logits = get_ensemble_logits(ensemble, grid_tensor.cpu().numpy())
# grid_l2_norms, _ = uncertainty_ensemble.predict(grid_logits)
# grid_uncertainty = grid_l2_norms.reshape(xx.shape)

print("\n✅ UncertaintyEnsemble usage example completed successfully!")
