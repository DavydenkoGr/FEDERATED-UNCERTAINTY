import sys
from pathlib import Path

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from federated_uncertainty.data.constants import DatasetName
from federated_uncertainty.data.data_utils import split_dataset
from federated_uncertainty.data.load_dataset import get_dataset
from federated_uncertainty.eval.eval_utils import get_ensemble_predictions
from federated_uncertainty.nn.constants import ModelName
from federated_uncertainty.nn.load_models import get_model
from federated_uncertainty.optim.train import train_ensembles
from federated_uncertainty.randomness import set_all_seeds

seed = 1
set_all_seeds(seed)

dataset_name = DatasetName.CIFAR10

n_classes = 10
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)
n_members = 5
input_dim = 2
hidden_dim = 32
n_epochs = 50
batch_size = 64
lambda_ = 1.0
lr = 1e-3
criterion = nn.CrossEntropyLoss()

eps = 0.5
max_iters = 1000
tol = 1e-6
random_state = seed

radius = 8.0
angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

dataset_params = {
    "n_samples": 4000,
    "cluster_std": 1.0,
    "centers": centers,
}

X, y = get_dataset(dataset_name, **dataset_params)

(
    X_train_main,
    X_train_cond,
    X_calib,
    X_test,
    y_train_main,
    y_train_cond,
    y_calib,
    y_test,
) = split_dataset(X, y)

X_tensor = torch.tensor(X_train_main, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y_train_main, dtype=torch.long, device=device)

ensemble = [
    get_model(
        ModelName.VGG11,
        n_classes,
        # input_dim=input_dim,
        # hidden_dim=hidden_dim,
    ).to(device)
    for _ in range(n_members)
]

ensemble = train_ensembles(
    ensemble,
    X_tensor,
    y_tensor,
    batch_size,
    n_epochs,
    lambda_=lambda_,
    criterion=criterion,
    lr=lr,
)

accuracies = []
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

for i, model in enumerate(ensemble):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == y_test_tensor).sum().item()
        acc = correct / len(y_test_tensor)
        accuracies.append(acc)
        print(f"Model {i + 1} accuracy: {acc:.4f}")