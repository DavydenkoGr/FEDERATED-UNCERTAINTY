import torch
from mdu.randomness import set_all_seeds
import numpy as np
from mdu.nn.architectures import ShallowNet
from mdu.optim.train import train_emsembles
import torch.nn as nn
from mdu.vis.toy_plots import plot_decision_boundaries

torch.manual_seed(0)
np.random.seed(0)

set_all_seeds(42)

from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split

toy_dataset = "moons"
n_classes = 2
device = torch.device("cuda:0")
n_members = 5
input_dim = 2
hidden_dim = 32
n_epochs = 50
batch_size = 64
lambda_ = 1.0
criterion = nn.CrossEntropyLoss()

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


X_train_main, X_temp, y_train_main, y_temp = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)
X_train_cond, X_temp2, y_train_cond, y_temp2 = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
X_calib, X_test, y_calib, y_test = train_test_split(
    X_temp2, y_temp2, test_size=0.5, random_state=42, stratify=y_temp2
)

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

plot_decision_boundaries(ensemble, X_test, y_test, accuracies, device, n_classes)


