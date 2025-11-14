import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import torch.optim as optim
from mdu.optim.regularizers import compute_anti_regularization
from federated_uncertainty.eval import evaluate_single_model_accuracy


def train_ensembles(
    models: list[nn.Module],
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    batch_size: int,
    n_epochs: int,
    lambda_: float,
    criterion: nn.Module,
    lr: float,
):
    n_members = len(models)

    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for i, model in tqdm(enumerate(models)):
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        print(f"\nTraining model {i + 1}/{n_members}")
        for epoch in range(n_epochs):
            epoch_losses = []
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)

                # Compute anti-regularization using the helper function
                anti_reg = compute_anti_regularization(model)

                # Add anti-regularization to the loss (maximize, so subtract)
                total_loss = loss - lambda_ * anti_reg

                total_loss.backward()
                optimizer.step()
                epoch_losses.append(total_loss.item())
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch + 1}/{n_epochs} - Mean Loss: {mean_loss:.4f}")

    return models


def train_ensembles_w_dataloader(
    models: list[nn.Module],
    train_loader,
    device,
    n_epochs: int,
    lambda_: float,
    criterion: nn.Module,
    lr: float,
):
    n_members = len(models)

    for i, model in tqdm(enumerate(models)):
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        print(f"\nTraining model {i + 1}/{n_members}")
        for epoch in range(n_epochs):
            epoch_losses = []
            for i, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)

                # Compute anti-regularization using the helper function
                anti_reg = compute_anti_regularization(model)

                # Add anti-regularization to the loss (maximize, so subtract)
                total_loss = loss - lambda_ * anti_reg

                total_loss.backward()
                optimizer.step()
                epoch_losses.append(total_loss.item())
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch + 1}/{n_epochs} - Mean Loss: {mean_loss:.4f}")

    return models


def train_ensembles_w_local_data(
    models: list[nn.Module],
    train_loaders: list[torch.utils.data.DataLoader],
    device,
    n_epochs: int,
    lambda_: float,
    criterion: nn.Module,
    lr: float,
):
    n_members = len(models)

    for i, model in tqdm(enumerate(models)):
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        print(f"\nTraining model {i + 1}/{n_members}")
        for epoch in range(n_epochs):
            epoch_losses = []
            for xb, yb in train_loaders[i]:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)

                # Compute anti-regularization using the helper function
                anti_reg = compute_anti_regularization(model)

                # Add anti-regularization to the loss (maximize, so subtract)
                total_loss = loss - lambda_ * anti_reg

                total_loss.backward()
                optimizer.step()
                epoch_losses.append(total_loss.item())
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
                model.eval()
                accuracy = evaluate_single_model_accuracy(model, train_loaders[i], device)
                print(f"  Epoch {epoch + 1}/{n_epochs} - Mean Loss: {mean_loss:.4f} - Acc: {accuracy:.4f}")
                model.train()

    return models