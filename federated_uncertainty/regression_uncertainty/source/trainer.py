import copy
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models.utils import variance_link, natural_link
from .utils.objectives import (
    GaussianNLL,
    NaturalGaussianNLL,
    compute_anti_regularization,
)

@torch.enable_grad()
def update(
    network: nn.Module,
    data: DataLoader,
    loss_fn: nn.Module,
    link_fn: callable,
    opt: torch.optim.Optimizer,
    anti_regularization_weight: float,
    device: str
) -> list:
    network.train()

    losses = list()
    for x, y in data:
        x, y = [t.to(device) for t in [x, y]]

        x_1, x_2 = link_fn(network.forward(x))
        loss = loss_fn(x_1, x_2, y)

        if anti_regularization_weight > 0:
            loss -= compute_anti_regularization(network) * anti_regularization_weight

        losses.append(loss.item())
        opt.zero_grad()
        try:
            loss.backward()
            opt.step()
        except:
            print("Exception in update step")

    return losses

@torch.no_grad()
def evaluate(
    network: nn.Module, data: DataLoader, metric: callable, link_fn: callable, device: str
) -> float:
    network.eval()

    x_1s, x_2s, ys = list(), list(), list()
    for x, y in data:
        x = x.to(device)
        y = y.to(device)

        x_1, x_2 = link_fn(network.forward(x))

        x_1s.append(x_1)
        x_2s.append(x_2)
        ys.append(y)

    return metric(
        torch.concat(x_1s, dim=0), torch.concat(x_2s, dim=0), torch.concat(ys, dim=0)
    ).cpu().item()


def fit(
    network: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience=0,
    use_natural=False,
    anti_regularization_weight=0.0,
    verbose=True
):
    optimizer = torch.optim.Adam(
        params=network.parameters(), lr=lr, weight_decay=weight_decay
    )

    if use_natural:
        loss_fn = NaturalGaussianNLL()
        link_fn = natural_link
    else:
        loss_fn = GaussianNLL()
        link_fn = variance_link
    metric = loss_fn

    train_losses, val_perfs = (
        list(),
        [evaluate(network=network, data=val_loader, metric=metric, link_fn=link_fn, device=device)],
    )

    pbar = tqdm(range(epochs), disable=not verbose)
    for _ in pbar:
        # update network
        tl = update(
            network=network,
            data=train_loader,
            loss_fn=loss_fn,
            link_fn=link_fn,
            opt=optimizer,
            anti_regularization_weight=anti_regularization_weight,
            device=device
        )
        train_losses.extend(tl)
        vp = evaluate(network=network, data=val_loader, metric=metric, link_fn=link_fn, device=device)

        if len(val_perfs) == 1 or vp < min(val_perfs):
            best_model = copy.deepcopy(network)

        val_perfs.append(vp)

        pbar.set_description_str(
            desc=f"Train loss {round(np.mean(tl), 4):8.4f}, "
            + f"val performance: {round(vp, 4):8.4f}"
        )

        # stop search if no improvements for defined number of epochs
        if (
            patience > 0
            and len(val_perfs) > patience
            and np.argmin(val_perfs) < len(val_perfs) - patience
        ):
            break

    return best_model, train_losses, val_perfs
