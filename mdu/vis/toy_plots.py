import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from typing import Optional


def plot_decision_boundaries(ensemble, X_test, y_test, accuracies, device, n_classes, return_grid: bool = True) -> Optional[torch.Tensor]:
    """
    Plot the decision boundaries of an ensemble of models.
    """
    # Define the mesh grid for plotting decision boundaries
    assert X_test.shape[1] == 2, "X_test must be a 2D array"

    h = 0.02  # step size in the mesh
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)

    # Calculate mean and std of ensemble accuracies
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    # Plot all ensemble member boundaries with low alpha for visual blending
    for i, model in enumerate(ensemble):
        model.eval()
        with torch.no_grad():
            Z = model(grid_tensor)
            Z = torch.argmax(Z, dim=1).cpu().numpy()
        Z = Z.reshape(xx.shape)
        # Use a single color for all boundaries, but low alpha for overlap effect
        ax.contour(
            xx,
            yy,
            Z,
            levels=[1.0 / n_classes],
            colors="k",
            linewidths=1.8,
            alpha=0.18,
            linestyles="--",
        )

    # Plot test data
    scatter = ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=90,
        linewidth=1.2,
        alpha=0.95,
        label="Test data",
    )

    # Style tweaks for publication
    ax.set_title(
        f"Ensemble Decision Boundaries\nMean accuracy: {mean_acc:.3f} ± {std_acc:.3f}",
        fontsize=24,
        pad=20,
    )
    ax.set_xlabel("$x_1$", fontsize=20)
    ax.set_ylabel("$x_2$", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=17)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal", adjustable="box")

    # Only show legend for test data
    # ax.legend(loc='lower right', fontsize=17, frameon=True)

    plt.tight_layout()

    # Ensure the pics directory exists
    os.makedirs("pics", exist_ok=True)
    plt.savefig("pics/2d_ensemble_decision_boundaries.pdf", bbox_inches="tight")

    plt.show()

    if return_grid:
        return grid_tensor
