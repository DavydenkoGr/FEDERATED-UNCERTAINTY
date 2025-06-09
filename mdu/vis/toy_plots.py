import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from typing import Optional
import matplotlib as mpl

# Ensure plt is imported before use
# Set global font sizes for a publication-quality figure
mpl.rcParams.update(
    {
        "font.size": 22,
        "axes.titlesize": 26,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "figure.titlesize": 28,
        "pdf.fonttype": 42,  # For editable text in PDF
        "ps.fonttype": 42,
    }
)


def plot_decision_boundaries(
    ensemble, X_test, y_test, accuracies, device, n_classes, return_grid: bool = True
) -> Optional[tuple[torch.Tensor, np.ndarray, np.ndarray]]:
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
        return grid_tensor, xx, yy


def plot_uncertainty_measures(xx, yy, uncertainty_measures_dict, X_test=None):
    """
    Plot uncertainty measures from a dictionary.

    Args:
        xx, yy: Grid coordinates for contour plots
        uncertainty_measures_dict: Dict with measure names as keys and grid values as values
        X_test: Optional test data points to scatter on top of plots
    """
    # Get number of uncertainty measures
    n_measures = len(uncertainty_measures_dict)

    # Determine optimal subplot layout
    if n_measures <= 4:
        n_cols = n_measures
        n_rows = 1
    else:
        # For more than 4 plots, use multiple rows
        n_cols = 4
        n_rows = (n_measures + n_cols - 1) // n_cols  # Ceiling division

    # Calculate figure size based on layout
    fig_width = n_cols * 10
    fig_height = n_rows * 8

    # Create subplots
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_width, fig_height), constrained_layout=True
    )

    # Handle case where we have only one row or one plot
    if n_rows == 1 and n_cols == 1:
        axes = [axes]  # Make it a list for consistency
    elif n_rows == 1:
        axes = axes  # Already a list for single row
    else:
        axes = axes.flatten()  # Flatten for easy indexing

    # Common scatter kwargs to reduce repetition
    scatter_kwargs = {
        "cmap": plt.cm.Set1,
        "edgecolor": "k",
        "s": 80,
        "linewidth": 1.2,
        "label": "Test data",
    }

    # Plot each uncertainty measure
    for idx, (measure_name, measure_grid) in enumerate(
        uncertainty_measures_dict.items()
    ):
        ax = axes[idx]

        # Create contour plot
        cont = ax.contourf(xx, yy, measure_grid, levels=30, cmap="magma", alpha=0.8)

        # Add colorbar
        cbar = fig.colorbar(cont, ax=ax, shrink=0.8, pad=0.02)

        # Add scatter plot of test data if provided
        if X_test is not None:
            ax.scatter(X_test[:, 0], X_test[:, 1], **scatter_kwargs)

        # Set labels and title
        ax.set_title(measure_name)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        # Remove top/right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplots if we have extra axes
    for idx in range(n_measures, len(axes)):
        axes[idx].set_visible(False)

    # Save and display
    plt.savefig("./pics/uq_grid_visualization.pdf", bbox_inches="tight", dpi=300)
    plt.show()
