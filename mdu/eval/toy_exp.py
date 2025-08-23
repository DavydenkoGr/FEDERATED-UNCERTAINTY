import numpy as np
import torch
from tqdm import tqdm
from mdu.optim.train import train_ensembles
from mdu.unc.multidimensional_uncertainty import MultiDimensionalUncertainty
from mdu.nn.load_models import get_model
from mdu.nn.constants import ModelName
import torch.nn as nn
from mdu.eval.eval_utils import get_ensemble_predictions
from sklearn.model_selection import train_test_split
from mdu.vis.toy_plots import plot_decision_boundaries
from mdu.unc.constants import VectorQuantileModel


def eval_unc_decomp(
    multidim_model: VectorQuantileModel,
    train_kwargs: dict,
    multidim_params: dict,
    X: np.ndarray,
    y: np.ndarray,
    test_point: np.ndarray,
    device: torch.device,
    calib_ratio: float,
    val_ratio: float,
    uncertainty_measures: list[dict],
    n_epochs: int,
    input_dim: int,
    n_members: int,
    batch_size: int,
    lambda_: float,
    samples_per_class: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    criterion: nn.Module = nn.CrossEntropyLoss(),
    lr: float = 1e-3,
):
    results = []
    n_classes = len(np.unique(y))

    X_temp, X_calib, y_temp, y_calib = train_test_split(
        X, y, test_size=calib_ratio, random_state=42, stratify=y
    )
    X_pool, X_val, y_pool, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=43, stratify=y_temp
    )
    class_indices = {cls: np.where(y_pool == cls)[0] for cls in np.unique(y_pool)}

    print(
        "Available samples per class in training pool: "
        + ", ".join([f"{cls}: {len(class_indices[cls])}" for cls in np.unique(y_pool)])
    )
    print(f"Validation set size: {len(X_val)}")
    print(f"Calibration set size: {len(X_calib)}")

    for n_samples in tqdm(samples_per_class):
        print(f"\n{'=' * 50}")
        print(
            f"Training with {n_samples} samples per class (total: {n_classes * n_samples})"
        )
        print(f"{'=' * 50}")

        ensemble = [
            get_model(
                ModelName.LINEAR_MODEL,
                n_classes,
                input_dim=input_dim,
            ).to(device)
            for _ in range(n_members)
        ]

        min_class_samples = min(len(class_indices[cls]) for cls in np.unique(y_pool))
        if n_samples <= min_class_samples:
            selected_indices_per_class = [
                np.random.choice(class_indices[cls], size=n_samples, replace=False)
                for cls in np.unique(y_pool)
            ]
            selected_idx = np.concatenate(selected_indices_per_class)

            X_train = X_pool[selected_idx]
            y_train = y_pool[selected_idx]
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)

            ensemble = train_ensembles(
                models=ensemble,
                X_tensor=X_train_tensor,
                y_tensor=y_train_tensor,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lambda_=lambda_,
                criterion=criterion,
                lr=lr,
            )

            plot_decision_boundaries(
                ensemble,
                X_train,
                y_train,
                None,
                device,
                n_classes,
                return_grid=False,
                name=f"toy_2d_{n_samples}",
            )

            val_accuracies = []
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=device)
            for model in ensemble:
                model.eval()
                with torch.no_grad():
                    outputs = model(X_val_tensor)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    acc = np.mean(preds == y_val)
                    val_accuracies.append(acc)
            avg_val_acc = np.mean(val_accuracies)
            print(f"Average validation accuracy: {avg_val_acc:.4f}")

            # Get logits for calibration set and test point
            X_calib_tensor = torch.tensor(X_calib, dtype=torch.float32, device=device)
            test_point_tensor = torch.tensor(
                test_point.reshape(1, -1), dtype=torch.float32, device=device
            )

            X_calib_logits = get_ensemble_predictions(
                ensemble, X_calib_tensor, return_logits=True
            )
            test_point_logits = get_ensemble_predictions(
                ensemble, test_point_tensor, return_logits=True
            )

            # Fit MultiDimensionalUncertainty on calibration logits
            multi_dim_uncertainty = MultiDimensionalUncertainty(
                uncertainty_measures, multidim_model, multidim_params
            )
            multi_dim_uncertainty.fit(
                logits_train=X_calib_logits,
                y_train=y_calib,
                logits_calib=X_calib_logits,
                train_kwargs=train_kwargs,
            )

            # Predict uncertainty for test point
            ordering_indices, uncertainty_results = multi_dim_uncertainty.predict(
                test_point_logits
            )

            # Store results
            result = {}

            # Add uncertainty results

            for key, values in uncertainty_results.items():
                if key == "multidim_scores":
                    result["multidim_scores"] = float(values)
                    continue
                if "BAYES" in key:
                    result["aleatoric_" + key] = float(values)
                else:
                    result["epistemic_" + key] = float(values)

            result["additive_total"] = sum(
                [v for k, v in result.items() if k != "multidim_scores"]
            )
            result.update(
                {
                    "n_samples_per_class": n_samples,
                    "total_samples": n_classes * n_samples,
                    "avg_val_acc": float(avg_val_acc),
                    "val_size": len(X_val),
                    "calib_size": len(X_calib),
                }
            )

            results.append(result)

        else:
            print(f"Skipping {n_samples} samples per class (not enough data available)")

    print(f"\nCompleted experiments for {len(results)} different training set sizes!")
    return results
