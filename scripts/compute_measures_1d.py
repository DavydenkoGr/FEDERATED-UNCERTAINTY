import sys
from pathlib import Path

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdu.unc.multidimensional_uncertainty import compute_all_uncertainties
from mdu.eval.eval_utils import load_pickle
import numpy as np
from collections import defaultdict
from mdu.data.constants import DatasetName
from mdu.unc.multidimensional_uncertainty import UncertaintyEstimator
import argparse
from mdu.unc.constants import UncertaintyType
from mdu.unc.risk_metrics.constants import GName, RiskType, ApproximationType
from mdu.data.data_utils import split_dataset_indices

ENSEMBLE_GROUPS = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ind_dataset",
    type=str,
    required=True,
    choices=[
        DatasetName.CIFAR10.value,
        DatasetName.CIFAR100.value,
        DatasetName.TINY_IMAGENET.value,
    ],
)
parser.add_argument(
    "--ood_dataset",
    type=str,
    required=True,
    choices=[
        DatasetName.CIFAR10.value,
        DatasetName.CIFAR100.value,
        DatasetName.TINY_IMAGENET.value,
        DatasetName.SVHN.value,
    ],
)
parser.add_argument(
    "--uncertainty_measure_type",
    type=str,
    required=True,
    choices=[
        UncertaintyType.MAHALANOBIS.value,
        UncertaintyType.GMM.value,
        UncertaintyType.RISK.value,
    ],
)
parser.add_argument("--uncertainty_measure_print_name", type=str, required=True)

parser.add_argument(
    "--uncertainty_measure_gname",
    type=str,
    required=False,
    choices=[
        GName.LOG_SCORE.value,
        GName.BRIER_SCORE.value,
        GName.SPHERICAL_SCORE.value,
        GName.ZERO_ONE_SCORE.value,
    ],
)
parser.add_argument(
    "--uncertainty_measure_risk_type",
    type=str,
    required=False,
    choices=[
        RiskType.EXCESS_RISK.value,
        RiskType.BAYES_RISK.value,
        RiskType.TOTAL_RISK.value,
    ],
)
parser.add_argument(
    "--uncertainty_measure_gt_approx",
    type=str,
    required=False,
    choices=[
        ApproximationType.OUTER.value,
        ApproximationType.INNER.value,
        ApproximationType.CENTRAL.value,
    ],
)
parser.add_argument(
    "--uncertainty_measure_pred_approx",
    type=str,
    required=False,
    choices=[
        ApproximationType.OUTER.value,
        ApproximationType.INNER.value,
        ApproximationType.CENTRAL.value,
    ],
)
parser.add_argument("--uncertainty_measure_T", type=float, required=False, default=1.0)


args = parser.parse_args()

ind_dataset = args.ind_dataset
ood_dataset = args.ood_dataset
uncertainty_measure_type = UncertaintyType(args.uncertainty_measure_type)
uncertainty_measure_print_name = args.uncertainty_measure_print_name
uncertainty_measure_gname = GName(args.uncertainty_measure_gname) if args.uncertainty_measure_gname else None
uncertainty_measure_risk_type = RiskType(args.uncertainty_measure_risk_type) if args.uncertainty_measure_risk_type else None
uncertainty_measure_gt_approx = ApproximationType(args.uncertainty_measure_gt_approx) if args.uncertainty_measure_gt_approx else None
uncertainty_measure_pred_approx = ApproximationType(args.uncertainty_measure_pred_approx) if args.uncertainty_measure_pred_approx else None
uncertainty_measure_T = args.uncertainty_measure_T


def form_config(
    uncertainty_measure_type_,
    uncertainty_measure_print_name_,
    uncertainty_measure_gname_,
    uncertainty_measure_risk_type_,
    uncertainty_measure_gt_approx_,
    uncertainty_measure_pred_approx_,
    uncertainty_measure_T_,
):
    if uncertainty_measure_type_ in [
        UncertaintyType.MAHALANOBIS,
        UncertaintyType.GMM,
    ]:
        return [
            {
                "type": uncertainty_measure_type_,
                "print_name": uncertainty_measure_print_name_,
                "kwargs": {},
            },
        ]
    elif uncertainty_measure_type_ in [UncertaintyType.RISK]:
        return [
            {
                "type": UncertaintyType.RISK,
                "print_name": uncertainty_measure_print_name_,
                "kwargs": {
                    "g_name": uncertainty_measure_gname_,
                    "risk_type": uncertainty_measure_risk_type_,
                    "gt_approx": uncertainty_measure_gt_approx_,
                    "pred_approx": uncertainty_measure_pred_approx_,
                    "T": uncertainty_measure_T_,
                },
            },
        ]
    else:
        raise ValueError(
            f"Uncertainty measure {uncertainty_measure_type_} not supported"
        )


uncertainty_config = form_config(
    uncertainty_measure_type,
    uncertainty_measure_print_name,
    uncertainty_measure_gname,
    uncertainty_measure_risk_type,
    uncertainty_measure_gt_approx,
    uncertainty_measure_pred_approx,
    uncertainty_measure_T,
)

estimator = [
    UncertaintyEstimator(
        config["type"], print_name=config.get("print_name", None), **config["kwargs"]
    )
    for config in uncertainty_config
]

results = defaultdict(list)

for group in ENSEMBLE_GROUPS:
    all_ind_logits = []
    all_ood_logits = []
    for model_id in group:
        ind_res = load_pickle(
            f"./resources/model_weights/{ind_dataset}/checkpoints/resnet18/CrossEntropy/{model_id}/{ind_dataset}.pkl"
        )
        ood_res = load_pickle(
            f"./resources/model_weights/{ind_dataset}/checkpoints/resnet18/CrossEntropy/{model_id}/{ood_dataset}.pkl"
        )

        logits_ind = ind_res["embeddings"]
        all_ind_logits.append(ind_res["embeddings"][None])
        all_ood_logits.append(ood_res["embeddings"][None])

    y_ind = ind_res["labels"]
    y_ood = ood_res["labels"]

    _, train_cond_idx, calib_idx, test_idx = split_dataset_indices(
        logits_ind,
        y_ind,
        train_ratio=0.0,
        calib_ratio=0.1,
        test_ratio=0.8,
    )

    y_train_cond = y_ind[train_cond_idx]
    y_calib = y_ind[calib_idx]

    X_train_cond = np.vstack(all_ind_logits)[:, train_cond_idx, :]
    X_calib = np.vstack(all_ind_logits)[:, calib_idx, :]
    X_test = np.vstack(all_ind_logits)[:, test_idx, :]

    estimator[0] = estimator[0].fit(X_train_cond, y_train_cond)
    
    X_ind = X_test
    X_calib = X_calib
    X_ood = np.vstack(all_ood_logits)

    ind_unc = compute_all_uncertainties(estimators=estimator, logits=X_ind)
    calib_unc = compute_all_uncertainties(estimators=estimator, logits=X_calib)
    ood_unc = compute_all_uncertainties(estimators=estimator, logits=X_ood)

    results[ind_dataset].append(ind_unc)
    results[ind_dataset + "_calib"].append(calib_unc)
    results[ood_dataset].append(ood_unc)

# Save results to npz file
# Create directory structure: results/{ind_dataset}/{uncertainty_type}/{ood_dataset}/
import os

# Determine uncertainty type string for path
if args.uncertainty_measure_type.lower() == "risk":
    uncertainty_type_str = f"risk_{args.uncertainty_measure_gname.lower()}_{args.uncertainty_measure_risk_type.lower()}_{args.uncertainty_measure_gt_approx.lower()}"
    if args.uncertainty_measure_pred_approx:
        uncertainty_type_str += f"_{args.uncertainty_measure_pred_approx.lower()}"
    uncertainty_type_str += f"_T_{args.uncertainty_measure_T}"
else:
    uncertainty_type_str = args.uncertainty_measure_type.lower()

# Create directory path
save_dir = os.path.join("results", ind_dataset, uncertainty_type_str, ood_dataset)
os.makedirs(save_dir, exist_ok=True)

# Create filename
if args.uncertainty_measure_type.lower() == "risk":
    filename = f"{ind_dataset}_{ood_dataset}_{uncertainty_type_str}.npz"
else:
    filename = f"{ind_dataset}_{ood_dataset}_{uncertainty_type_str}.npz"

# Save to file
save_path = os.path.join(save_dir, filename)
np.savez(save_path, ind_test=results[ind_dataset], ind_calib=results[ind_dataset + "_calib"], ood=results[ood_dataset])
print(f"Results saved to: {save_path}")

