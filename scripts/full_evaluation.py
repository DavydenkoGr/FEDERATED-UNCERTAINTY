import sys
from pathlib import Path

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdu.unc.risk_metrics import RiskType, GName, ApproximationType
from mdu.unc.constants import UncertaintyType
from mdu.data.constants import DatasetName

import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from scripts.parse_args import get_args

from configs.interesting_compositions import INTERESTING_COMPOSITIONS
from mdu.eval.eval_utils import (
    create_output_filename,
    load_predictions_and_split,
    process_multidimensional_composition,
    process_uncertainty_measure,
)

# Define all possible combinations
datasets_ind = [
    DatasetName.CIFAR10,
    DatasetName.CIFAR100,
    DatasetName.TINY_IMAGENET,
]
datasets_ood_ = [
    DatasetName.CIFAR10,
    DatasetName.CIFAR100,
    DatasetName.TINY_IMAGENET,
    DatasetName.SVHN,
]
datasets_ood_tiny_imagenet_ = [
    DatasetName.TINY_IMAGENET,
    DatasetName.IMAGENET_A,
    DatasetName.IMAGENET_O,
    DatasetName.IMAGENET_R,
]
uncertainty_types = [
    UncertaintyType.RISK,
    UncertaintyType.MAHALANOBIS,
    UncertaintyType.GMM,
]
gnames = [
    GName.LOG_SCORE,
    GName.BRIER_SCORE,
    GName.ZERO_ONE_SCORE,
    GName.SPHERICAL_SCORE,
]
risk_types = [RiskType.TOTAL_RISK, RiskType.EXCESS_RISK, RiskType.BAYES_RISK]
approximations = [
    ApproximationType.OUTER,
    ApproximationType.INNER,
    ApproximationType.CENTRAL,
]


def main():
    args = get_args()
    results = []

    print("Loading predictions for all datasets...")
    prediction_data = {}
    for ind_dataset in datasets_ind:
        try:
            pred_data = load_predictions_and_split(
                ind_dataset, weights_root=args.weights_root
            )
            prediction_data[ind_dataset] = pred_data

        except Exception as e:
            print(f"✗ Failed to load predictions for {ind_dataset.value}: {e}")
            prediction_data[ind_dataset] = None

    # Create progress bar - count regular measures + compositions
    total_combinations = 0
    for ind_dataset in datasets_ind:
        if ind_dataset == DatasetName.TINY_IMAGENET:
            datasets_ood = datasets_ood_tiny_imagenet_
        else:
            datasets_ood = datasets_ood_

        for ood_dataset in datasets_ood:
            for uncertainty_type in uncertainty_types:
                if uncertainty_type == UncertaintyType.RISK:
                    for gname, risk_type, gt_approx in product(
                        gnames, risk_types, approximations
                    ):
                        if risk_type == RiskType.BAYES_RISK:
                            total_combinations += 1
                        else:
                            total_combinations += len(approximations)
                else:
                    total_combinations += 1
            # Add multidimensional compositions
            total_combinations += len(INTERESTING_COMPOSITIONS)

    pbar = tqdm(total=total_combinations, desc="Processing combinations")

    # Main evaluation loop
    processed_same_dataset = (
        set()
    )  # Track which ind_datasets we've processed for misclassification/selective

    for ind_dataset in datasets_ind:
        if ind_dataset == DatasetName.TINY_IMAGENET:
            datasets_ood = datasets_ood_tiny_imagenet_
        else:
            datasets_ood = datasets_ood_

        for ood_dataset in datasets_ood:
            for uncertainty_type in uncertainty_types:
                if uncertainty_type == UncertaintyType.RISK:
                    # Test all risk combinations
                    for gname, risk_type, gt_approx in product(
                        gnames, risk_types, approximations
                    ):
                        if risk_type == RiskType.BAYES_RISK:
                            # BayesRisk doesn't use pred_approximation
                            pred_approx = None
                            process_uncertainty_measure(
                                ind_dataset,
                                ood_dataset,
                                uncertainty_type,
                                gname,
                                risk_type,
                                gt_approx,
                                pred_approx,
                                prediction_data,
                                results,
                                args,
                                pbar,
                                processed_same_dataset,
                            )
                        else:
                            # TotalRisk and ExcessRisk use pred_approximation
                            for pred_approx in approximations:
                                process_uncertainty_measure(
                                    ind_dataset,
                                    ood_dataset,
                                    uncertainty_type,
                                    gname,
                                    risk_type,
                                    gt_approx,
                                    pred_approx,
                                    prediction_data,
                                    results,
                                    args,
                                    pbar,
                                    processed_same_dataset,
                                )
                else:
                    # Mahalanobis and GMM
                    process_uncertainty_measure(
                        ind_dataset,
                        ood_dataset,
                        uncertainty_type,
                        None,
                        None,
                        None,
                        None,
                        prediction_data,
                        results,
                        args,
                        pbar,
                        processed_same_dataset,
                    )

            # Process multidimensional compositions for this dataset pair
            for composition_name, configs in INTERESTING_COMPOSITIONS.items():
                process_multidimensional_composition(
                    composition_name,
                    configs,
                    ind_dataset,
                    ood_dataset,
                    prediction_data,
                    results,
                    args,
                    processed_same_dataset,
                )
                if pbar is not None:
                    pbar.update(1)

    pbar.close()

    # Convert results to DataFrame and save
    df = pd.DataFrame(results)

    # Create output filename with EntropicOT hyperparameters
    output_filename = create_output_filename(args.output_file, args)
    df.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")
    print(f"Total rows: {len(df)}")

    # Print summary statistics
    print("\n=== SUMMARY ===")
    print(f"Unique ind_datasets: {df['ind_dataset'].nunique()}")
    print(f"Unique ood_datasets: {df['ood_dataset'].nunique()}")
    print(f"Unique measures: {df['measure'].nunique()}")
    print(f"Problem types: {df['problem_type'].unique()}")

    return df


if __name__ == "__main__":
    df = main()
