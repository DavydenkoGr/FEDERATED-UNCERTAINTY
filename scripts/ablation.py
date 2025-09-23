import sys
from pathlib import Path

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from configs.uncertainty_measures_configs import (
    ADDITIVE_TOTALS,
    BAYES_DIFFERENT_APPROXIMATIONS_LOGSCORE,
    BAYES_DIFFERENT_APPROXIMATIONS_SPHERICALSCORE,
    BAYES_DIFFERENT_INSTANTIATIONS,
    BAYES_RISK_AND_BAYES_RISK,
    CORRESPONDING_COMPONENTS_TO_ADDITIVE_TOTALS,
    EAT_M,
    EXCESSES_DIFFERENT_APPROXIMATIONS_BRIERSCORE,
    EXCESSES_DIFFERENT_APPROXIMATIONS_LOGSCORE,
    EXCESSES_DIFFERENT_APPROXIMATIONS_SPHERICALSCORE,
    EXCESSES_DIFFERENT_INSTANTIATIONS,
    MAHALANOBIS_AND_BAYES_RISK,
    MULTIPLE_SAME_MEASURES,
    SINGLE_AND_FAKE,
    SINGLE_MEASURE,
)
from mdu.data.constants import DatasetName
from mdu.unc.constants import OTTarget, SamplingMethod, ScalingType
from scripts.main_images_mis import main as main_mis
from scripts.main_images_ood import main as main_ood

if __name__ == "__main__":
    seed = 42
    # UNCERTAINTY_MEASURES = MAHALANOBIS_AND_BAYES_RISK # + BAYES_RISK_AND_BAYES_RISK + EXCESSES_DIFFERENT_INSTANTIATIONS
    UNCERTAINTY_MEASURES = EAT_M
    print(UNCERTAINTY_MEASURES)
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    full_ablation_df = None

    ENSEMBLE_GROUPS = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
    ]

    ind_dataset = DatasetName.TINY_IMAGENET.value
    ood_dataset = DatasetName.IMAGENET_R.value
    weights_root = "./resources/model_weights"

    for configuration in [
        {
            "target": OTTarget.EXP,
            "scaling_type": ScalingType.FEATURE_WISE,
            "grid_size": 5,
        },
        {
            "target": OTTarget.EXP,
            "scaling_type": ScalingType.FEATURE_WISE,
            "grid_size": 2,
        },
        {
            "target": OTTarget.EXP,
            "scaling_type": ScalingType.FEATURE_WISE,
            "grid_size": 0,
        },
        {"target": OTTarget.EXP, "scaling_type": ScalingType.GLOBAL, "grid_size": 5},
        {"target": OTTarget.EXP, "scaling_type": ScalingType.GLOBAL, "grid_size": 2},
        {"target": OTTarget.EXP, "scaling_type": ScalingType.GLOBAL, "grid_size": 0},
        {"target": OTTarget.EXP, "scaling_type": ScalingType.IDENTITY, "grid_size": 5},
        {"target": OTTarget.EXP, "scaling_type": ScalingType.IDENTITY, "grid_size": 2},
        {"target": OTTarget.EXP, "scaling_type": ScalingType.IDENTITY, "grid_size": 0},
        {
            "target": OTTarget.BETA,
            "scaling_type": ScalingType.FEATURE_WISE,
            "grid_size": 5,
        },
        {
            "target": OTTarget.BETA,
            "scaling_type": ScalingType.FEATURE_WISE,
            "grid_size": 2,
        },
        {
            "target": OTTarget.BETA,
            "scaling_type": ScalingType.FEATURE_WISE,
            "grid_size": 0,
        },
        {"target": OTTarget.BETA, "scaling_type": ScalingType.GLOBAL, "grid_size": 5},
        {"target": OTTarget.BETA, "scaling_type": ScalingType.GLOBAL, "grid_size": 2},
        {"target": OTTarget.BETA, "scaling_type": ScalingType.GLOBAL, "grid_size": 0},
        {"target": OTTarget.BETA, "scaling_type": ScalingType.IDENTITY, "grid_size": 5},
        {"target": OTTarget.BETA, "scaling_type": ScalingType.IDENTITY, "grid_size": 2},
        {"target": OTTarget.BETA, "scaling_type": ScalingType.IDENTITY, "grid_size": 0},
    ]:

        target = configuration["target"]
        sampling_method = SamplingMethod.GRID
        scaling_type = configuration["scaling_type"]
        grid_size = configuration["grid_size"]
        n_targets_multiplier = 1
        eps = 0.5
        max_iters = 1000
        tol = 1e-6
        random_state = seed

        df = main_ood(
            ensemble_groups=ENSEMBLE_GROUPS,
            ind_dataset=ind_dataset,
            ood_dataset=ood_dataset,
            uncertainty_measures=UNCERTAINTY_MEASURES,
            weights_root=weights_root,
            seed=seed,
            target=target,
            sampling_method=sampling_method,
            scaling_type=scaling_type,
            grid_size=grid_size,
            n_targets_multiplier=n_targets_multiplier,
            eps=eps,
            max_iters=max_iters,
            tol=tol,
        )
        df["target"] = target.value
        df["scaling_type"] = scaling_type.value
        df["grid_size"] = grid_size
        print(df)
        if full_ablation_df is None:
            full_ablation_df = df
        else:
            full_ablation_df = pd.concat([full_ablation_df, df])
    full_ablation_df.to_csv(
        f"./ablation_ood_results_{ind_dataset.lower()}_{ood_dataset.lower()}.csv",
        index=False,
    )
