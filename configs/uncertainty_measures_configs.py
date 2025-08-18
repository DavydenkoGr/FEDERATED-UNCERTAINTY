from mdu.unc.constants import UncertaintyType
from mdu.unc.risk_metrics.constants import GName, RiskType, ApproximationType


SINGLE_MEASURE = [
    {
        "type": UncertaintyType.RISK,
        "print_name": "EXC 1 1 (log)",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.OUTER,
            "pred_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
]

BAYES_RISK_AND_BAYES_RISK = [
    {
        "type": UncertaintyType.RISK,
        "print_name": "Predictive entropy 1",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.BAYES_RISK,
            "gt_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "print_name": "Predictive entropy 2",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.BAYES_RISK,
            "gt_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    # {
    #     "type": UncertaintyType.RISK,
    #     "print_name": "Predictive entropy 3",
    #     "kwargs": {
    #         "g_name": GName.LOG_SCORE,
    #         "risk_type": RiskType.BAYES_RISK,
    #         "gt_approx": ApproximationType.OUTER,
    #         "T": 1.0,
    #     },
    # },
    # {
    #     "type": UncertaintyType.RISK,
    #     "print_name": "Predictive entropy 4",
    #     "kwargs": {
    #         "g_name": GName.LOG_SCORE,
    #         "risk_type": RiskType.BAYES_RISK,
    #         "gt_approx": ApproximationType.OUTER,
    #         "T": 1.0,
    #     },
    # },
    # {
    #     "type": UncertaintyType.RISK,
    #     "print_name": "Predictive entropy 5",
    #     "kwargs": {
    #         "g_name": GName.LOG_SCORE,
    #         "risk_type": RiskType.BAYES_RISK,
    #         "gt_approx": ApproximationType.OUTER,
    #         "T": 1.0,
    #     },
    # },
]

MAHALANOBIS_AND_BAYES_RISK = [
    {
        "type": UncertaintyType.RISK,
        "print_name": "Predictive entropy",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.BAYES_RISK,
            "gt_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.MAHALANOBIS,
        "print_name": "Mahalanobis score",
        "kwargs": {},
    },
]

EXCESSES_DIFFERENT_INSTANTIATIONS = [
    {
        "type": UncertaintyType.RISK,
        "print_name": "EXC 1 1 (log)",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.OUTER,
            "pred_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "print_name": "EXC 1 1 (brier)",
        "kwargs": {
            "g_name": GName.BRIER_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.OUTER,
            "pred_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "print_name": "EXC 1 1 (spherical)",
        "kwargs": {
            "g_name": GName.SPHERICAL_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.OUTER,
            "pred_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "print_name": "EXC 1 1 (zero one)",
        "kwargs": {
            "g_name": GName.ZERO_ONE_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.OUTER,
            "pred_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
]

EXCESSES_DIFFERENT_APPROXIMATIONS = [
    {
        "type": UncertaintyType.RISK,
        "print_name": "EXC 1 1",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.OUTER,
            "pred_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "print_name": "EXC 2 1",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.INNER,
            "pred_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "print_name": "EXC 1 2",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.OUTER,
            "pred_approx": ApproximationType.INNER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "print_name": "EXC 1 3",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.OUTER,
            "pred_approx": ApproximationType.CENTRAL,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "print_name": "EXC 3 1",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.CENTRAL,
            "pred_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "print_name": "EXC 2 3",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.INNER,
            "pred_approx": ApproximationType.CENTRAL,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "print_name": "EXC 1 3",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.EXCESS_RISK,
            "gt_approx": ApproximationType.CENTRAL,
            "pred_approx": ApproximationType.INNER,
            "T": 1.0,
        },
    },
]


BAYES_DIFFERENT_APPROXIMATIONS = [
    {
        "type": UncertaintyType.RISK,
        "print_name": "B 1",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.BAYES_RISK,
            "gt_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "print_name": "B 2",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.BAYES_RISK,
            "gt_approx": ApproximationType.INNER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "print_name": "B 3",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.BAYES_RISK,
            "gt_approx": ApproximationType.CENTRAL,
            "T": 1.0,
        },
    },
]


BAYES_DIFFERENT_INSTANTIATIONS = [
    {
        "type": UncertaintyType.RISK,
        "print_name": "B 1 (log)",
        "kwargs": {
            "g_name": GName.LOG_SCORE,
            "risk_type": RiskType.BAYES_RISK,
            "gt_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "print_name": "B 1 (brier)",
        "kwargs": {
            "g_name": GName.BRIER_SCORE,
            "risk_type": RiskType.BAYES_RISK,
            "gt_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "print_name": "B 1 (spherical)",
        "kwargs": {
            "g_name": GName.SPHERICAL_SCORE,
            "risk_type": RiskType.BAYES_RISK,
            "gt_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
    {
        "type": UncertaintyType.RISK,
        "print_name": "B 1 (zero one)",
        "kwargs": {
            "g_name": GName.ZERO_ONE_SCORE,
            "risk_type": RiskType.BAYES_RISK,
            "gt_approx": ApproximationType.OUTER,
            "T": 1.0,
        },
    },
]
