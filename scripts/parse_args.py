import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Full evaluation of uncertainty measures"
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="./resources/results_cleaned",
        help="Root directory for uncertainty measure results",
    )
    parser.add_argument(
        "--weights_root",
        type=str,
        default="./resources/model_weights",
        help="Root directory for model weights",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./full_evaluation_results.csv",
        help="Output CSV file path",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # EntropicOT hyperparameters
    parser.add_argument(
        "--entropic_target",
        type=str,
        default="exp",
        help="EntropicOT target function (default: exp)",
    )
    parser.add_argument(
        "--entropic_eps",
        type=float,
        default=0.5,
        help="EntropicOT epsilon parameter (default: 0.5)",
    )
    parser.add_argument(
        "--entropic_max_iters",
        type=int,
        default=1000,
        help="EntropicOT maximum iterations (default: 1000)",
    )
    parser.add_argument(
        "--entropic_tol",
        type=float,
        default=1e-6,
        help="EntropicOT tolerance (default: 1e-6)",
    )
    parser.add_argument(
        "--entropic_random_state",
        type=int,
        default=42,
        help="EntropicOT random state (default: 42)",
    )
    parser.add_argument(
        "--entropic_grid_size",
        type=int,
        default=5,
        help="EntropicOT grid size (default: 5)",
    )
    parser.add_argument(
        "--entropic_scaling_type",
        type=str,
        default="FeatureWise",
        choices=["FeatureWise", "Global"],
        help="Scaler type: FeatureWise (default MinMaxScaler), Global (GlobalMinMaxScaler)",
    )
    parser.add_argument(
        "--entropic_sampling_method",
        type=str,
        default="Grid",
        choices=["Random", "Sobol", "Grid"],
        help="Sampling method: Grid (default), Sobol, Random",
    )
    parser.add_argument(
        "--entropic_n_targets_multiplier",
        type=int,
        default=1,
        help="EntropicOT number of targets multiplier (default: 1)",
    )
    # Keep backward compatibility

    args = parser.parse_args()
    return args
