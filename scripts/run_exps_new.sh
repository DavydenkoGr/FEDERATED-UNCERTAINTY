uv run python scripts/full_evaluation.py --entropic_target Exp --entropic_eps 0.5 --entropic_scaling_type FeatureWise --output_file ./resources/refactored/benchmark.csv --verbose
uv run python scripts/full_evaluation.py --entropic_target Exp --entropic_eps 0.5 --entropic_scaling_type Global --output_file ./resources/refactored/benchmark.csv --verbose
uv run python scripts/full_evaluation.py --entropic_target Beta --entropic_eps 0.5 --entropic_scaling_type FeatureWise --output_file ./resources/refactored/benchmark.csv --verbose
uv run python scripts/full_evaluation.py --entropic_target Beta --entropic_eps 0.5 --entropic_scaling_type Global --output_file ./resources/refactored/benchmark.csv --verbose
