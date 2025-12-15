clear
CUDA_VISIBLE_DEVICES=1 python3 ./federated_uncertainty_scripts/compute_metrics.py \
    --n_clients 3 \
    --data_path "./data/saved_models/all_classes"
