clear
CUDA_VISIBLE_DEVICES=1 python3 ./federated_uncertainty_scripts/compute_measures_ood.py \
    --n_clients 5 \
    --data_path "./data/saved_models/lambda_antireg=0.01"
