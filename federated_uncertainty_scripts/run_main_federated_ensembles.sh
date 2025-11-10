clear
CUDA_VISIBLE_DEVICES=6 uv run python ./federated_uncertainty_scripts/main_federated_ensembles.py \
    --n_models 20 \
    --n_clients 5 \
    --ensemble_selection_size 3 \
    --lambda_disagreement 0.1 \
    --lambda_antireg 0.01 \
    --fraction 0.25 \
    --n_epochs 5 \
    --batch_size 128 \
    --lr 1e-3
