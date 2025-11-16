clear
CUDA_VISIBLE_DEVICES=2 python3 ./federated_uncertainty_scripts/main_federated_ensembles.py \
    --n_models 30 \
    --n_clients 5 \
    --ensemble_selection_size 3 \
    --lambda_disagreement 0.1 \
    --lambda_antireg 0 \
    --fraction 0.5 \
    --n_epochs 5 \
    --batch_size 128 \
    --lr 1e-4 \
    --save_dir "./data/saved_models/run_20251116_132306"
