clear
CUDA_VISIBLE_DEVICES=2 python3 ./federated_uncertainty_scripts/main_federated_ensembles.py \
    --n_models 50 \
    --n_clients 5 \
    --ensemble_size 5 \
    --lambda_disagreement 0.1 \
    --lambda_antireg 0.01 \
    --fraction 0.5 \
    --n_epochs 20 \
    --batch_size 128 \
    --lr 1e-4 \
    --spoiler_noise 0 \
    --market_lr 0.5 \
    --market_epochs 50 \
    --save_dir "./data/saved_models/n_models=50,lambda_antireg=0.01"
