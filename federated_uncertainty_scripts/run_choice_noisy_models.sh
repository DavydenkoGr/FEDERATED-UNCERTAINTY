clear
CUDA_VISIBLE_DEVICES=6 python3 ./federated_uncertainty_scripts/choice_noisy_models.py \
    --n_models 25 \
    --n_clients 5 \
    --ensemble_size 3 \
    --lambda_disagreement 0.1 \
    --lambda_antireg 0.01 \
    --fraction 0.5 \
    --n_epochs 4 \
    --batch_size 256 \
    --lr 1e-4 \
    --spoiler_noise 0 \
    --market_lr 0.5 \
    --market_epochs 50 \
    --save_dir "./data/saved_models/n_models=2,lambda_antireg=0.01"
