clear
CUDA_VISIBLE_DEVICES=6 python3 ./federated_uncertainty_scripts/main_federated_ensembles.py \
    --n_models 3 \
    --n_clients 3 \
    --ensemble_size 1 \
    --lambda_disagreement 0.1 \
    --lambda_antireg 0.01 \
    --fraction 0.05 \
    --n_epochs 5 \
    --batch_size 128 \
    --lr 1e-4 \
    --spoiler_noise 1e-7 \
    --market_lr 0.5 \
    --market_epochs 50 \
    --model_min_classes 10 \
    --model_max_classes 10 \
    --save_dir "./data/saved_models/all_classes" \
    --dataset "tiny-imagenet"
