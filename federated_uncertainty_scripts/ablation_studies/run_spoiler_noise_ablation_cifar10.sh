#!/bin/bash

seed=0

CUDA_VISIBLE_DEVICES=1 python3 ./federated_uncertainty_scripts/spoiler_noise_ablation.py \
    --n_models 30 \
    --n_clients 5 \
    --ensemble_size 3 \
    --lambda_disagreement 0.1 \
    --lambda_antireg 0.01 \
    --fraction 0.5 \
    --n_epochs 50 \
    --batch_size 256 \
    --lr 5e-4 \
    --spoiler_noise 6e-5 \
    --noise_type "noize_weights" \
    --market_lr 5e-3 \
    --market_epochs 2 \
    --model_min_classes 5 \
    --model_max_classes 8 \
    --client_min_classes 2 \
    --client_max_classes 5 \
    --save_dir "./data/saved_models/cifar10/spoiler_noise_ablation_study" \
    --dataset "cifar10" \
    --seed ${seed}
