#!/bin/bash

for seed in 0; do
    CUDA_VISIBLE_DEVICES=0 python3 ./federated_uncertainty_scripts/regression/main_federated_mosaic_regression.py \
        --n_models 10 \
        --n_clients 3 \
        --ensemble_size 3 \
        --lambda_disagreement 0.1 \
        --lambda_antireg 0.01 \
        --fraction 0.5 \
        --n_epochs 10 \
        --batch_size 256 \
        --lr 5e-4 \
        --market_lr 5e-6 \
        --market_epochs 1 \
        --model_min_digits 8 \
        --model_max_digits 8 \
        --client_min_digits 4 \
        --client_max_digits 5 \
        --model_pool_split_ratio 0.6 \
        --network "cnn" \
        --tile_size 32 \
        --n_id_train 100000 \
        --n_id_test 10000 \
        --data_root "./data" \
        --save_dir "./data/saved_models/regression/test_mosaic2" \
        --seed $seed
done
