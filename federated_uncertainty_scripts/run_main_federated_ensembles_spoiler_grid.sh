clear
for spoiler_noise in 1e-6 5e-6 1e-6 5e-7 1e-7 5e-8 1e-8 5e-9 1e-9; do
    CUDA_VISIBLE_DEVICES=2 python3 ./federated_uncertainty_scripts/main_federated_ensembles.py \
        --n_models 30 \
        --n_clients 5 \
        --ensemble_size 3 \
        --lambda_disagreement 0.1 \
        --lambda_antireg 0.01 \
        --fraction 0.5 \
        --n_epochs 50 \
        --batch_size 256 \
        --lr 1e-4 \
        --spoiler_noise $spoiler_noise \
        --noise_type "noize_weights" \
        --market_lr 0.05 \
        --market_epochs 2 \
        --model_min_classes 5 \
        --model_max_classes 8 \
        --save_dir "./data/saved_models/cifar10/spoiler_noise_search/spoiler_noise_${spoiler_noise}" \
        --dataset "cifar10"
done