# Обучить ансамбль из 10 CNN моделей
CUDA_VISIBLE_DEVICES=1 python3 ./federated_uncertainty_scripts/regression/main_mosaic_regression.py \
    --network cnn \
    --num_networks 1 \
    --epochs 50 \
    --batch_size 256 \
    --lr 1e-4 \
    --output_dir "./data/saved_models/regression/test/test2"

# Обучить ResNet модель
# python main_mosaic_regression.py \
#     --network resnet \
#     --num_networks 1 \
#     --epochs 50 \
#     --data_root ./data \
#     --output_dir ./results/mosaic_regression