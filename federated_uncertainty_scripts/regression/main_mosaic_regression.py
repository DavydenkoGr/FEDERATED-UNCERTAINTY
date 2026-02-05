#!/usr/bin/env python3
"""
Training script for mosaic regression task.

This script trains regression models on 2x2 mosaic datasets created from MNIST digits.
Each mosaic consists of 4 digits arranged in a 2x2 grid, and the regression target
is the 4-digit number formed by reading the digits (top-left, top-right, bottom-left, bottom-right).

Example usage:
    python main_mosaic_regression.py --network cnn --num_networks 10 --epochs 100
    python main_mosaic_regression.py --network resnet --method ensemble --n_id_train 50000
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Project root = parent of "federated_uncertainty_scripts"
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated_uncertainty.regression_uncertainty.datasets.mosaic_datasets import build_id_and_ood
from federated_uncertainty.regression_uncertainty.source.models.cnn import get_cnn
from federated_uncertainty.regression_uncertainty.source.models.resnet import get_resnet18
from federated_uncertainty.regression_uncertainty.source.trainer import fit
from federated_uncertainty.regression_uncertainty.source.utils.seeding import fix_seeds


def main():
    parser = argparse.ArgumentParser(description="Train regression model on mosaic datasets")
    
    # General arguments
    parser.add_argument("--network", type=str, default="cnn", choices=["cnn", "resnet"],
                        help="Network architecture: cnn or resnet")
    parser.add_argument("--method", type=str, default="ensemble", choices=["ensemble", "mc_dropout"],
                        help="Training method: ensemble or mc_dropout")
    parser.add_argument("--method_seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (cuda:0, cpu, etc.)")
    
    # Data arguments
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--tile_size", type=int, default=32,
                        help="Size of each tile in the mosaic (default: 32, resulting in 64x64 images)")
    parser.add_argument("--n_id_train", type=int, default=100000,
                        help="Number of training samples")
    parser.add_argument("--n_id_test", type=int, default=1000,
                        help="Number of test samples")
    parser.add_argument("--n_ood_each", type=int, default=1000,
                        help="Number of OOD samples from each source")
    
    # Training arguments
    parser.add_argument("--use_natural", action="store_true",
                        help="Use natural parametrization")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = no early stopping)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of data loading workers")
    
    # Ensemble arguments
    parser.add_argument("--num_networks", type=int, default=10,
                        help="Number of networks in ensemble")
    
    # Loss arguments
    parser.add_argument("--anti_regularization_weight", type=float, default=0.0,
                        help="Anti-regularization weight")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./results/mosaic_regression",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 80)
    print("Mosaic Regression Training Configuration")
    print("=" * 80)
    print(f"Network: {args.network}")
    print(f"Method: {args.method}")
    print(f"Seed: {args.method_seed}")
    print(f"Device: {args.device}")
    print(f"Data root: {args.data_root}")
    print(f"Tile size: {args.tile_size} (image size: {2*args.tile_size}x{2*args.tile_size})")
    print(f"Training samples: {args.n_id_train}")
    print(f"Test samples: {args.n_id_test}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Number of networks: {args.num_networks if args.method == 'ensemble' else 1}")
    print("=" * 80)
    
    # Create output directory
    run_path = os.path.join(
        args.output_dir,
        f"mosaic_{args.network}_natural{args.use_natural}_seed{args.method_seed}_arw{args.anti_regularization_weight}"
    )
    os.makedirs(run_path, exist_ok=True)
    
    # Save arguments
    formatted_args = "\n".join(f"{key}: {value}" for key, value in vars(args).items())
    with open(os.path.join(run_path, "args.txt"), "w") as f:
        f.write(formatted_args)
    print(f"Arguments saved to {os.path.join(run_path, 'args.txt')}")
    
    # Fix seeds for reproducibility
    fix_seeds(seed=args.method_seed)
    
    # Load mosaic datasets
    print("\nLoading mosaic datasets...")
    id_train, id_test, ood_fashion, ood_cifar10, ood_svhn, ood_mixture = build_id_and_ood(
        root=args.data_root,
        tile_size=args.tile_size,
        seed=args.method_seed,
        n_id_train=args.n_id_train,
        n_id_test=args.n_id_test,
        n_ood_each=args.n_ood_each,
        download=True,
        normalize_images=True,
    )
    
    # Create validation set (using different seed)
    _, id_val, _, _, _, _ = build_id_and_ood(
        root=args.data_root,
        tile_size=args.tile_size,
        seed=args.method_seed + 1,
        n_id_train=args.n_id_train,
        n_id_test=args.n_id_test,
        n_ood_each=args.n_ood_each,
        download=True,
        normalize_images=True,
    )
    
    print(f"Training set size: {len(id_train)}")
    print(f"Validation set size: {len(id_val)}")
    print(f"Test set size: {len(id_test)}")
    
    # Check data shapes
    x_sample, y_sample = id_train[0]
    print(f"Input shape: {x_sample.shape}")
    print(f"Target shape: {y_sample.shape}")
    print(f"Sample targets: {[id_train[i][1].item() for i in range(5)]}")
    
    # Model configuration
    model_kwargs = {
        'in_channels': 1,  # Grayscale images
        'image_size': 2 * args.tile_size,  # 64x64 for tile_size=32
    }
    
    # Create data loaders
    train_loader = DataLoader(
        id_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        id_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
        
    # Determine number of networks
    n_networks = 1 if args.method == "mc_dropout" else args.num_networks
    
    # Train models
    print(f"\nTraining {n_networks} network(s)...")
    print("-" * 80)
    
    for n in range(n_networks):
        print(f"\nTraining model {n+1}/{n_networks}")
        
        # Create model
        if args.network == "cnn":
            network = get_cnn(**model_kwargs)
        elif args.network == "resnet":
            network = get_resnet18(**model_kwargs)
        else:
            raise ValueError(f"Unknown network: {args.network}")
        
        network.to(args.device)
        network.train()
        
        print(f"Model parameters: {sum(p.numel() for p in network.parameters()):,}")
        
        # Train model
        best_model, train_losses, val_perfs = fit(
            network=network,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            use_natural=args.use_natural,
            anti_regularization_weight=args.anti_regularization_weight,
            verbose=True,
        )
        
        # Save model
        models_dir = os.path.join(run_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"model_{n}.pt")
        torch.save(best_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Save training history
        val_perfs_dir = os.path.join(run_path, "val_perfs")
        os.makedirs(val_perfs_dir, exist_ok=True)
        
        # Save full validation performance history
        with open(os.path.join(val_perfs_dir, f"model_{n}.txt"), "w") as f:
            f.write("\n".join(map(str, val_perfs)))
        
        # Save best validation performance
        best_val_perf = min(val_perfs)
        if n == 0:
            # Create or overwrite the summary file
            with open(os.path.join(run_path, "val_perfs.txt"), "w") as f:
                f.write(f"Model {n}: {best_val_perf:.6f}\n")
        else:
            # Append to the summary file
            with open(os.path.join(run_path, "val_perfs.txt"), "a") as f:
                f.write(f"Model {n}: {best_val_perf:.6f}\n")
        
        print(f"Best validation performance: {best_val_perf:.6f}")
        print("-" * 80)
    
    print(f"\nTraining completed! Results saved to: {run_path}")
    print(f"Models: {os.path.join(run_path, 'models')}")
    print(f"Validation performances: {os.path.join(run_path, 'val_perfs')}")


if __name__ == "__main__":
    main()
