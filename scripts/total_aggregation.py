from mdu.eval.eval_utils import load_pickle
import numpy as np
import torch
from collections import defaultdict
from mdu.data.constants import DatasetName
from sklearn.metrics import roc_auc_score
from mdu.data.data_utils import split_dataset_indices
from mdu.unc.constants import VectorQuantileModel, UncertaintyType
from mdu.unc.multidimensional_uncertainty import MultiDimensionalUncertainty
from mdu.unc.risk_metrics.constants import GName, RiskType, ApproximationType
import pandas as pd
from mdu.randomness import set_all_seeds

# Set random seed for reproducibility
seed = 42
set_all_seeds(seed)

# Dataset pairs: (in-distribution, out-of-distribution)
DATASET_PAIRS = [
    (DatasetName.CIFAR10.value, DatasetName.CIFAR100.value),
    (DatasetName.CIFAR10.value, DatasetName.SVHN.value),
    (DatasetName.CIFAR10.value, DatasetName.TINY_IMAGENET.value),
    (DatasetName.CIFAR10.value, DatasetName.CIFAR10C_1.value),
    (DatasetName.CIFAR10.value, DatasetName.CIFAR10C_2.value),
    (DatasetName.CIFAR10.value, DatasetName.CIFAR10C_3.value),
    (DatasetName.CIFAR10.value, DatasetName.CIFAR10C_4.value),
    (DatasetName.CIFAR10.value, DatasetName.CIFAR10C_5.value),
    (DatasetName.CIFAR100.value, DatasetName.CIFAR10.value),
    (DatasetName.CIFAR100.value, DatasetName.SVHN.value),
    (DatasetName.CIFAR100.value, DatasetName.TINY_IMAGENET.value),
    (DatasetName.CIFAR100.value, DatasetName.CIFAR10C_1.value),
    (DatasetName.CIFAR100.value, DatasetName.CIFAR10C_2.value),
    (DatasetName.CIFAR100.value, DatasetName.CIFAR10C_3.value),
    (DatasetName.CIFAR100.value, DatasetName.CIFAR10C_4.value),
    (DatasetName.CIFAR100.value, DatasetName.CIFAR10C_5.value),
]

# Available G-functions
GNAMES = [GName.LOG_SCORE, GName.BRIER_SCORE, GName.SPHERICAL_SCORE, GName.ZERO_ONE_SCORE]

# Ensemble groups for model aggregation
ENSEMBLE_GROUPS = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
]

# Multidimensional model configuration
MULTIDIM_MODEL = VectorQuantileModel.ENTROPIC_OT
device = torch.device("cuda:0")

train_kwargs = {
    "batch_size": 64,
    "device": device,
}
multidim_params = {
    "target": "exp",
    "standardize": False,
    "fit_mse_params": False,
    "eps": 0.07,
    "max_iters": 2000,
    "tol": 1e-8,
    "random_state": seed,
}

def create_uncertainty_measures_pair(gname):
    """
    Create a pair of uncertainty measures for a given G-function:
    1. EXCESS_RISK measure
    2. BAYES_RISK measure
    """
    return [
        {
            "type": UncertaintyType.RISK,
            "print_name": f"EXC 1 1 ({gname.value.lower()})",
            "kwargs": {
                "g_name": gname,
                "risk_type": RiskType.EXCESS_RISK,
                "gt_approx": ApproximationType.OUTER,
                "pred_approx": ApproximationType.OUTER,
                "T": 1.0,
            },
        },
        {
            "type": UncertaintyType.RISK,
            "print_name": f"BAYES 1 ({gname.value.lower()})",
            "kwargs": {
                "g_name": gname,
                "risk_type": RiskType.BAYES_RISK,
                "gt_approx": ApproximationType.OUTER,
                "T": 1.0,
            },
        },
    ]

def create_total_risk_measure(gname):
    """
    Create a single TOTAL_RISK measure for a given G-function
    """
    return [
        {
            "type": UncertaintyType.RISK,
            "print_name": f"TOT 1 1 ({gname.value.lower()})",
            "kwargs": {
                "g_name": gname,
                "risk_type": RiskType.TOTAL_RISK,
                "gt_approx": ApproximationType.OUTER,
                "pred_approx": ApproximationType.OUTER,
                "T": 1.0,
            },
        },
    ]

def evaluate_uncertainty_measures(ind_dataset, ood_dataset, uncertainty_measures):
    """
    Evaluate uncertainty measures for a given dataset pair
    """
    results = defaultdict(list)
    
    for group in ENSEMBLE_GROUPS:
        all_ind_logits = []
        all_ood_logits = []
        
        for model_id in group:
            ind_res = load_pickle(
                f"./model_weights/{ind_dataset}/checkpoints/resnet18/CrossEntropy/{model_id}/{ind_dataset}.pkl"
            )
            ood_res = load_pickle(
                f"./model_weights/{ind_dataset}/checkpoints/resnet18/CrossEntropy/{model_id}/{ood_dataset}.pkl"
            )

            all_ind_logits.append(ind_res["embeddings"][None])
            all_ood_logits.append(ood_res["embeddings"][None])

        y_ind = ind_res["labels"]
        y_ood = ood_res["labels"]

        # Split dataset indices
        _, train_cond_idx, calib_idx, test_idx = split_dataset_indices(
            ind_res["embeddings"],
            y_ind,
            train_ratio=0.0,
            calib_ratio=0.1,
            test_ratio=0.8,
            random_state=seed
        )

        y_train_cond = y_ind[train_cond_idx]
        y_calib = y_ind[calib_idx]

        X_train_cond = np.vstack(all_ind_logits)[:, train_cond_idx, :]
        X_calib = np.vstack(all_ind_logits)[:, calib_idx, :]
        X_test = np.vstack(all_ind_logits)[:, test_idx, :]
        X_ood = np.vstack(all_ood_logits)

        # Update feature dimension for CPFLOW
        if MULTIDIM_MODEL == VectorQuantileModel.CPFLOW:
            multidim_params["feature_dimension"] = len(uncertainty_measures)

        # Initialize and fit multidimensional uncertainty model
        multi_dim_uncertainty = MultiDimensionalUncertainty(
            uncertainty_measures,
            multidim_model=MULTIDIM_MODEL,
            multidim_params=multidim_params,
            if_add_maximal_elements=True,
        )
        
        multi_dim_uncertainty.fit(
            logits_train=X_train_cond,
            y_train=y_train_cond,
            logits_calib=X_calib,
            train_kwargs=train_kwargs,
        )

        # Convert to torch tensors if needed
        if MULTIDIM_MODEL == VectorQuantileModel.CPFLOW:
            X_test = torch.from_numpy(X_test).to(torch.float32).to(train_kwargs["device"])
            X_ood = torch.from_numpy(X_ood).to(torch.float32).to(train_kwargs["device"])

        # Get predictions
        _, uncertainty_scores_ind = multi_dim_uncertainty.predict(X_test)
        _, uncertainty_scores_ood = multi_dim_uncertainty.predict(X_ood)

        # Compute ROC AUC for each measure
        for measure_name in uncertainty_scores_ind.keys():
            ind_scores = uncertainty_scores_ind[measure_name]
            ood_scores = uncertainty_scores_ood[measure_name]

            # Concatenate scores and labels
            all_scores = np.concatenate([ind_scores, ood_scores])
            all_labels = np.concatenate([
                np.zeros_like(ind_scores),  # class 0: in-distribution
                np.ones_like(ood_scores),   # class 1: OOD
            ])

            auc = roc_auc_score(all_labels, all_scores)
            results[measure_name].append(auc)
    
    return results

def main():
    """
    Main function to run the total aggregation experiment
    """
    all_results = []
    
    # Iterate over dataset pairs
    for ind_dataset, ood_dataset in DATASET_PAIRS:
        print(f"\nProcessing {ind_dataset} -> {ood_dataset}")
        
        # Iterate over G-functions
        for gname in GNAMES:
            print(f"  Processing G-function: {gname.value}")
            
            # Test 1: EXCESS_RISK + BAYES_RISK pair
            print(f"    Testing EXCESS_RISK + BAYES_RISK pair")
            uncertainty_measures_pair = create_uncertainty_measures_pair(gname)
            results_pair = evaluate_uncertainty_measures(ind_dataset, ood_dataset, uncertainty_measures_pair)
            
            # Add results for pair measures
            for measure_name, aucs in results_pair.items():
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                
                # Extract measure characteristics
                if "EXC" in measure_name:
                    risk_type = "EXCESS_RISK"
                elif "BAYES" in measure_name:
                    risk_type = "BAYES_RISK"
                else:
                    risk_type = "UNKNOWN"
                
                all_results.append({
                    "In-distribution": ind_dataset,
                    "Out-of-distribution": ood_dataset,
                    "G-function": gname.value,
                    "Risk_type": risk_type,
                    "Measure_type": "PAIR",
                    "Measure_name": measure_name,
                    "ROC_AUC_Scores": aucs,
                    "Mean_ROC_AUC": mean_auc,
                    "Std_ROC_AUC": std_auc,
                })
            
            # Test 2: TOTAL_RISK single measure
            print(f"    Testing TOTAL_RISK single measure")
            uncertainty_measures_total = create_total_risk_measure(gname)
            results_total = evaluate_uncertainty_measures(ind_dataset, ood_dataset, uncertainty_measures_total)
            
            # Add results for total risk measure
            for measure_name, aucs in results_total.items():
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                
                all_results.append({
                    "In-distribution": ind_dataset,
                    "Out-of-distribution": ood_dataset,
                    "G-function": gname.value,
                    "Risk_type": "TOTAL_RISK",
                    "Measure_type": "SINGLE",
                    "Measure_name": measure_name,
                    "ROC_AUC_Scores": aucs,
                    "Mean_ROC_AUC": mean_auc,
                    "Std_ROC_AUC": std_auc,
                })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    column_order = [
        "In-distribution", "Out-of-distribution", "G-function", "Risk_type", 
        "Measure_type", "Measure_name", "Mean_ROC_AUC", "Std_ROC_AUC", "ROC_AUC_Scores"
    ]
    df = df[column_order]
    
    # Save results
    output_filename = "total_11_eps007_aggregation_results.csv"
    df.to_csv(output_filename, index=False)
    
    print(f"\nResults saved to {output_filename}")
    print(f"Total experiments: {len(df)}")
    
    # Print summary
    print("\nSummary of results:")
    print("=" * 80)
    print(df.groupby(['G-function', 'Risk_type'])['Mean_ROC_AUC'].mean().round(4))
    
    return df

if __name__ == "__main__":
    results_df = main()
