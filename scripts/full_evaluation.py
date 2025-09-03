import sys
from pathlib import Path

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdu.unc.risk_metrics import RiskType, GName, ApproximationType
from mdu.unc.constants import UncertaintyType
from mdu.data.constants import DatasetName
from mdu.eval.eval_utils import load_pickle
from mdu.data.data_utils import split_dataset_indices

from typing import Optional
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')


def get_results_path(
    ind_dataset_: DatasetName,
    ood_dataset_: DatasetName,
    uncertainty_type_: UncertaintyType,
    gname_: Optional[GName] = None,
    risk_type_: Optional[RiskType] = None,
    gt_approximation_: Optional[ApproximationType] = None,
    pred_approximation_: Optional[ApproximationType] = None,
    results_root: str = "./resources/results"
):
    """Get path to uncertainty measure results file"""
    ind_dataset = ind_dataset_.value.lower()
    ood_dataset = ood_dataset_.value.lower()
    uncertainty_type = uncertainty_type_.value.lower()
    T = 1.0

    if uncertainty_type_ is UncertaintyType.RISK:
        proper_scoring_rule = gname_.value.lower()
        risk_type = risk_type_.value.lower()
        gt_approximation = gt_approximation_.value.lower()
        pred_approximation = pred_approximation_.value.lower()
        folder_path = f"{results_root}/{ind_dataset}/{uncertainty_type}_{proper_scoring_rule}_{risk_type}_{gt_approximation}_{pred_approximation}_T_{T}/{ood_dataset}"
        file_name = f"{ind_dataset}_{ood_dataset}_{uncertainty_type}_{proper_scoring_rule}_{risk_type}_{gt_approximation}_{pred_approximation}_T_{T}.npz"
    else:
        folder_path = f"{results_root}/{ind_dataset}/{uncertainty_type}/{ood_dataset}"
        file_name = f"{ind_dataset}_{ood_dataset}_{uncertainty_type}.npz"

    return f"{folder_path}/{file_name}"


def load_predictions_and_split(
    ind_dataset_: DatasetName,
    weights_root: str = "./resources/model_weights",
    ensemble_groups: list = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
    train_ratio: float = 0.0,
    calib_ratio: float = 0.1,
    test_ratio: float = 0.8,
    random_state: int = 42
):
    """Load predictions from pickle files and split data"""
    ind_dataset = ind_dataset_.value
    
    results_by_group = {}
    
    for group_idx, group in enumerate(ensemble_groups):
        # Load logits from ensemble models in this group
        all_ind_logits = []
        for model_id in group:
            eval_res = load_pickle(
                f"{weights_root}/{ind_dataset}/checkpoints/resnet18/CrossEntropy/{model_id}/{ind_dataset}.pkl"
            )
            all_ind_logits.append(eval_res["embeddings"][None])
        
        y_true = eval_res["labels"]
        logits_ind = eval_res["embeddings"]  # Use last model's logits for splitting
        
        # Split dataset indices (same split for all groups)
        _, train_cond_idx, calib_idx, test_idx = split_dataset_indices(
            logits_ind,
            y_true,
            train_ratio=train_ratio,
            calib_ratio=calib_ratio,
            test_ratio=test_ratio,
            random_state=random_state
        )
        
        # Split labels
        y_train_cond = y_true[train_cond_idx]
        y_calib = y_true[calib_idx]
        y_test = y_true[test_idx]
        
        # Split features (logits from ensemble)
        ensemble_logits = np.vstack(all_ind_logits)  # Shape: (n_models_in_group, n_samples, n_classes)
        X_train_cond = ensemble_logits[:, train_cond_idx, :]
        X_calib = ensemble_logits[:, calib_idx, :]
        X_test = ensemble_logits[:, test_idx, :]
        
        # Compute ensemble predictions for this group
        y_pred = np.argmax(np.mean(X_test, axis=0), axis=-1)
        ensemble_accuracy = np.mean(y_pred == y_test)
        
        results_by_group[f'group_{group_idx}'] = {
            'group_models': group,
            'X_train_cond': X_train_cond,
            'X_calib': X_calib, 
            'X_test': X_test,
            'y_train_cond': y_train_cond,
            'y_calib': y_calib,
            'y_test': y_test,
            'y_pred': y_pred,
            'ensemble_accuracy': ensemble_accuracy,
        }
    
    return results_by_group


def compute_ood_detection_metrics(ind_scores, ood_scores):
    """Compute OOD detection metrics using ROC AUC"""
    # Combine scores and labels (0 for in-distribution, 1 for OOD)
    all_scores = np.concatenate([ind_scores, ood_scores])
    all_labels = np.concatenate([
        np.zeros_like(ind_scores),  # 0: in-distribution
        np.ones_like(ood_scores)    # 1: OOD
    ])
    
    # Compute ROC AUC
    roc_auc = roc_auc_score(all_labels, all_scores)
    
    return {
        'roc_auc': roc_auc,
        'n_ind_samples': len(ind_scores),
        'n_ood_samples': len(ood_scores)
    }


def compute_misclassification_detection_metrics(uncertainty_scores, y_pred, y_true):
    """Compute misclassification detection metrics"""
    # Create binary labels: 0 for correct, 1 for incorrect
    correct_mask = (y_pred == y_true)
    incorrect_mask = ~correct_mask
    
    correct_scores = uncertainty_scores[correct_mask]
    incorrect_scores = uncertainty_scores[incorrect_mask]
    
    # Combine scores and labels
    all_scores = np.concatenate([correct_scores, incorrect_scores])
    all_labels = np.concatenate([
        np.zeros_like(correct_scores),  # 0: correct predictions
        np.ones_like(incorrect_scores)  # 1: incorrect predictions
    ])
    
    # Compute ROC AUC
    roc_auc = roc_auc_score(all_labels, all_scores)
    
    # Compute Average Precision (AP) for error detection
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    
    return {
        'roc_auc': roc_auc,
        'average_precision': ap,
        'accuracy': np.mean(correct_mask),
        'n_correct': len(correct_scores),
        'n_incorrect': len(incorrect_scores)
    }


def compute_selective_prediction_metrics(uncertainty_scores, y_pred, y_true):
    """Compute selective prediction metrics including AURC"""
    n = len(uncertainty_scores)
    
    # Sort by uncertainty (low uncertainty first for selective prediction)
    order = np.argsort(uncertainty_scores)
    correct = (y_pred == y_true).astype(int)[order]
    
    # Compute coverage and accuracy curves
    coverage = np.arange(1, n+1) / n
    accuracy = np.cumsum(correct) / np.arange(1, n+1)
    risk = 1 - accuracy  # Risk = 1 - Accuracy
    
    # Compute AURC (Area Under Risk-Coverage curve) - lower is better
    aurc = np.trapz(risk, coverage)
    
    # Compute AUC for accuracy-coverage curve - higher is better
    acc_cov_auc = np.trapz(accuracy, coverage)
    
    # Compute coverage at different error rates
    coverage_at_error = {}
    for error_rate in [0.01, 0.02, 0.05]:  # 1%, 2%, 5% error rates
        target_accuracy = 1 - error_rate
        # Find first point where accuracy >= target_accuracy
        valid_indices = np.where(accuracy >= target_accuracy)[0]
        if len(valid_indices) > 0:
            coverage_at_error[f'{int(error_rate*100)}%err'] = coverage[valid_indices[0]]
        else:
            coverage_at_error[f'{int(error_rate*100)}%err'] = 1.0  # Need all data
    
    overall_accuracy = np.mean(y_pred == y_true)
    
    return {
        'aurc': aurc,
        'acc_cov_auc': acc_cov_auc,
        'overall_accuracy': overall_accuracy,
        'coverage_at_1pct_error': coverage_at_error.get('1%err', 1.0),
        'coverage_at_2pct_error': coverage_at_error.get('2%err', 1.0),
        'coverage_at_5pct_error': coverage_at_error.get('5%err', 1.0),
        'n_samples': n
    }


def create_measure_identifier(uncertainty_type, gname=None, risk_type=None, gt_approx=None, pred_approx=None):
    """Create a unique identifier for each uncertainty measure"""
    if uncertainty_type == UncertaintyType.RISK:
        return f"{uncertainty_type.value}_{gname.value}_{risk_type.value}_{gt_approx.value}_{pred_approx.value}"
    elif uncertainty_type == UncertaintyType.MAHALANOBIS:
        return uncertainty_type.value
    elif uncertainty_type == UncertaintyType.GMM:
        return uncertainty_type.value
    else:
        return "combination"


def main():
    parser = argparse.ArgumentParser(description='Full evaluation of uncertainty measures')
    parser.add_argument('--results_root', type=str, default='./resources/results',
                       help='Root directory for uncertainty measure results')
    parser.add_argument('--weights_root', type=str, default='./resources/model_weights',
                       help='Root directory for model weights')
    parser.add_argument('--output_file', type=str, default='./full_evaluation_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Define all possible combinations
    datasets_ind = [DatasetName.CIFAR10, DatasetName.CIFAR100]
    datasets_ood = [DatasetName.CIFAR10, DatasetName.CIFAR100, DatasetName.TINY_IMAGENET, DatasetName.SVHN]
    uncertainty_types = [UncertaintyType.RISK, UncertaintyType.MAHALANOBIS, UncertaintyType.GMM]
    gnames = [GName.LOG_SCORE, GName.BRIER_SCORE, GName.ZERO_ONE_SCORE, GName.SPHERICAL_SCORE]
    risk_types = [RiskType.TOTAL_RISK, RiskType.EXCESS_RISK, RiskType.BAYES_RISK]
    approximations = [ApproximationType.OUTER, ApproximationType.INNER, ApproximationType.CENTRAL]
    
    results = []
    
    # Load predictions for all ind_datasets once
    print("Loading predictions for all datasets...")
    prediction_data = {}
    for ind_dataset in datasets_ind:
        try:
            pred_data = load_predictions_and_split(ind_dataset, weights_root=args.weights_root)
            prediction_data[ind_dataset] = pred_data
            if args.verbose:
                print(f"✓ Loaded predictions for {ind_dataset.value}")
        except Exception as e:
            print(f"✗ Failed to load predictions for {ind_dataset.value}: {e}")
            prediction_data[ind_dataset] = None
    
    # Create progress bar
    total_combinations = 0
    for ind_dataset, ood_dataset in product(datasets_ind, datasets_ood):
        for uncertainty_type in uncertainty_types:
            if uncertainty_type == UncertaintyType.RISK:
                for gname, risk_type, gt_approx in product(gnames, risk_types, approximations):
                    if risk_type == RiskType.BAYES_RISK:
                        total_combinations += 1
                    else:
                        total_combinations += len(approximations)
            else:
                total_combinations += 1
    
    pbar = tqdm(total=total_combinations, desc="Processing combinations")
    
    # Main evaluation loop
    processed_same_dataset = set()  # Track which ind_datasets we've processed for misclassification/selective
    
    for ind_dataset, ood_dataset in product(datasets_ind, datasets_ood):
        for uncertainty_type in uncertainty_types:
            if uncertainty_type == UncertaintyType.RISK:
                # Test all risk combinations
                for gname, risk_type, gt_approx in product(gnames, risk_types, approximations):
                    if risk_type == RiskType.BAYES_RISK:
                        # BayesRisk doesn't use pred_approximation
                        pred_approx = ApproximationType.CENTRAL  # dummy value
                        process_uncertainty_measure(
                            ind_dataset, ood_dataset, uncertainty_type, gname, risk_type, 
                            gt_approx, pred_approx, prediction_data, results, args, pbar,
                            processed_same_dataset
                        )
                    else:
                        # TotalRisk and ExcessRisk use pred_approximation
                        for pred_approx in approximations:
                            process_uncertainty_measure(
                                ind_dataset, ood_dataset, uncertainty_type, gname, risk_type, 
                                gt_approx, pred_approx, prediction_data, results, args, pbar,
                                processed_same_dataset
                            )
            else:
                # Mahalanobis and GMM
                process_uncertainty_measure(
                    ind_dataset, ood_dataset, uncertainty_type, None, None, 
                    None, None, prediction_data, results, args, pbar,
                    processed_same_dataset
                )
    
    pbar.close()
    
    # Convert results to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    print(f"\nResults saved to {args.output_file}")
    print(f"Total rows: {len(df)}")
    
    # Print summary statistics
    print("\n=== SUMMARY ===")
    print(f"Unique ind_datasets: {df['ind_dataset'].nunique()}")
    print(f"Unique ood_datasets: {df['ood_dataset'].nunique()}")
    print(f"Unique measures: {df['measure'].nunique()}")
    print(f"Problem types: {df['problem_type'].unique()}")
    
    return df


def process_uncertainty_measure(ind_dataset, ood_dataset, uncertainty_type, gname, risk_type, 
                              gt_approx, pred_approx, prediction_data, results, args, pbar, processed_same_dataset):
    """Process a single uncertainty measure configuration"""
    
    try:
        # Load uncertainty measure data
        path = get_results_path(
            ind_dataset, ood_dataset, uncertainty_type, gname, risk_type, 
            gt_approx, pred_approx, args.results_root
        )
        uncertainty_data = np.load(path)
        
        # Create measure identifier
        measure_id = create_measure_identifier(uncertainty_type, gname, risk_type, gt_approx, pred_approx)
        
        # Get prediction data
        pred_data = prediction_data.get(ind_dataset)
        
        if pred_data is None:
            if args.verbose:
                print(f"Skipping {measure_id} for {ind_dataset.value}->{ood_dataset.value}: No prediction data")
            pbar.update(1)
            return
        
        # Process each ensemble group
        for group_key, group_data in pred_data.items():
            group_idx = int(group_key.split('_')[1])  # Extract group index
            
            # Extract uncertainty scores for this group
            ind_test_scores = uncertainty_data['ind_test'][group_idx, 0, :]  # Shape: (n_test_samples,)
            ind_calib_scores = uncertainty_data['ind_calib'][group_idx, 0, :]  # Shape: (n_calib_samples,)
            ood_scores = uncertainty_data['ood'][group_idx, 0, :]  # Shape: (n_ood_samples,)
            
            # Get predictions and labels
            y_pred = group_data['y_pred']
            y_test = group_data['y_test']
            
            # Determine problem types to evaluate
            if ind_dataset != ood_dataset:
                # Case 1: OOD Detection (different datasets)
                ood_metrics = compute_ood_detection_metrics(ind_test_scores, ood_scores)
                
                results.append({
                    'ind_dataset': ind_dataset.value,
                    'ood_dataset': ood_dataset.value,
                    'measure': measure_id,
                    'uncertainty_type': uncertainty_type.value,
                    'gname': gname.value if gname else None,
                    'risk_type': risk_type.value if risk_type else None,
                    'gt_approximation': gt_approx.value if gt_approx else None,
                    'pred_approximation': pred_approx.value if pred_approx else None,
                    'ensemble_group': group_idx,
                    'problem_type': 'ood_detection',
                    'roc_auc': ood_metrics['roc_auc'],
                    'average_precision': None,
                    'accuracy': None,
                    'aurc': None,
                    'acc_cov_auc': None,
                    'coverage_at_1pct_error': None,
                    'coverage_at_2pct_error': None,
                    'coverage_at_5pct_error': None,
                    'n_ind_samples': ood_metrics['n_ind_samples'],
                    'n_ood_samples': ood_metrics['n_ood_samples'],
                    'n_correct': None,
                    'n_incorrect': None,
                    'ensemble_accuracy': group_data['ensemble_accuracy']
                })
                
            
            # Case 2 & 3: Same dataset evaluation - Use ind_test scores for misclassification and selective prediction
            # Only do this once per (ind_dataset, measure_id, group_idx) combination to avoid duplicates across OOD datasets
            same_dataset_key = (ind_dataset, measure_id, group_idx)
            if same_dataset_key not in processed_same_dataset:
                processed_same_dataset.add(same_dataset_key)
                
                # Misclassification detection using ind_test scores
                misc_metrics = compute_misclassification_detection_metrics(ind_test_scores, y_pred, y_test)
                
                results.append({
                    'ind_dataset': ind_dataset.value,
                    'ood_dataset': ind_dataset.value,  # Same dataset for both
                    'measure': measure_id,
                    'uncertainty_type': uncertainty_type.value,
                    'gname': gname.value if gname else None,
                    'risk_type': risk_type.value if risk_type else None,
                    'gt_approximation': gt_approx.value if gt_approx else None,
                    'pred_approximation': pred_approx.value if pred_approx else None,
                    'ensemble_group': group_idx,
                    'problem_type': 'misclassification_detection',
                    'roc_auc': misc_metrics['roc_auc'],
                    'average_precision': misc_metrics['average_precision'],
                    'accuracy': misc_metrics['accuracy'],
                    'aurc': None,
                    'acc_cov_auc': None,
                    'coverage_at_1pct_error': None,
                    'coverage_at_2pct_error': None,
                    'coverage_at_5pct_error': None,
                    'n_ind_samples': len(ind_test_scores),
                    'n_ood_samples': None,
                    'n_correct': misc_metrics['n_correct'],
                    'n_incorrect': misc_metrics['n_incorrect'],
                    'ensemble_accuracy': group_data['ensemble_accuracy']
                })
                
                # Selective prediction using ind_test scores
                sel_metrics = compute_selective_prediction_metrics(ind_test_scores, y_pred, y_test)
                
                results.append({
                    'ind_dataset': ind_dataset.value,
                    'ood_dataset': ind_dataset.value,  # Same dataset for both
                    'measure': measure_id,
                    'uncertainty_type': uncertainty_type.value,
                    'gname': gname.value if gname else None,
                    'risk_type': risk_type.value if risk_type else None,
                    'gt_approximation': gt_approx.value if gt_approx else None,
                    'pred_approximation': pred_approx.value if pred_approx else None,
                    'ensemble_group': group_idx,
                    'problem_type': 'selective_prediction',
                    'roc_auc': None,
                    'average_precision': None,
                    'accuracy': sel_metrics['overall_accuracy'],
                    'aurc': sel_metrics['aurc'],
                    'acc_cov_auc': sel_metrics['acc_cov_auc'],
                    'coverage_at_1pct_error': sel_metrics['coverage_at_1pct_error'],
                    'coverage_at_2pct_error': sel_metrics['coverage_at_2pct_error'],
                    'coverage_at_5pct_error': sel_metrics['coverage_at_5pct_error'],
                    'n_ind_samples': sel_metrics['n_samples'],
                    'n_ood_samples': None,
                    'n_correct': None,
                    'n_incorrect': None,
                    'ensemble_accuracy': group_data['ensemble_accuracy']
                })
        
        if args.verbose:
            print(f"✓ Processed {measure_id} for {ind_dataset.value}->{ood_dataset.value}")
            
    except Exception as e:
        if args.verbose:
            print(f"✗ Failed to process {measure_id if 'measure_id' in locals() else 'unknown'} for {ind_dataset.value}->{ood_dataset.value}: {e}")
    
    if pbar is not None:
        pbar.update(1)


if __name__ == "__main__":
    df = main()
