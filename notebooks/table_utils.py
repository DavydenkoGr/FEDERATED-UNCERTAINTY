import sys
import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from mdu.unc.constants import ScalingType, OTTarget

sys.path.insert(0, "../")

from mdu.eval.table_analysis_utils import (
    transform_by_tasks,
    select_composite_and_components,
    check_composite_dominance,
    compute_average_ranks,
    analyze_composite_pareto_performance,
)
from configs.interesting_compositions import INTERESTING_COMPOSITIONS

# Set pandas display options to show all columns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


HYPERPARAM_COLS = ['eps', 'grid_size', 'n_targets_multiplier', 'scaling_type', 'target']



def parse_hyperparameters_from_filename(filename: str) -> Dict[str, Union[str, float]]:
    """
    Parse hyperparameters from CSV filename.
    
    Args:
        filename: CSV filename
    
    Returns:
        Dictionary with parsed hyperparameters
    """
    hyperparams = {}
    
    # Extract eps (epsilon)
    eps_match = re.search(r'eps_([0-9.]+)', filename)
    if eps_match:
        hyperparams['eps'] = float(eps_match.group(1))
    
    # Extract iterations
    iters_match = re.search(r'iters_(\d+)', filename)
    if iters_match:
        hyperparams['iters'] = int(iters_match.group(1))
    
    # Extract tolerance
    tol_match = re.search(r'tol_([0-9e.-]+)', filename)
    if tol_match:
        hyperparams['tol'] = float(tol_match.group(1))
    
    # Extract random seed
    rs_match = re.search(r'rs_(\d+)', filename)
    if rs_match:
        hyperparams['rs'] = int(rs_match.group(1))
    
    # Extract grid size
    grid_match = re.search(r'grid_size_(\d+)', filename)
    if grid_match:
        hyperparams['grid_size'] = int(grid_match.group(1))
    
    # Extract n_targets_multiplier
    targets_match = re.search(r'n_targets_multiplier_(\d+)', filename)
    if targets_match:
        hyperparams['n_targets_multiplier'] = int(targets_match.group(1))
    
    # Extract target parameter (exp or beta)
    target_match = re.search(rf'target_({OTTarget.EXP.value}|{OTTarget.BETA.value}|{OTTarget.BALL.value})', filename)
    if target_match:
        hyperparams['target'] = target_match.group(1)
    
    target_match = re.search(rf'scaling_type_({ScalingType.GLOBAL.value}|{ScalingType.FEATURE_WISE.value})', filename)
    if target_match:
        hyperparams['scaling_type'] = target_match.group(1)
    
    # Extract base experiment type
    if 'per_comp_scaled_benchmark' in filename:
        hyperparams['experiment_type'] = 'per_comp_scaled_benchmark'
    elif 'extended_benchmark' in filename:
        hyperparams['experiment_type'] = 'extended_benchmark'
    elif 'benchmark' in filename:
        hyperparams['experiment_type'] = 'benchmark'
    else:
        hyperparams['experiment_type'] = 'other'
    
    return hyperparams


def load_and_combine_csv_files(pattern: str = "per_comp_scaled_benchmark*.csv", 
                              base_dir: str = "../") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load multiple CSV files matching the pattern and combine them with hyperparameter metadata.
    
    Args:
        pattern: Glob pattern to match CSV files
        base_dir: Base directory to search for files
    
    Returns:
        Tuple of (combined_df, loading_summary_df)
    """
    csv_files = glob.glob(os.path.join(base_dir, pattern))
    
    if not csv_files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    
    combined_data = []
    loading_info = []
    
    for csv_file in csv_files:
        # Load the CSV
        df = pd.read_csv(csv_file)
        
        # Parse hyperparameters from filename
        filename = os.path.basename(csv_file)
        hyperparams = parse_hyperparameters_from_filename(filename)
        
        # Add hyperparameter columns
        for param, value in hyperparams.items():
            df[param] = value
        
        # Add source filename for reference
        df['source_file'] = filename
        
        combined_data.append(df)
        
        # Track loading info
        loading_info.append({
            'filename': filename,
            'rows': len(df),
            **hyperparams
        })
    
    # Combine all DataFrames
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Create summary DataFrame
    loading_summary = pd.DataFrame(loading_info)
    
    return combined_df, loading_summary



def analyze_hyperparameter_effects_general(
    combined_df: pd.DataFrame, 
    composite_names: Optional[List[str]] = None,
    selective_metric: str = "acc_cov_auc"
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Analyze how hyperparameter changes affect composite uncertainty measures across all datasets.
    
    Args:
        combined_df: Combined DataFrame with hyperparameter metadata
        composite_names: List of composite names to analyze (if None, uses all from INTERESTING_COMPOSITIONS)
        selective_metric: Metric to use for selective prediction
    
    Returns:
        Tuple of (results_dict, analysis_summary_df)
    """
    if composite_names is None:
        composite_names = list(INTERESTING_COMPOSITIONS.keys())
    
    results = {}
    analysis_info = []
    
    # Get unique hyperparameter combinations
    hyperparam_cols = [col for col in combined_df.columns 
                      if col in HYPERPARAM_COLS]
    
    # Group by hyperparameter combinations
    grouped = combined_df.groupby(hyperparam_cols)
    
    # Track analysis info
    for name, group in grouped:
        if isinstance(name, tuple):
            param_str = ", ".join([f"{col}={val}" for col, val in zip(hyperparam_cols, name)])
            param_dict = dict(zip(hyperparam_cols, name))
        else:
            param_str = f"{hyperparam_cols[0]}={name}"
            param_dict = {hyperparam_cols[0]: name}
        
        analysis_info.append({
            'hyperparameter_combination': param_str,
            'rows_count': len(group),
            **param_dict
        })
    
    # Analyze each composite measure
    for composite_name in composite_names:
        composite_results = []
        
        for param_combination, group_df in grouped:
            try:
                # Transform the data for this hyperparameter combination
                transformed_df = transform_by_tasks(group_df, selective_metric=selective_metric)
                
                # Get composite and components
                composite_df = select_composite_and_components(transformed_df, composite_name)
                
                # Check dominance
                dominance_df = check_composite_dominance(composite_df)
                
                # Compute average ranks
                avg_ranks = compute_average_ranks(transformed_df)
                
                # Pareto analysis
                pareto_results = analyze_composite_pareto_performance(
                    transformed_df, {composite_name: INTERESTING_COMPOSITIONS[composite_name]}
                )
                
                # Create summary for this parameter combination
                if isinstance(param_combination, tuple):
                    param_dict = dict(zip(hyperparam_cols, param_combination))
                else:
                    param_dict = {hyperparam_cols[0]: param_combination}
                
                summary = {
                    **param_dict,
                    'n_problems': len(dominance_df),
                    'dominates_100_pct': dominance_df['if_dominates_100%'].sum(),
                    'dominates_75_pct': dominance_df['if_dominates_75%'].sum(),
                    'dominates_50_pct': dominance_df['if_dominates_50%'].sum(),
                    'beats_worst_component': dominance_df['beats_worst_component'].sum(),
                }
                
                # Add composite rank if available
                composite_key = composite_name.lower()
                if composite_key in avg_ranks:
                    summary['avg_rank'] = avg_ranks[composite_key]
                
                # Add pareto performance
                if composite_name in pareto_results:
                    pareto_data = pareto_results[composite_name]
                    summary['pareto_count'] = pareto_data['pareto_count']
                    summary['pareto_total'] = pareto_data['total_problems']
                    summary['pareto_percentage'] = pareto_data['pareto_percentage']
                
                composite_results.append(summary)
                
            except Exception as e:
                continue
        
        if composite_results:
            results[composite_name] = pd.DataFrame(composite_results)
    
    analysis_summary = pd.DataFrame(analysis_info)
    
    return results, analysis_summary


def compare_hyperparameters_by_metric(
    analysis_results: Dict[str, pd.DataFrame],
    metric: str = 'pareto_percentage'
) -> pd.DataFrame:
    """
    Compare hyperparameter effects across different composite measures for a specific metric.
    
    Args:
        analysis_results: Results from analyze_hyperparameter_effects_general
        metric: Metric to compare ('pareto_percentage', 'avg_rank', 'dominates_50_pct', etc.)
    
    Returns:
        DataFrame with hyperparameter combinations as rows and composite measures as columns
    """
    comparison_data = []
    
    for composite_name, results_df in analysis_results.items():
        if metric not in results_df.columns:
            print(f"Metric '{metric}' not found in results for {composite_name}")
            continue
        
        # Get hyperparameter columns (exclude metric columns)
        hyperparam_cols = [col for col in results_df.columns 
                          if col in HYPERPARAM_COLS]
        
        for _, row in results_df.iterrows():
            param_combo = tuple(row[col] for col in hyperparam_cols)
            comparison_data.append({
                'hyperparams': param_combo,
                'hyperparam_str': ", ".join([f"{col}={row[col]}" for col in hyperparam_cols]),
                'composite': composite_name,
                'value': row[metric]
            })
    
    if not comparison_data:
        return pd.DataFrame()
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Pivot to get hyperparameters as rows, composites as columns
    pivot_df = comparison_df.pivot(index=['hyperparams', 'hyperparam_str'], 
                                  columns='composite', 
                                  values='value')
    
    # Reset index to make hyperparams a column
    pivot_df = pivot_df.reset_index()
    pivot_df = pivot_df.set_index('hyperparam_str')
    pivot_df = pivot_df.drop('hyperparams', axis=1)
    
    return pivot_df



def analyze_hyperparameter_effects_per_dataset(
    combined_df: pd.DataFrame,
    composite_names: Optional[List[str]] = None,
    selective_metric: str = "acc_cov_auc"
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], pd.DataFrame]:
    """
    Analyze how hyperparameter changes affect composite uncertainty measures per dataset.
    
    Args:
        combined_df: Combined DataFrame with hyperparameter metadata
        composite_names: List of composite names to analyze (if None, uses all from INTERESTING_COMPOSITIONS)
        selective_metric: Metric to use for selective prediction
    
    Returns:
        Tuple of (nested_results_dict, dataset_summary_df)
    """
    if composite_names is None:
        composite_names = list(INTERESTING_COMPOSITIONS.keys())
    
    results_by_dataset = {}
    dataset_summary = []
    
    # Get unique datasets
    datasets = combined_df['ind_dataset'].unique()
    
    # Get hyperparameter columns
    hyperparam_cols = [col for col in combined_df.columns 
                      if col in HYPERPARAM_COLS]
    
    for dataset in datasets:
        dataset_df = combined_df[combined_df['ind_dataset'] == dataset]
        results_by_dataset[dataset] = {}
        
        # Group by hyperparameter combinations for this dataset
        grouped = dataset_df.groupby(hyperparam_cols)
        
        dataset_info = {
            'dataset': dataset,
            'hyperparameter_combinations': len(grouped),
            'total_rows': len(dataset_df)
        }
        
        for composite_name in composite_names:
            composite_results = []
            
            for param_combination, group_df in grouped:
                try:
                    # Transform the data for this hyperparameter combination
                    transformed_df = transform_by_tasks(group_df, selective_metric=selective_metric)
                    
                    # Filter to only this dataset's rows
                    dataset_transformed = transformed_df.loc[
                        transformed_df.index.get_level_values('ind_dataset') == dataset
                    ]
                    
                    if dataset_transformed.empty:
                        continue
                    
                    # Get composite and components
                    composite_df = select_composite_and_components(dataset_transformed, composite_name)
                    
                    if composite_df.empty:
                        continue
                    
                    # Check dominance
                    dominance_df = check_composite_dominance(composite_df)
                    
                    # Create summary for this parameter combination
                    if isinstance(param_combination, tuple):
                        param_dict = dict(zip(hyperparam_cols, param_combination))
                    else:
                        param_dict = {hyperparam_cols[0]: param_combination}
                    
                    # Calculate performance metrics
                    composite_cols = [c for c in composite_df.columns if c.startswith('composite')]
                    if composite_cols:
                        composite_col = composite_cols[0]
                        composite_values = composite_df[composite_col].dropna()
                        
                        summary = {
                            **param_dict,
                            'dataset': dataset,
                            'n_problems': len(dominance_df),
                            'dominates_100_pct': dominance_df['if_dominates_100%'].sum(),
                            'dominates_75_pct': dominance_df['if_dominates_75%'].sum(),
                            'dominates_50_pct': dominance_df['if_dominates_50%'].sum(),
                            'beats_worst_component': dominance_df['beats_worst_component'].sum(),
                            'composite_mean': composite_values.mean() if len(composite_values) > 0 else np.nan,
                            'composite_std': composite_values.std() if len(composite_values) > 0 else np.nan,
                            'composite_min': composite_values.min() if len(composite_values) > 0 else np.nan,
                            'composite_max': composite_values.max() if len(composite_values) > 0 else np.nan,
                        }
                        
                        composite_results.append(summary)
                
                except Exception as e:
                    continue
            
            if composite_results:
                results_by_dataset[dataset][composite_name] = pd.DataFrame(composite_results)
                dataset_info[f'{composite_name}_results_count'] = len(composite_results)
            else:
                dataset_info[f'{composite_name}_results_count'] = 0
        
        dataset_summary.append(dataset_info)
    
    dataset_summary_df = pd.DataFrame(dataset_summary)
    
    return results_by_dataset, dataset_summary_df


def summarize_per_dataset_effects(
    per_dataset_results: Dict[str, Dict[str, pd.DataFrame]],
    metric: str = 'composite_mean'
) -> pd.DataFrame:
    """
    Summarize hyperparameter effects across datasets for a specific metric.
    
    Args:
        per_dataset_results: Results from analyze_hyperparameter_effects_per_dataset
        metric: Metric to summarize
    
    Returns:
        DataFrame with datasets as rows, hyperparameter combinations as columns
    """
    summary_data = []
    
    for dataset, composites_dict in per_dataset_results.items():
        for composite_name, results_df in composites_dict.items():
            if metric not in results_df.columns:
                continue
            
            # Get hyperparameter columns
            hyperparam_cols = [col for col in results_df.columns 
                              if col in HYPERPARAM_COLS]
            
            for _, row in results_df.iterrows():
                param_str = ", ".join([f"{col}={row[col]}" for col in hyperparam_cols])
                
                summary_data.append({
                    'dataset': dataset,
                    'composite': composite_name,
                    'hyperparams': param_str,
                    'value': row[metric]
                })
    
    if not summary_data:
        return pd.DataFrame()
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create a pivot table
    pivot_df = summary_df.pivot_table(
        index=['dataset', 'composite'], 
        columns='hyperparams', 
        values='value',
        aggfunc='first'  # Should be unique anyway
    )
    
    return pivot_df


def plot_hyperparameter_effects(
    analysis_results: Dict[str, pd.DataFrame],
    metric: str = 'pareto_percentage',
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create visualizations of hyperparameter effects on composite measures.
    
    Args:
        analysis_results: Results from analyze_hyperparameter_effects_general
        metric: Metric to plot
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    # Create comparison DataFrame
    comparison_df = compare_hyperparameters_by_metric(analysis_results, metric)
    
    if comparison_df.empty:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Hyperparameter Effects on Composite Measures ({metric})', fontsize=16, y=0.98)
    
    # Heatmap of all results
    ax1 = axes[0, 0]
    sns.heatmap(comparison_df.T, annot=True, fmt='.2f', cmap='viridis', ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Heatmap: All Combinations', fontsize=12, pad=20)
    ax1.set_xlabel('Hyperparameter Combinations', fontsize=10)
    ax1.set_ylabel('Composite Measures', fontsize=10)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.tick_params(axis='y', rotation=0, labelsize=8)
    
    # Bar plot of best performance per composite
    ax2 = axes[0, 1]
    best_per_composite = comparison_df.max(axis=0)
    best_per_composite.plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_title('Best Performance per Composite', fontsize=12, pad=20)
    ax2.set_ylabel(metric, fontsize=10)
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    
    # Bar plot of best performance per hyperparameter combination
    ax3 = axes[1, 0]
    best_per_hyperparam = comparison_df.max(axis=1)
    best_per_hyperparam.plot(kind='bar', ax=ax3, color='lightcoral')
    ax3.set_title('Best Performance per Hyperparameter', fontsize=12, pad=20)
    ax3.set_ylabel(metric, fontsize=10)
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)
    
    # Box plot showing distribution across hyperparameters
    ax4 = axes[1, 1]
    melted_df = comparison_df.melt(var_name='Composite', value_name=metric)
    melted_df = melted_df.dropna()
    if len(melted_df) > 0:
        sns.boxplot(data=melted_df, x='Composite', y=metric, ax=ax4)
        ax4.set_title('Distribution Across Hyperparameters', fontsize=12, pad=20)
        ax4.tick_params(axis='x', rotation=45, labelsize=8)
        ax4.tick_params(axis='y', labelsize=8)
        ax4.set_xlabel('Composite', fontsize=10)
        ax4.set_ylabel(metric, fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_per_dataset_effects(
    per_dataset_results: Dict[str, Dict[str, pd.DataFrame]],
    composite_name: str,
    metric: str = 'composite_mean',
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot hyperparameter effects for a specific composite measure across datasets.
    
    Args:
        per_dataset_results: Results from analyze_hyperparameter_effects_per_dataset
        composite_name: Name of composite measure to analyze
        metric: Metric to plot
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    # Collect data for this composite across all datasets
    plot_data = []
    
    for dataset, composites_dict in per_dataset_results.items():
        if composite_name not in composites_dict:
            continue
        
        results_df = composites_dict[composite_name]
        if metric not in results_df.columns:
            continue
        
        # Get hyperparameter columns
        hyperparam_cols = [col for col in results_df.columns 
                          if col in ['eps', 'grid_size', 'n_targets_multiplier', 'scaler_type', 'target']]
        
        for _, row in results_df.iterrows():
            param_str = ", ".join([f"{col}={row[col]}" for col in hyperparam_cols])
            plot_data.append({
                'dataset': dataset,
                'hyperparams': param_str,
                'value': row[metric],
                **{col: row[col] for col in hyperparam_cols}
            })
    
    if not plot_data:
        return None
    
    plot_df = pd.DataFrame(plot_data)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{composite_name} - {metric} across Datasets', fontsize=16, y=0.98)
    
    # Heatmap: datasets vs hyperparameters
    ax1 = axes[0, 0]
    pivot_df = plot_df.pivot(index='dataset', columns='hyperparams', values='value')
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='viridis', ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Datasets vs Hyperparameters', fontsize=12, pad=20)
    ax1.set_xlabel('Hyperparameters', fontsize=10)
    ax1.set_ylabel('Datasets', fontsize=10)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.tick_params(axis='y', rotation=0, labelsize=8)
    
    # Bar plot: average performance per hyperparameter
    ax2 = axes[0, 1]
    avg_per_hyperparam = plot_df.groupby('hyperparams')['value'].mean()
    avg_per_hyperparam.plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_title('Average Performance per Hyperparameter', fontsize=12, pad=20)
    ax2.set_ylabel(metric, fontsize=10)
    ax2.set_xlabel('Hyperparameters', fontsize=10)
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    
    # Bar plot: average performance per dataset
    ax3 = axes[1, 0]
    avg_per_dataset = plot_df.groupby('dataset')['value'].mean()
    avg_per_dataset.plot(kind='bar', ax=ax3, color='lightcoral')
    ax3.set_title('Average Performance per Dataset', fontsize=12, pad=20)
    ax3.set_ylabel(metric, fontsize=10)
    ax3.set_xlabel('Datasets', fontsize=10)
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)
    
    # Box plot: distribution by hyperparameter if we have individual parameters
    ax4 = axes[1, 1]
    if 'eps' in plot_df.columns:
        sns.boxplot(data=plot_df, x='eps', y='value', ax=ax4)
        ax4.set_title('Distribution by Epsilon', fontsize=12, pad=20)
        ax4.set_ylabel(metric, fontsize=10)
        ax4.set_xlabel('Epsilon', fontsize=10)
    elif 'grid_size' in plot_df.columns:
        sns.boxplot(data=plot_df, x='grid_size', y='value', ax=ax4)
        ax4.set_title('Distribution by Grid Size', fontsize=12, pad=20)
        ax4.set_ylabel(metric, fontsize=10)
        ax4.set_xlabel('Grid Size', fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'No individual\nparameter to plot', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        ax4.set_title('Parameter Distribution', fontsize=12, pad=20)
    
    ax4.tick_params(axis='x', labelsize=8)
    ax4.tick_params(axis='y', labelsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def create_hyperparameter_summary_table(
    analysis_results: Dict[str, pd.DataFrame],
    metrics: List[str] = ['pareto_percentage', 'dominates_50_pct', 'avg_rank']
) -> pd.DataFrame:
    """
    Create a comprehensive summary table of hyperparameter effects.
    
    Args:
        analysis_results: Results from analyze_hyperparameter_effects_general
        metrics: List of metrics to include in summary
    
    Returns:
        Summary DataFrame
    """
    summary_data = []
    
    for composite_name, results_df in analysis_results.items():
        # Get hyperparameter columns
        hyperparam_cols = [col for col in results_df.columns 
                          if col in HYPERPARAM_COLS]
        
        for _, row in results_df.iterrows():
            param_str = ", ".join([f"{col}={row[col]}" for col in hyperparam_cols])
            
            summary_row = {
                'composite': composite_name,
                'hyperparams': param_str,
                **{col: row[col] for col in hyperparam_cols}
            }
            
            # Add metrics
            for metric in metrics:
                if metric in results_df.columns:
                    summary_row[metric] = row[metric]
                else:
                    summary_row[metric] = np.nan
            
            summary_data.append(summary_row)
    
    return pd.DataFrame(summary_data)


def analyze_individual_component_performance(
    combined_df: pd.DataFrame,
    composite_name: str,
    selective_metric: str = "acc_cov_auc"
) -> pd.DataFrame:
    """
    Analyze individual component performance metrics for a given composite measure.
    
    Since individual component performance doesn't depend on hyperparameters,
    we use a sample hyperparameter combination to get the individual metrics.
    
    The avg_rank returned is the GLOBAL average rank among ALL measures, not just within components.
    
    Args:
        combined_df: Combined DataFrame with all results
        composite_name: Name of the composite measure to analyze components for
        selective_metric: Metric to use for selective prediction
    
    Returns:
        DataFrame with individual component statistics
    """
    # Get a sample hyperparameter combination (since individual metrics don't depend on hyperparams)
    hyperparam_cols = [col for col in combined_df.columns 
                      if col in HYPERPARAM_COLS]
    
    if hyperparam_cols:
        first_combo = combined_df.groupby(hyperparam_cols).first().index[0]
        if isinstance(first_combo, tuple):
            filter_conditions = {col: val for col, val in zip(hyperparam_cols, first_combo)}
        else:
            filter_conditions = {hyperparam_cols[0]: first_combo}
        
        # Filter to get just this hyperparameter combination
        mask = pd.Series(True, index=combined_df.index)
        for col, val in filter_conditions.items():
            mask &= (combined_df[col] == val)
        
        sample_df = combined_df[mask].copy()
    else:
        sample_df = combined_df.copy()
    
    # Transform the data to get ALL measures (not just the composite and its components)
    transformed_df = transform_by_tasks(sample_df, selective_metric=selective_metric)
    
    # Compute GLOBAL average ranks for ALL measures
    global_avg_ranks = compute_average_ranks(transformed_df)
    
    # Get composite and components data to identify which components to analyze
    composite_df = select_composite_and_components(transformed_df, composite_name)
    
    # Get all individual component names (exclude the composite itself)
    component_names = [col for col in composite_df.columns 
                      if col not in ['ind_dataset', 'eval'] and not col.lower().startswith('composite')]
    
    # Calculate dominance statistics for individual components
    results_data = []
    
    for component in component_names:
        if component not in global_avg_ranks.index:
            continue
            
        # Use GLOBAL average rank, not just rank within this composite
        component_global_avg_rank = global_avg_ranks[component]
        
        # Calculate how often this component beats others
        dominance_stats = {'component': component, 'avg_rank': component_global_avg_rank}
        
        # Count dominance percentages
        total_components = len(component_names)
        better_than_50_pct = 0
        better_than_75_pct = 0  
        better_than_100_pct = 0
        
        # For each problem instance, check dominance
        df_reset = composite_df.reset_index()
        for _, row in df_reset.iterrows():
            component_values = []
            for comp in component_names:
                if comp in row and pd.notna(row[comp]):
                    component_values.append((comp, row[comp]))
            
            if len(component_values) < 2:
                continue
                
            # Sort by performance (higher is better)
            component_values.sort(key=lambda x: x[1], reverse=True)
            
            # Find position of our component
            component_rank = None
            for i, (comp_name, _) in enumerate(component_values):
                if comp_name == component:
                    component_rank = i
                    break
            
            if component_rank is not None:
                total_comps = len(component_values)
                percentile = (total_comps - component_rank) / total_comps
                
                if percentile >= 0.5:
                    better_than_50_pct += 1
                if percentile >= 0.75:
                    better_than_75_pct += 1
                if percentile == 1.0:  # Best component
                    better_than_100_pct += 1
        
        total_problems = len(df_reset)
        dominance_stats.update({
            'dominates_50_pct': better_than_50_pct,
            'dominates_75_pct': better_than_75_pct, 
            'dominates_100_pct': better_than_100_pct,
            'n_problems': total_problems,
            'dominates_50_pct_rate': better_than_50_pct / total_problems * 100 if total_problems > 0 else 0,
            'dominates_75_pct_rate': better_than_75_pct / total_problems * 100 if total_problems > 0 else 0,
            'dominates_100_pct_rate': better_than_100_pct / total_problems * 100 if total_problems > 0 else 0,
        })
        
        results_data.append(dominance_stats)
    
    return pd.DataFrame(results_data).sort_values('avg_rank', ascending=True)

def fmt_valvar(df_mean: pd.DataFrame,
               df_std: pd.DataFrame,
               mean_decimals: int = 3,
               std_decimals: int = 3,
               leading_dot_std: bool = True,
               bold: bool = True,
               underline: bool = True,
               best_mask: pd.DataFrame | None = None,
               na: str = "--") -> pd.DataFrame:
    """
    Create DataFrame of LaTeX strings like:
        \\valvar{\\textbf{\\underline{0.318}}}{.007}
    
    Parameters
    ----------
    df_mean, df_std : same-shaped DataFrames
    mean_decimals, std_decimals : rounding for mean/std
    leading_dot_std : drop leading '0' in std (0.007 -> .007)
    bold, underline : default styling for mean
    best_mask : optional boolean DataFrame (same shape).
                If provided, styling (bold/underline) is applied only where best_mask==True.
                Else styling applied to all cells.
    na : placeholder for missing values
    """
    assert df_mean.shape == df_std.shape, "Shapes must match"
    if best_mask is not None:
        assert best_mask.shape == df_mean.shape, "best_mask shape must match"

    def style_mean(m_str: str, i: int, j: int) -> str:
        apply_style = True if best_mask is None else bool(best_mask.iat[i, j])
        if apply_style:
            if underline:
                m_str = f"\\underline{{{m_str}}}"
            if bold:
                m_str = f"\\textbf{{{m_str}}}"
        return m_str

    out = pd.DataFrame(index=df_mean.index, columns=df_mean.columns, dtype=object)
    for i, row in enumerate(df_mean.index):
        for j, col in enumerate(df_mean.columns):
            m = df_mean.iat[i, j]
            s = df_std.iat[i, j]
            if pd.notna(m) and pd.notna(s):
                m_str = f"{m:.{mean_decimals}f}"
                s_str = f"{s:.{std_decimals}f}"
                if leading_dot_std and s_str.startswith("0"):
                    s_str = s_str[1:]
                m_str = style_mean(m_str, i, j)
                out.iat[i, j] = f"\\valvar{{{m_str}}}{{{s_str}}}"
            else:
                out.iat[i, j] = na
    return out

def with_avg_row(df: pd.DataFrame, label=("AVG", "mean over rows")) -> pd.DataFrame:
    avg = df.mean(axis=0, numeric_only=True)  # column-wise mean
    avg_row = pd.DataFrame([avg], 
        index=pd.MultiIndex.from_tuples([label], names=df.index.names))
    return pd.concat([df, avg_row])  # keeps it as the last row

def mean_pm_std(df_mean: pd.DataFrame, df_std: pd.DataFrame, decimals: int = 3, nan="--") -> pd.DataFrame:
    """Return a DataFrame of LaTeX strings like '$1.234 \\pm 0.056$'."""
    assert df_mean.shape == df_std.shape
    def combine_cols(mcol: pd.Series, scol: pd.Series) -> pd.Series:
        return mcol.combine(
            scol,
            lambda m, s: (f"${m:.{decimals}f} \\pm {s:.{decimals}f}$") 
                         if pd.notna(m) and pd.notna(s) else nan
        )
    return df_mean.combine(df_std, combine_cols)
