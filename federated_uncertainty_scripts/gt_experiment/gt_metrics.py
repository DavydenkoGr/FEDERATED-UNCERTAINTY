#!/usr/bin/env python3
"""
Script to analyze selected model sets across all clients and strategies.
Computes the accuracy of each strategy relative to the ground truth set.
"""

import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description='Analyze selected models across clients and strategies')
parser.add_argument('--save_dir', 
                    type=str, 
                    required=True,
                    help='Path to the saved models directory (e.g., ./data/saved_models/run_...)')
parser.add_argument('--output', 
                    type=str, 
                    default=None,
                    help='Path to save analysis results (default: save_dir/selected_models_analysis.json)')

args = parser.parse_args()

save_dir = Path(args.save_dir)
if not save_dir.exists():
    raise ValueError(f"Directory {save_dir} does not exist")

# Set default output path if not provided
if args.output is None:
    args.output = str(save_dir / "selected_models_analysis.json")

selected_models_dir = save_dir / "selected_models"
if not selected_models_dir.exists():
    raise ValueError(f"Directory {selected_models_dir} does not exist. Make sure you ran the main script first.")

# Collect all strategies and clients
strategies_per_client = defaultdict(dict)
client_dirs = sorted(selected_models_dir.glob("client_*"))

if not client_dirs:
    raise ValueError(f"No client directories found in {selected_models_dir}")

print(f"Found {len(client_dirs)} clients")

# Load data for each client
for client_dir in client_dirs:
    client_num = int(client_dir.name.split("_")[1])
    
    # Load all JSON files for this client
    json_files = list(client_dir.glob("*.json"))
    
    for json_file in json_files:
        strategy = json_file.stem
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            strategies_per_client[client_num][strategy] = data["selected_indices"]

# Check for ground_truth presence for each client
clients_with_gt = []
clients_without_gt = []

for client_num in strategies_per_client:
    if "ground_truth" in strategies_per_client[client_num]:
        clients_with_gt.append(client_num)
    else:
        clients_without_gt.append(client_num)

if clients_without_gt:
    print(f"WARNING: Clients without ground_truth: {clients_without_gt}")
    print("Analysis will only be performed for clients with ground_truth.")

if not clients_with_gt:
    raise ValueError("No clients with ground_truth found. Cannot perform analysis.")

print(f"\nAnalyzing {len(clients_with_gt)} clients with ground_truth")

# Analysis: for each strategy compute accuracy relative to ground_truth
results = defaultdict(lambda: {"matches": 0, "total": 0, "clients": []})

strategies_to_analyze = set()
for client_num in clients_with_gt:
    strategies_to_analyze.update(strategies_per_client[client_num].keys())
strategies_to_analyze.discard("ground_truth")

for client_num in clients_with_gt:
    gt_indices = set(strategies_per_client[client_num]["ground_truth"])
    gt_size = len(gt_indices)
    
    for strategy in strategies_to_analyze:
        if strategy not in strategies_per_client[client_num]:
            continue
        
        strategy_indices = set(strategies_per_client[client_num][strategy])
        strategy_size = len(strategy_indices)
        
        # Compute intersection
        intersection = gt_indices & strategy_indices
        matches = len(intersection)
        
        # Compute accuracy (how many models from ground_truth are in the strategy)
        accuracy = matches / gt_size if gt_size > 0 else 0.0
        
        results[strategy]["matches"] += matches
        results[strategy]["total"] += gt_size
        results[strategy]["clients"].append({
            "client_num": client_num,
            "matches": matches,
            "gt_size": gt_size,
            "strategy_size": strategy_size,
            "accuracy": accuracy,
            "gt_indices": sorted(list(gt_indices)),
            "strategy_indices": sorted(list(strategy_indices)),
            "intersection": sorted(list(intersection))
        })

# Compute overall accuracy for each strategy
summary = []
for strategy in sorted(strategies_to_analyze):
    if results[strategy]["total"] > 0:
        overall_accuracy = results[strategy]["matches"] / results[strategy]["total"]
        summary.append({
            "strategy": strategy,
            "overall_accuracy": overall_accuracy,
            "matches": results[strategy]["matches"],
            "total": results[strategy]["total"],
            "num_clients": len(results[strategy]["clients"])
        })

# Print results
print("\n" + "="*80)
print("ANALYSIS RESULTS")
print("="*80)

print("\nSummary (overall accuracy per strategy):")
print("-" * 80)
print(f"{'Strategy':<20} {'Accuracy':<15} {'Matches':<10} {'Total':<10} {'Clients':<10}")
print("-" * 80)
for s in summary:
    print(f"{s['strategy']:<20} {s['overall_accuracy']:<15.4f} {s['matches']:<10} {s['total']:<10} {s['num_clients']:<10}")

print("\n" + "="*80)
print("Detailed results per client:")
print("="*80)

for strategy in sorted(strategies_to_analyze):
    print(f"\nStrategy: {strategy}")
    print("-" * 80)
    for client_data in results[strategy]["clients"]:
        print(f"  Client {client_data['client_num']:2d}: "
              f"accuracy={client_data['accuracy']:.4f} "
              f"({client_data['matches']}/{client_data['gt_size']} models matched)")
        print(f"    GT indices:     {client_data['gt_indices']}")
        print(f"    Strategy indices: {client_data['strategy_indices']}")
        print(f"    Intersection:    {client_data['intersection']}")

# Save results
output_file = Path(args.output)
output_file.parent.mkdir(parents=True, exist_ok=True)

output_data = {
    "summary": summary,
    "detailed_results": {strategy: results[strategy]["clients"] 
                         for strategy in sorted(strategies_to_analyze)},
    "clients_analyzed": clients_with_gt,
    "clients_without_gt": clients_without_gt
}

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\nResults saved to: {output_file}")

# Also save to CSV for convenience
csv_output = output_file.with_suffix('.csv')
df_summary = pd.DataFrame(summary)
df_summary.to_csv(csv_output, index=False)
print(f"Summary CSV saved to: {csv_output}")
