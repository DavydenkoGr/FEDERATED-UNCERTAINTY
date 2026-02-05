#!/usr/bin/env python3
"""
Script to plot averaged accuracy histogram from three analysis files.
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser(description='Plot averaged accuracy histogram from three analysis files')
parser.add_argument('file1', type=str, help='First analysis file (JSON or CSV)')
parser.add_argument('file2', type=str, help='Second analysis file (JSON or CSV)')
parser.add_argument('file3', type=str, help='Third analysis file (JSON or CSV)')
parser.add_argument('--output', type=str, default=None, help='Output file path (default: averaged_accuracy.png)')

args = parser.parse_args()

def load_data(file_path):
    """Load data from JSON or CSV file."""
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Extract summary data
        result = {}
        for item in data['summary']:
            result[item['strategy']] = item['overall_accuracy']
        return result
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        result = {}
        for _, row in df.iterrows():
            result[row['strategy']] = row['overall_accuracy']
        return result
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

# Load data from all three files
data1 = load_data(args.file1)
data2 = load_data(args.file2)
data3 = load_data(args.file3)

# Collect all strategies and their accuracies
strategy_accuracies = defaultdict(list)

for strategy in data1:
    strategy_accuracies[strategy].append(data1[strategy])
for strategy in data2:
    strategy_accuracies[strategy].append(data2[strategy])
for strategy in data3:
    strategy_accuracies[strategy].append(data3[strategy])

# Strategy name mapping and order
STRATEGY_MAPPING = {
    'random': 'Random',
    'accuracy': 'Accuracy',
    'greedy_accuracy': 'Greedy accuracy',
    'market': 'Simplex projection',
    'uncertainty': 'Uncertainty-Aware'
}

STRATEGY_ORDER = ['Random', 'Accuracy', 'Greedy accuracy', 'Simplex projection', 'Uncertainty-Aware']

# Reverse mapping for lookup
REVERSE_MAPPING = {v: k for k, v in STRATEGY_MAPPING.items()}

# Compute averages
averaged_data = {}
for strategy in strategy_accuracies.keys():
    accuracies = strategy_accuracies[strategy]
    mapped_name = STRATEGY_MAPPING.get(strategy, strategy)
    averaged_data[mapped_name] = sum(accuracies) / len(accuracies)

# Order strategies according to STRATEGY_ORDER
strategies = [s for s in STRATEGY_ORDER if s in averaged_data]
accuracies = [averaged_data[s] for s in strategies]

plt.figure(figsize=(10, 7))
plt.bar(strategies, accuracies)
plt.xlabel('selection strategy')
plt.ylabel('ground truth accuracy')
plt.tight_layout()

# Save plot
output_file = args.output if args.output else 'averaged_accuracy.png'
plt.savefig(output_file)
print(f"Plot saved to: {output_file}")
