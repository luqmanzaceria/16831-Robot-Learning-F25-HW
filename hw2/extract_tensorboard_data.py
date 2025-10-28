#!/usr/bin/env python3
"""
Script to extract learning curve data from TensorBoard event files.
This script reads TensorBoard logs and extracts the average return data for plotting.
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import json

def extract_scalar_data(log_dir, scalar_name='Eval_AverageReturn'):
    """
    Extract scalar data from a TensorBoard log directory.
    
    Args:
        log_dir: Path to the TensorBoard log directory
        scalar_name: Name of the scalar to extract (default: 'Eval_AverageReturn')
    
    Returns:
        tuple: (steps, values) arrays
    """
    # Find the event file
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None, None
    
    event_file = event_files[0]  # Take the first (and usually only) event file
    
    # Load the event file
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    # Get available scalar tags
    scalar_tags = ea.Tags()['scalars']
    print(f"Available scalar tags in {log_dir}: {scalar_tags}")
    
    if scalar_name not in scalar_tags:
        print(f"Scalar '{scalar_name}' not found. Available: {scalar_tags}")
        # Try alternative names
        alternatives = ['Eval_AverageReturn', 'Train_AverageReturn', 'AverageReturn', 'Eval_Return', 'Train_Return']
        for alt in alternatives:
            if alt in scalar_tags:
                scalar_name = alt
                print(f"Using alternative: {scalar_name}")
                break
        else:
            return None, None
    
    # Extract the scalar data
    scalar_events = ea.Scalars(scalar_name)
    steps = [s.step for s in scalar_events]
    values = [s.value for s in scalar_events]
    
    return np.array(steps), np.array(values)

def extract_all_experiments(data_dir, experiment_prefix):
    """
    Extract data from all experiments with a given prefix.
    
    Args:
        data_dir: Directory containing experiment folders
        experiment_prefix: Prefix to filter experiments (e.g., 'q1_sb_', 'q1_lb_')
    
    Returns:
        dict: Dictionary mapping experiment names to (steps, values) tuples
    """
    experiments = {}
    
    # Find all directories with the given prefix
    pattern = os.path.join(data_dir, f"{experiment_prefix}*")
    experiment_dirs = glob.glob(pattern)
    
    print(f"Found {len(experiment_dirs)} experiments with prefix '{experiment_prefix}':")
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        print(f"  - {exp_name}")
        
        # Extract data from this experiment
        steps, values = extract_scalar_data(exp_dir)
        if steps is not None and values is not None:
            experiments[exp_name] = (steps, values)
            print(f"    Extracted {len(steps)} data points, range [{min(values):.1f}, {max(values):.1f}]")
        else:
            print(f"    Failed to extract data")
    
    # Check for duplicate data
    print(f"\nChecking for duplicate data in {len(experiments)} experiments...")
    data_signatures = {}
    for exp_name, (steps, values) in experiments.items():
        # Create a signature based on the first few and last few values
        sig = tuple(values[:5]) + tuple(values[-5:]) + (len(values),)
        if sig in data_signatures:
            print(f"  WARNING: {exp_name} has identical data to {data_signatures[sig]}")
        else:
            data_signatures[sig] = exp_name
    
    return experiments

def create_learning_curve_plot(experiments, title, output_file, figsize=(12, 8)):
    """
    Create a learning curve plot from experiment data.
    
    Args:
        experiments: Dictionary mapping experiment names to (steps, values) tuples
        title: Title for the plot
        output_file: Output file path for the plot
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    print(f"Creating plot with {len(experiments)} experiments:")
    for exp_name in experiments.keys():
        print(f"  - {exp_name}")
    
    for i, (exp_name, (steps, values)) in enumerate(experiments.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        
        # Parse experiment name to create a readable label
        # e.g., "q1_sb_no_rtg_dsa_CartPole-v0_03-10-2025_20-29-38" -> "no_rtg_dsa"
        parts = exp_name.split('_')
        if len(parts) >= 4:
            # Extract the configuration part (e.g., "no_rtg_dsa", "rtg_dsa", "rtg_na")
            config_parts = parts[2:-2]  # Remove prefix and timestamp parts
            label = '_'.join(config_parts)
        else:
            label = exp_name
        
        # Add some transparency to help distinguish overlapping lines
        alpha = 0.8 if i < 3 else 0.6
        
        plt.plot(steps, values, label=label, color=color, linewidth=2.5, 
                marker=marker, markersize=6, markevery=max(1, len(steps)//10),
                linestyle=linestyle, alpha=alpha)
        
        print(f"  Plotted {label}: {len(steps)} points, range [{min(values):.1f}, {max(values):.1f}]")
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Average Return', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Also save as PNG for easier viewing
    png_file = output_file.replace('.pdf', '.png')
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    print(f"Plot also saved as: {png_file}")
    
    plt.show()

def save_data_to_json(experiments, output_file):
    """
    Save experiment data to JSON file for later use.
    
    Args:
        experiments: Dictionary mapping experiment names to (steps, values) tuples
        output_file: Output JSON file path
    """
    data_to_save = {}
    for exp_name, (steps, values) in experiments.items():
        data_to_save[exp_name] = {
            'steps': steps.tolist(),
            'values': values.tolist()
        }
    
    with open(output_file, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Data saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract and plot learning curves from TensorBoard logs')
    parser.add_argument('--data_dir', default='data', help='Directory containing experiment data')
    parser.add_argument('--output_dir', default='plots', help='Directory to save plots')
    parser.add_argument('--small_batch_prefix', default='q1_sb_', help='Prefix for small batch experiments')
    parser.add_argument('--large_batch_prefix', default='q1_lb_', help='Prefix for large batch experiments')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract small batch experiments
    print("=" * 50)
    print("EXTRACTING SMALL BATCH EXPERIMENTS")
    print("=" * 50)
    small_batch_experiments = extract_all_experiments(args.data_dir, args.small_batch_prefix)
    
    if small_batch_experiments:
        # Create small batch plot
        create_learning_curve_plot(
            small_batch_experiments,
            'Learning Curves - Small Batch Experiments (q1_sb_)',
            os.path.join(args.output_dir, 'small_batch_learning_curves.pdf')
        )
        
        # Save data
        save_data_to_json(
            small_batch_experiments,
            os.path.join(args.output_dir, 'small_batch_data.json')
        )
    else:
        print("No small batch experiments found!")
    
    # Extract large batch experiments
    print("\n" + "=" * 50)
    print("EXTRACTING LARGE BATCH EXPERIMENTS")
    print("=" * 50)
    large_batch_experiments = extract_all_experiments(args.data_dir, args.large_batch_prefix)
    
    if large_batch_experiments:
        # Create large batch plot
        create_learning_curve_plot(
            large_batch_experiments,
            'Learning Curves - Large Batch Experiments (q1_lb_)',
            os.path.join(args.output_dir, 'large_batch_learning_curves.pdf')
        )
        
        # Save data
        save_data_to_json(
            large_batch_experiments,
            os.path.join(args.output_dir, 'large_batch_data.json')
        )
    else:
        print("No large batch experiments found!")
        print("You may need to run the large batch experiments first.")
        print("Use the following commands:")
        print("python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 6000 -dsa --exp_name q1_lb_no_rtg_dsa")
        print("python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 6000 -rtg -dsa --exp_name q1_lb_rtg_dsa")
        print("python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 6000 -rtg --exp_name q1_lb_rtg_na")

if __name__ == "__main__":
    main()
