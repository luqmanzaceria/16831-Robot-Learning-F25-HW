#!/usr/bin/env python3
"""
Script to plot learning curves for q1_sb_na_na and q1_lb_na_na experiments
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalar_data(log_dir, scalar_name='Eval_AverageReturn'):
    """Extract scalar data from TensorBoard log directory."""
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None, None

    event_file = event_files[0]
    ea = EventAccumulator(event_file)
    ea.Reload()

    scalar_tags = ea.Tags()['scalars']

    if scalar_name not in scalar_tags:
        alternatives = ['Eval_AverageReturn', 'Train_AverageReturn', 'AverageReturn']
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

def find_experiment(data_dir, exp_name):
    """Find experiment directory matching the name."""
    pattern = os.path.join(data_dir, f"{exp_name}*")
    exp_dirs = glob.glob(pattern)

    if exp_dirs:
        return exp_dirs[0]
    return None

def main():
    data_dir = 'data'
    output_dir = 'plots_na'
    os.makedirs(output_dir, exist_ok=True)

    # Experiments to plot
    experiments = {
        'q1_sb_na_na': 'Small Batch (b=1500), No RTG, No Adv Std',
        'q1_lb_na_na': 'Large Batch (b=6000), No RTG, No Adv Std'
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (exp_name, exp_label) in enumerate(experiments.items()):
        exp_dir = find_experiment(data_dir, exp_name)

        if exp_dir:
            print(f"Found experiment: {exp_dir}")
            steps, values = extract_scalar_data(exp_dir)

            if steps is not None and values is not None:
                print(f"  Extracted {len(steps)} data points")
                print(f"  Range: [{min(values):.1f}, {max(values):.1f}]")

                # Plot on subplot
                axes[idx].plot(steps, values, 'b-', linewidth=2)
                axes[idx].set_xlabel('Iteration', fontsize=12)
                axes[idx].set_ylabel('Average Return', fontsize=12)
                axes[idx].set_title(exp_label, fontsize=13, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].set_ylim(0, 200)

                # Add final performance text
                final_perf = np.mean(values[-10:])
                axes[idx].text(0.95, 0.05, f'Final: {final_perf:.1f}',
                             transform=axes[idx].transAxes,
                             ha='right', va='bottom',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                             fontsize=11, fontweight='bold')
            else:
                print(f"  Failed to extract data")
                axes[idx].text(0.5, 0.5, 'No data available',
                             transform=axes[idx].transAxes,
                             ha='center', va='center', fontsize=14)
        else:
            print(f"Experiment not found: {exp_name}")
            axes[idx].text(0.5, 0.5, 'Experiment not found',
                         transform=axes[idx].transAxes,
                         ha='center', va='center', fontsize=14)

    plt.suptitle('CartPole-v0: No Reward-to-Go, No Advantage Standardization',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save plots
    pdf_file = os.path.join(output_dir, 'na_comparison.pdf')
    png_file = os.path.join(output_dir, 'na_comparison.png')

    plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
    plt.savefig(png_file, dpi=300, bbox_inches='tight')

    print(f"\nPlots saved to:")
    print(f"  {pdf_file}")
    print(f"  {png_file}")

    plt.show()

if __name__ == "__main__":
    main()
