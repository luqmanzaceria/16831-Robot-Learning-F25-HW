#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import argparse

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not available")

def extract_tensorboard_data(log_dir):
    """Extract DAgger data from TensorBoard logs"""
    if not HAS_TENSORBOARD:
        print("TensorBoard not available, cannot extract data")
        return None, None, None, None

    log_path = Path(log_dir)
    tf_logs = list(log_path.glob("events.out.tfevents*"))

    if not tf_logs:
        print(f"No TensorBoard logs found in {log_dir}")
        return None, None, None, None

    # Use the most recent log file
    log_file = max(tf_logs, key=os.path.getctime)
    print(f"Reading from: {log_file}")

    # Create event accumulator
    ea = EventAccumulator(str(log_file))
    ea.Reload()

    # Extract data
    iterations = []
    eval_returns = []
    eval_stds = []
    initial_expert = None

    # Get Eval_AverageReturn
    if 'Eval_AverageReturn' in ea.Tags()['scalars']:
        scalar_events = ea.Scalars('Eval_AverageReturn')
        iterations = [s.step for s in scalar_events]
        eval_returns = [s.value for s in scalar_events]

    # Get Eval_StdReturn
    if 'Eval_StdReturn' in ea.Tags()['scalars']:
        std_events = ea.Scalars('Eval_StdReturn')
        eval_stds = [s.value for s in std_events]
    else:
        eval_stds = [0] * len(eval_returns)  # Default to 0 if no std data

    # Get Initial_DataCollection_AverageReturn (expert performance)
    if 'Initial_DataCollection_AverageReturn' in ea.Tags()['scalars']:
        initial_events = ea.Scalars('Initial_DataCollection_AverageReturn')
        if initial_events:
            initial_expert = initial_events[0].value

    return iterations, eval_returns, eval_stds, initial_expert

def create_dagger_figure(ant_bc=4323.74, halfcheetah_bc=1500):
    """Create Figure 2 for DAgger results extracting data from specific experiment directories"""

    # Experiment directories
    data_dir = Path("data")
    ant_exp = "q2_dagger_ant_final1_Ant-v2_18-09-2025_23-41-07"
    halfcheetah_exp = "q2_dagger_halfcheetah_final1_HalfCheetah-v2_18-09-2025_23-42-28"

    # Extract Ant-v2 data
    ant_dir = data_dir / ant_exp
    print(f"Extracting Ant-v2 data from: {ant_dir}")
    ant_iterations, ant_returns, ant_stds, ant_expert = extract_tensorboard_data(ant_dir)

    # Extract HalfCheetah-v2 data
    halfcheetah_dir = data_dir / halfcheetah_exp
    print(f"Extracting HalfCheetah-v2 data from: {halfcheetah_dir}")
    halfcheetah_iterations, halfcheetah_returns, halfcheetah_stds, halfcheetah_expert = extract_tensorboard_data(halfcheetah_dir)

    if ant_returns is None or halfcheetah_returns is None:
        print("Failed to extract data from TensorBoard logs")
        return None

    # Convert to numpy arrays
    ant_iterations = np.array(ant_iterations)
    ant_returns = np.array(ant_returns)
    ant_stds = np.array(ant_stds)

    halfcheetah_iterations = np.array(halfcheetah_iterations)
    halfcheetah_returns = np.array(halfcheetah_returns)
    halfcheetah_stds = np.array(halfcheetah_stds)

    # Baseline performances
    baselines = {
        'Ant-v2': {
            'expert': ant_expert if ant_expert else 4205.78,
            'bc': ant_bc
        },
        'HalfCheetah-v2': {
            'expert': halfcheetah_expert if halfcheetah_expert else 4000,
            'bc': halfcheetah_bc
        }
    }

    print(f"Ant-v2 Expert Performance: {baselines['Ant-v2']['expert']:.1f}")
    print(f"HalfCheetah-v2 Expert Performance: {baselines['HalfCheetah-v2']['expert']:.1f}")

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Ant-v2 (left)
    ax1.errorbar(ant_iterations, ant_returns, yerr=ant_stds,
                marker='o', label='DAgger', capsize=5, capthick=2,
                linewidth=2, markersize=6, color='blue')

    # Add baseline lines for Ant-v2
    ax1.axhline(y=baselines['Ant-v2']['expert'], color='red', linestyle='--',
               linewidth=2, label=f"Expert ({baselines['Ant-v2']['expert']:.0f})")
    ax1.axhline(y=baselines['Ant-v2']['bc'], color='green', linestyle='--',
               linewidth=2, label=f"BC ({baselines['Ant-v2']['bc']:.0f})")

    ax1.set_xlabel('DAgger Iteration')
    ax1.set_ylabel('Mean Return')
    ax1.set_title('Ant-v2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 7.5)

    # Plot HalfCheetah-v2 (right)
    ax2.errorbar(halfcheetah_iterations, halfcheetah_returns, yerr=halfcheetah_stds,
                marker='o', label='DAgger', capsize=5, capthick=2,
                linewidth=2, markersize=6, color='blue')

    # Add baseline lines for HalfCheetah-v2
    ax2.axhline(y=baselines['HalfCheetah-v2']['expert'], color='red', linestyle='--',
               linewidth=2, label=f"Expert ({baselines['HalfCheetah-v2']['expert']:.0f})")
    ax2.axhline(y=baselines['HalfCheetah-v2']['bc'], color='green', linestyle='--',
               linewidth=2, label=f"BC ({baselines['HalfCheetah-v2']['bc']:.0f})")

    ax2.set_xlabel('DAgger Iteration')
    ax2.set_ylabel('Mean Return')
    ax2.set_title('HalfCheetah-v2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, 7.5)

    plt.tight_layout()

    # Save the figure
    plt.savefig('dagger_results_figure.png', dpi=300, bbox_inches='tight')
    plt.savefig('dagger_results_figure.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Save the data
    results_data = {
        'ant_v2': {
            'iterations': ant_iterations.tolist(),
            'returns': ant_returns.tolist(),
            'stds': ant_stds.tolist(),
            'expert_baseline': baselines['Ant-v2']['expert'],
            'bc_baseline': baselines['Ant-v2']['bc']
        },
        'halfcheetah_v2': {
            'iterations': halfcheetah_iterations.tolist(),
            'returns': halfcheetah_returns.tolist(),
            'stds': halfcheetah_stds.tolist(),
            'expert_baseline': baselines['HalfCheetah-v2']['expert'],
            'bc_baseline': baselines['HalfCheetah-v2']['bc']
        }
    }

    with open('dagger_results_data.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    print("Figure saved as:")
    print("- dagger_results_figure.png")
    print("- dagger_results_figure.pdf")
    print("- dagger_results_data.json")

    return fig

def update_with_actual_data():
    """Helper function to guide updating with actual experimental data"""
    print("To update this script with your actual data:")
    print("\n1. For Ant-v2:")
    print("   - Update ant_returns array with Eval_AverageReturn from each iteration")
    print("   - Update ant_stds array with Eval_StdReturn from each iteration")
    print("   - Update baselines['Ant-v2']['bc'] with your BC result from section 1.3")

    print("\n2. For HalfCheetah-v2:")
    print("   - Run the HalfCheetah DAgger experiment")
    print("   - Update halfcheetah_returns and halfcheetah_stds arrays")
    print("   - Update expert and BC baselines")

    print("\n3. The script will generate the publication-ready Figure 2")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ant-bc', type=float, default=4323.74,
                       help='Ant-v2 BC baseline performance')
    parser.add_argument('--halfcheetah-bc', type=float, default=1500,
                       help='HalfCheetah-v2 BC baseline performance')
    parser.add_argument('--help-update', action='store_true',
                       help='Show help for updating with actual data')

    args = parser.parse_args()

    if args.help_update:
        update_with_actual_data()
    else:
        create_dagger_figure(ant_bc=args.ant_bc, halfcheetah_bc=args.halfcheetah_bc)