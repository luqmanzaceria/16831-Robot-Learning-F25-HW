#!/usr/bin/env python3
"""
Analyze InvertedPendulum-v4 experiments to find optimal batch size and learning rate
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import re

def extract_scalar_data(log_dir, scalar_name='Eval_AverageReturn'):
    """Extract scalar data from TensorBoard log directory."""
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
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

def parse_config_from_name(exp_name):
    """Parse batch size and learning rate from experiment name."""
    # e.g., "q2_b1000_r0.01_InvertedPendulum-v4_..."
    batch_match = re.search(r'_b(\d+)', exp_name)
    lr_match = re.search(r'_r([0-9.]+)', exp_name)

    batch = int(batch_match.group(1)) if batch_match else None
    lr = float(lr_match.group(1)) if lr_match else None

    return batch, lr

def check_success(values, threshold=1000, max_iters=100):
    """Check if experiment reached threshold within max iterations."""
    if len(values) > max_iters:
        values = values[:max_iters]

    # Find first iteration that reaches threshold
    for i, val in enumerate(values):
        if val >= threshold:
            return True, i

    return False, None

def main():
    data_dir = 'data'
    output_dir = 'plots_pendulum'
    os.makedirs(output_dir, exist_ok=True)

    # Find all InvertedPendulum experiments
    all_exp_dirs = glob.glob(os.path.join(data_dir, 'q2_*'))

    results = []

    print("="*70)
    print("ANALYZING INVERTED PENDULUM EXPERIMENTS")
    print("="*70)

    for exp_dir in all_exp_dirs:
        exp_name = os.path.basename(exp_dir)
        batch, lr = parse_config_from_name(exp_name)

        if batch is None or lr is None:
            continue

        steps, values = extract_scalar_data(exp_dir)

        if steps is not None and values is not None:
            success, convergence_iter = check_success(values)

            results.append({
                'exp_name': exp_name,
                'batch': batch,
                'lr': lr,
                'success': success,
                'convergence_iter': convergence_iter,
                'steps': steps,
                'values': values,
                'max_value': np.max(values),
                'final_value': np.mean(values[-10:])
            })

            status = f"✓ SUCCESS at iter {convergence_iter}" if success else "✗ FAILED"
            print(f"\n{exp_name}")
            print(f"  Batch: {batch}, LR: {lr}")
            print(f"  {status}")
            print(f"  Max value: {np.max(values):.1f}, Final: {np.mean(values[-10:]):.1f}")

    # Find successful experiments
    successful = [r for r in results if r['success']]

    if not successful:
        print("\n" + "="*70)
        print("NO SUCCESSFUL CONFIGURATIONS FOUND")
        print("Try adjusting the search parameters")
        print("="*70)
        return

    # Sort by: smallest batch first, then largest learning rate
    successful.sort(key=lambda x: (x['batch'], -x['lr']))

    print("\n" + "="*70)
    print("SUCCESSFUL CONFIGURATIONS (sorted by batch size, then LR desc)")
    print("="*70)

    for r in successful:
        print(f"\nBatch: {r['batch']:5d}, LR: {r['lr']:.4f} -> Converged at iter {r['convergence_iter']}")

    # Find optimal configuration (smallest batch, largest LR)
    optimal = successful[0]

    print("\n" + "="*70)
    print("OPTIMAL CONFIGURATION")
    print("="*70)
    print(f"\nb* (smallest batch size): {optimal['batch']}")
    print(f"r* (largest learning rate): {optimal['lr']}")
    print(f"Convergence iteration: {optimal['convergence_iter']}")
    print(f"\nCommand line:")
    print(f"python rob831/scripts/run_hw2.py \\")
    print(f"    --env_name InvertedPendulum-v4 \\")
    print(f"    --ep_len 1000 \\")
    print(f"    --discount 0.9 \\")
    print(f"    -n 100 \\")
    print(f"    -l 2 \\")
    print(f"    -s 64 \\")
    print(f"    -b {optimal['batch']} \\")
    print(f"    -lr {optimal['lr']} \\")
    print(f"    -rtg \\")
    print(f"    --exp_name q2_b{optimal['batch']}_r{optimal['lr']}")

    # Create visualization
    print("\n" + "="*70)
    print("Creating visualizations...")
    print("="*70)

    # Plot 1: Learning curve for optimal configuration
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(optimal['steps'], optimal['values'], 'b-', linewidth=2)
    ax.axhline(y=1000, color='r', linestyle='--', linewidth=2, label='Target (1000)')
    ax.axvline(x=optimal['convergence_iter'], color='g', linestyle='--', linewidth=2,
               label=f'Convergence (iter {optimal["convergence_iter"]})')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Average Return', fontsize=12)
    ax.set_title(f'Optimal Config: b*={optimal["batch"]}, r*={optimal["lr"]}',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    # Plot 2: Comparison of successful configurations
    ax = axes[1]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(successful)))

    for i, r in enumerate(successful[:10]):  # Plot top 10
        label = f"b={r['batch']}, lr={r['lr']}"
        ax.plot(r['steps'], r['values'], linewidth=2, label=label, color=colors[i], alpha=0.7)

    ax.axhline(y=1000, color='r', linestyle='--', linewidth=2, label='Target (1000)')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Average Return', fontsize=12)
    ax.set_title('Top Configurations Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    plt.suptitle('InvertedPendulum-v4: Hyperparameter Search Results',
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()

    pdf_file = os.path.join(output_dir, 'inverted_pendulum_results.pdf')
    png_file = os.path.join(output_dir, 'inverted_pendulum_results.png')

    plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
    plt.savefig(png_file, dpi=300, bbox_inches='tight')

    print(f"\nVisualization saved to:")
    print(f"  {pdf_file}")
    print(f"  {png_file}")

    # Create a separate plot just for the optimal configuration
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(optimal['steps'], optimal['values'], 'b-', linewidth=2.5, marker='o',
            markersize=4, markevery=5)
    ax.axhline(y=1000, color='r', linestyle='--', linewidth=2, label='Target (1000)')
    ax.axvline(x=optimal['convergence_iter'], color='g', linestyle='--', linewidth=2,
               label=f'First success at iteration {optimal["convergence_iter"]}')
    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('Average Return', fontsize=13)
    ax.set_title(f'InvertedPendulum-v4: Optimal Configuration\n'
                 f'Batch Size b*={optimal["batch"]}, Learning Rate r*={optimal["lr"]}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(1200, np.max(optimal['values']) * 1.1))

    plt.tight_layout()

    pdf_file_opt = os.path.join(output_dir, 'optimal_config_learning_curve.pdf')
    png_file_opt = os.path.join(output_dir, 'optimal_config_learning_curve.png')

    plt.savefig(pdf_file_opt, dpi=300, bbox_inches='tight')
    plt.savefig(png_file_opt, dpi=300, bbox_inches='tight')

    print(f"\nOptimal config plot saved to:")
    print(f"  {pdf_file_opt}")
    print(f"  {png_file_opt}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    plt.show()

if __name__ == "__main__":
    main()
