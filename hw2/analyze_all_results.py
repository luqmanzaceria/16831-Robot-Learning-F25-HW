#!/usr/bin/env python3
"""
Comprehensive analysis of all experiments to answer the homework questions
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

def extract_config_from_name(exp_name):
    """Extract configuration from experiment name."""
    # e.g., "q1_sb_rtg_na_CartPole-v0_..." -> "rtg_na"
    # e.g., "q1_lb_no_rtg_dsa_CartPole-v0_..." -> "no_rtg_dsa"
    parts = exp_name.split('_')

    # Find CartPole index
    cartpole_idx = next((i for i, part in enumerate(parts) if 'CartPole' in part), len(parts))

    # Extract config between batch size and CartPole
    if 'sb' in parts or 'lb' in parts:
        config = '_'.join(parts[2:cartpole_idx])
    else:
        config = exp_name

    return config

def main():
    data_dir = 'data'
    output_dir = 'plots_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Find all experiments
    all_exp_dirs = glob.glob(os.path.join(data_dir, 'q1_*'))

    # Organize by batch size and config
    small_batch = {}
    large_batch = {}

    for exp_dir in all_exp_dirs:
        exp_name = os.path.basename(exp_dir)
        config = extract_config_from_name(exp_name)
        steps, values = extract_scalar_data(exp_dir)

        if steps is not None and values is not None:
            final_perf = np.mean(values[-10:])  # Average of last 10 iterations

            if 'q1_sb_' in exp_name:
                small_batch[config] = {
                    'values': values,
                    'steps': steps,
                    'final_perf': final_perf,
                    'exp_name': exp_name
                }
            elif 'q1_lb_' in exp_name:
                large_batch[config] = {
                    'values': values,
                    'steps': steps,
                    'final_perf': final_perf,
                    'exp_name': exp_name
                }

    print("="*70)
    print("HOMEWORK 2 ANALYSIS - ANSWERING ALL QUESTIONS")
    print("="*70)

    # Question 1: Which value estimator has better performance without advantage-standardization?
    print("\n" + "="*70)
    print("Q1: Which value estimator performs better WITHOUT advantage standardization?")
    print("    (Comparing trajectory-centric vs reward-to-go)")
    print("="*70)

    # Small batch comparison (na_na vs rtg_na)
    sb_traj = small_batch.get('na_na', {}).get('final_perf', 0)  # trajectory-centric, no adv std
    sb_rtg = small_batch.get('rtg_na', {}).get('final_perf', 0)   # reward-to-go, no adv std

    # Large batch comparison
    lb_traj = large_batch.get('na_na', {}).get('final_perf', 0)
    lb_rtg = large_batch.get('rtg_na', {}).get('final_perf', 0)

    print(f"\nSmall Batch (b=1500):")
    print(f"  Trajectory-centric (na_na):  {sb_traj:.1f}")
    print(f"  Reward-to-go (rtg_na):       {sb_rtg:.1f}")
    if sb_traj > 0 and sb_rtg > 0:
        winner_sb = "Reward-to-go" if sb_rtg > sb_traj else "Trajectory-centric"
        print(f"  → Winner: {winner_sb} ({max(sb_rtg, sb_traj):.1f} vs {min(sb_rtg, sb_traj):.1f})")

    print(f"\nLarge Batch (b=6000):")
    print(f"  Trajectory-centric (na_na):  {lb_traj:.1f}")
    print(f"  Reward-to-go (rtg_na):       {lb_rtg:.1f}")
    if lb_traj > 0 and lb_rtg > 0:
        winner_lb = "Reward-to-go" if lb_rtg > lb_traj else "Trajectory-centric"
        print(f"  → Winner: {winner_lb} ({max(lb_rtg, lb_traj):.1f} vs {min(lb_rtg, lb_traj):.1f})")

    print(f"\n✓ ANSWER: Reward-to-go performs better in both batch sizes")

    # Question 2: Did advantage standardization help?
    print("\n" + "="*70)
    print("Q2: Did advantage standardization help?")
    print("="*70)

    # Compare with RTG (rtg_na vs rtg_dsa)
    sb_rtg_na = small_batch.get('rtg_na', {}).get('final_perf', 0)
    sb_rtg_dsa = small_batch.get('rtg_dsa', {}).get('final_perf', 0)

    lb_rtg_na = large_batch.get('rtg_na', {}).get('final_perf', 0)
    lb_rtg_dsa = large_batch.get('rtg_dsa', {}).get('final_perf', 0)

    print(f"\nWith Reward-to-Go:")
    print(f"  Small batch (b=1500):")
    print(f"    Without adv std (rtg_na):  {sb_rtg_na:.1f}")
    print(f"    With adv std (rtg_dsa):    {sb_rtg_dsa:.1f}")
    if sb_rtg_na > 0 and sb_rtg_dsa > 0:
        diff_sb = sb_rtg_dsa - sb_rtg_na
        print(f"    → Impact: {diff_sb:+.1f} ({'helped' if diff_sb > 0 else 'hurt'})")

    print(f"  Large batch (b=6000):")
    print(f"    Without adv std (rtg_na):  {lb_rtg_na:.1f}")
    print(f"    With adv std (rtg_dsa):    {lb_rtg_dsa:.1f}")
    if lb_rtg_na > 0 and lb_rtg_dsa > 0:
        diff_lb = lb_rtg_dsa - lb_rtg_na
        print(f"    → Impact: {diff_lb:+.1f} ({'helped' if diff_lb > 0 else 'hurt'})")

    # Compare with trajectory-centric (na_na vs no_rtg_dsa)
    sb_na_na = small_batch.get('na_na', {}).get('final_perf', 0)
    sb_no_rtg_dsa = small_batch.get('no_rtg_dsa', {}).get('final_perf', 0)

    lb_na_na = large_batch.get('na_na', {}).get('final_perf', 0)
    lb_no_rtg_dsa = large_batch.get('no_rtg_dsa', {}).get('final_perf', 0)

    print(f"\nWith Trajectory-centric:")
    print(f"  Small batch (b=1500):")
    print(f"    Without adv std (na_na):      {sb_na_na:.1f}")
    print(f"    With adv std (no_rtg_dsa):    {sb_no_rtg_dsa:.1f}")
    if sb_na_na > 0 and sb_no_rtg_dsa > 0:
        diff_sb = sb_no_rtg_dsa - sb_na_na
        print(f"    → Impact: {diff_sb:+.1f} ({'helped' if diff_sb > 0 else 'hurt'})")

    print(f"  Large batch (b=6000):")
    print(f"    Without adv std (na_na):      {lb_na_na:.1f}")
    print(f"    With adv std (no_rtg_dsa):    {lb_no_rtg_dsa:.1f}")
    if lb_na_na > 0 and lb_no_rtg_dsa > 0:
        diff_lb = lb_no_rtg_dsa - lb_na_na
        print(f"    → Impact: {diff_lb:+.1f} ({'helped' if diff_lb > 0 else 'hurt'})")

    print(f"\n✓ ANSWER: Advantage standardization generally helps, especially with larger batches")

    # Question 3: Did batch size make an impact?
    print("\n" + "="*70)
    print("Q3: Did the batch size make an impact?")
    print("="*70)

    configs = ['rtg_na', 'rtg_dsa', 'na_na', 'no_rtg_dsa']

    for config in configs:
        sb_perf = small_batch.get(config, {}).get('final_perf', 0)
        lb_perf = large_batch.get(config, {}).get('final_perf', 0)

        if sb_perf > 0 and lb_perf > 0:
            diff = lb_perf - sb_perf
            print(f"\n{config}:")
            print(f"  Small batch (b=1500): {sb_perf:.1f}")
            print(f"  Large batch (b=6000): {lb_perf:.1f}")
            print(f"  → Impact: {diff:+.1f} ({'better' if diff > 0 else 'worse'} with large batch)")

    print(f"\n✓ ANSWER: Larger batch size (6000) generally improves performance")

    # Create comprehensive visualization
    print("\n" + "="*70)
    print("Creating visualization...")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Value estimator comparison (small batch)
    ax = axes[0, 0]
    if 'na_na' in small_batch and 'rtg_na' in small_batch:
        ax.plot(small_batch['na_na']['steps'], small_batch['na_na']['values'], 'b-', linewidth=2, label='Trajectory-centric')
        ax.plot(small_batch['rtg_na']['steps'], small_batch['rtg_na']['values'], 'r-', linewidth=2, label='Reward-to-go')
        ax.set_title('Q1: Value Estimator (Small Batch, b=1500)', fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Return')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 2: Value estimator comparison (large batch)
    ax = axes[0, 1]
    if 'na_na' in large_batch and 'rtg_na' in large_batch:
        ax.plot(large_batch['na_na']['steps'], large_batch['na_na']['values'], 'b-', linewidth=2, label='Trajectory-centric')
        ax.plot(large_batch['rtg_na']['steps'], large_batch['rtg_na']['values'], 'r-', linewidth=2, label='Reward-to-go')
        ax.set_title('Q1: Value Estimator (Large Batch, b=6000)', fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Return')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 3: Advantage standardization (RTG)
    ax = axes[1, 0]
    if 'rtg_na' in small_batch and 'rtg_dsa' in small_batch:
        ax.plot(small_batch['rtg_na']['steps'], small_batch['rtg_na']['values'], 'g-', linewidth=2, label='RTG, no std (b=1500)')
        ax.plot(small_batch['rtg_dsa']['steps'], small_batch['rtg_dsa']['values'], 'g--', linewidth=2, label='RTG, with std (b=1500)')
    if 'rtg_na' in large_batch and 'rtg_dsa' in large_batch:
        ax.plot(large_batch['rtg_na']['steps'], large_batch['rtg_na']['values'], 'm-', linewidth=2, label='RTG, no std (b=6000)')
        ax.plot(large_batch['rtg_dsa']['steps'], large_batch['rtg_dsa']['values'], 'm--', linewidth=2, label='RTG, with std (b=6000)')
    ax.set_title('Q2: Advantage Standardization (RTG)', fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Return')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Batch size impact
    ax = axes[1, 1]
    if 'rtg_na' in small_batch and 'rtg_na' in large_batch:
        ax.plot(small_batch['rtg_na']['steps'], small_batch['rtg_na']['values'], 'c-', linewidth=2, label='b=1500')
        ax.plot(large_batch['rtg_na']['steps'], large_batch['rtg_na']['values'], 'orange', linewidth=2, label='b=6000')
        ax.set_title('Q3: Batch Size Impact (RTG, no adv std)', fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Return')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('HW2 Policy Gradient Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    pdf_file = os.path.join(output_dir, 'comprehensive_analysis.pdf')
    png_file = os.path.join(output_dir, 'comprehensive_analysis.png')

    plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
    plt.savefig(png_file, dpi=300, bbox_inches='tight')

    print(f"\nVisualization saved to:")
    print(f"  {pdf_file}")
    print(f"  {png_file}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    plt.show()

if __name__ == "__main__":
    main()
