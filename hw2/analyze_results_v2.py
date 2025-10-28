#!/usr/bin/env python3
"""
Script to analyze the experiment results for both Analysis 1 and Analysis 2.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def load_experiment_data(data_dir):
    """Load the experiment data from JSON files."""
    small_batch_file = os.path.join(data_dir, 'small_batch_data.json')
    large_batch_file = os.path.join(data_dir, 'large_batch_data.json')
    
    small_batch_data = {}
    large_batch_data = {}
    
    if os.path.exists(small_batch_file):
        with open(small_batch_file, 'r') as f:
            small_batch_data = json.load(f)
    
    if os.path.exists(large_batch_file):
        with open(large_batch_file, 'r') as f:
            large_batch_data = json.load(f)
    
    return small_batch_data, large_batch_data

def extract_config_from_name(exp_name):
    """Extract configuration from experiment name."""
    # Handle different naming patterns
    if 'q1_sb_' in exp_name or 'q1_lb_' in exp_name:
        # New naming: q1_sb_rtg_na_CartPole-v0_... -> rtg_na
        parts = exp_name.split('_')
        if len(parts) >= 4:
            cartpole_idx = next((i for i, part in enumerate(parts) if 'CartPole' in part), len(parts))
            config = '_'.join(parts[2:cartpole_idx])
        else:
            config = exp_name
    else:
        # Old naming: q1_sb_no_rtg_dsa_CartPole-v0_... -> no_rtg_dsa
        parts = exp_name.split('_')
        if len(parts) >= 4:
            cartpole_idx = next((i for i, part in enumerate(parts) if 'CartPole' in part), len(parts))
            config = '_'.join(parts[2:cartpole_idx])
        else:
            config = exp_name
    
    return config

def analyze_experiment_performance(data, batch_type):
    """Analyze the performance of different experiment configurations."""
    print(f"\n{'='*60}")
    print(f"ANALYZING {batch_type.upper()} BATCH EXPERIMENTS")
    print(f"{'='*60}")
    
    results = {}
    
    for exp_name, exp_data in data.items():
        values = np.array(exp_data['values'])
        
        # Extract configuration from experiment name
        config = extract_config_from_name(exp_name)
        
        # Calculate performance metrics
        final_performance = np.mean(values[-10:])  # Average of last 10 iterations
        max_performance = np.max(values)
        convergence_iteration = None
        
        # Find convergence (first iteration where performance > 190)
        for i, val in enumerate(values):
            if val >= 190:
                convergence_iteration = i
                break
        
        results[config] = {
            'final_performance': final_performance,
            'max_performance': max_performance,
            'convergence_iteration': convergence_iteration,
            'all_values': values,
            'exp_name': exp_name
        }
        
        print(f"\n{config} ({exp_name}):")
        print(f"  Final Performance (last 10 avg): {final_performance:.1f}")
        print(f"  Max Performance: {max_performance:.1f}")
        print(f"  Convergence Iteration: {convergence_iteration if convergence_iteration else 'Did not converge'}")
    
    return results

def answer_analysis1_questions(small_results, large_results):
    """Answer Analysis 1 questions about value estimators."""
    print(f"\n{'='*60}")
    print("ANALYSIS 1: VALUE ESTIMATOR COMPARISON")
    print(f"{'='*60}")
    
    print("\nQuestion: Which value estimator has better performance WITHOUT advantage-standardization?")
    print("(Comparing 'rtg_na' vs 'na' - both without advantage standardization)")
    
    small_rtg_na = small_results.get('rtg_na', {}).get('final_performance', 0)
    small_na = small_results.get('na', {}).get('final_performance', 0)
    
    large_rtg_na = large_results.get('rtg_na', {}).get('final_performance', 0)
    large_na = large_results.get('na', {}).get('final_performance', 0)
    
    print(f"\nSmall batch (b=1500):")
    print(f"  - rtg_na (reward-to-go): {small_rtg_na:.1f}")
    print(f"  - na (trajectory-centric): {small_na:.1f}")
    if small_rtg_na > 0 and small_na > 0:
        winner_small = 'rtg_na' if small_rtg_na > small_na else 'na'
        print(f"  - Winner: {winner_small}")
    
    print(f"\nLarge batch (b=6000):")
    print(f"  - rtg_na (reward-to-go): {large_rtg_na:.1f}")
    print(f"  - na (trajectory-centric): {large_na:.1f}")
    if large_rtg_na > 0 and large_na > 0:
        winner_large = 'rtg_na' if large_rtg_na > large_na else 'na'
        print(f"  - Winner: {winner_large}")

def answer_analysis2_questions(small_results, large_results):
    """Answer Analysis 2 questions about advantage standardization."""
    print(f"\n{'='*60}")
    print("ANALYSIS 2: ADVANTAGE STANDARDIZATION COMPARISON")
    print(f"{'='*60}")
    
    print("\nQuestion: Did advantage standardization help?")
    
    # Compare rtg_na vs rtg_dsa (both with reward-to-go)
    small_rtg_na = small_results.get('rtg_na', {}).get('final_performance', 0)
    small_rtg_dsa = small_results.get('rtg_dsa', {}).get('final_performance', 0)
    
    large_rtg_na = large_results.get('rtg_na', {}).get('final_performance', 0)
    large_rtg_dsa = large_results.get('rtg_dsa', {}).get('final_performance', 0)
    
    print(f"\nWith reward-to-go (rtg_na vs rtg_dsa):")
    print(f"  Small batch (b=1500):")
    print(f"    - rtg_na (no adv std): {small_rtg_na:.1f}")
    print(f"    - rtg_dsa (with adv std): {small_rtg_dsa:.1f}")
    if small_rtg_na > 0 and small_rtg_dsa > 0:
        diff = small_rtg_dsa - small_rtg_na
        print(f"    - Advantage standardization {'helped' if diff > 0 else 'hurt'} ({diff:+.1f})")
    
    print(f"  Large batch (b=6000):")
    print(f"    - rtg_na (no adv std): {large_rtg_na:.1f}")
    print(f"    - rtg_dsa (with adv std): {large_rtg_dsa:.1f}")
    if large_rtg_na > 0 and large_rtg_dsa > 0:
        diff = large_rtg_dsa - large_rtg_na
        print(f"    - Advantage standardization {'helped' if diff > 0 else 'hurt'} ({diff:+.1f})")
    
    # Compare na vs dsa (both trajectory-centric)
    small_na = small_results.get('na', {}).get('final_performance', 0)
    small_dsa = small_results.get('dsa', {}).get('final_performance', 0)
    
    large_na = large_results.get('na', {}).get('final_performance', 0)
    large_dsa = large_results.get('dsa', {}).get('final_performance', 0)
    
    print(f"\nWith trajectory-centric (na vs dsa):")
    print(f"  Small batch (b=1500):")
    print(f"    - na (no adv std): {small_na:.1f}")
    print(f"    - dsa (with adv std): {small_dsa:.1f}")
    if small_na > 0 and small_dsa > 0:
        diff = small_dsa - small_na
        print(f"    - Advantage standardization {'helped' if diff > 0 else 'hurt'} ({diff:+.1f})")
    
    print(f"  Large batch (b=6000):")
    print(f"    - na (no adv std): {large_na:.1f}")
    print(f"    - dsa (with adv std): {large_dsa:.1f}")
    if large_na > 0 and large_dsa > 0:
        diff = large_dsa - large_na
        print(f"    - Advantage standardization {'helped' if diff > 0 else 'hurt'} ({diff:+.1f})")

def create_comparison_plot(small_results, large_results, output_dir, analysis_name):
    """Create a comparison plot showing the key differences."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get configurations that exist in both small and large batch
    small_configs = list(small_results.keys())
    large_configs = list(large_results.keys())
    
    # Small batch comparison
    small_perfs = [small_results.get(config, {}).get('final_performance', 0) for config in small_configs]
    
    bars1 = ax1.bar(small_configs, small_perfs, color=['blue', 'red', 'green', 'orange'][:len(small_configs)], alpha=0.7)
    ax1.set_title(f'Small Batch (b=1500) - {analysis_name}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Return')
    ax1.set_ylim(0, 200)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, perf in zip(bars1, small_perfs):
        if perf > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{perf:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Large batch comparison
    large_perfs = [large_results.get(config, {}).get('final_performance', 0) for config in large_configs]
    
    bars2 = ax2.bar(large_configs, large_perfs, color=['blue', 'red', 'green', 'orange'][:len(large_configs)], alpha=0.7)
    ax2.set_title(f'Large Batch (b=6000) - {analysis_name}', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Return')
    ax2.set_ylim(0, 200)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, perf in zip(bars2, large_perfs):
        if perf > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{perf:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_comparison_{analysis_name.lower().replace(" ", "_")}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/performance_comparison_{analysis_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_dir}/performance_comparison_{analysis_name.lower().replace(' ', '_')}.pdf/png")
    plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--data_dir', default='plots', help='Directory containing the data files')
    parser.add_argument('--analysis', choices=['a1', 'a2', 'both'], default='both', 
                       help='Which analysis to run (a1=value estimators, a2=advantage standardization, both=all)')
    
    args = parser.parse_args()
    
    print("HW2 Experiment Analysis")
    print("=" * 60)
    
    # Load data
    small_batch_data, large_batch_data = load_experiment_data(args.data_dir)
    
    if not small_batch_data and not large_batch_data:
        print(f"No data found in {args.data_dir}")
        return
    
    # Analyze performance
    small_results = analyze_experiment_performance(small_batch_data, "small")
    large_results = analyze_experiment_performance(large_batch_data, "large")
    
    # Answer questions based on analysis type
    if args.analysis in ['a1', 'both']:
        answer_analysis1_questions(small_results, large_results)
        create_comparison_plot(small_results, large_results, args.data_dir, "Value Estimator Comparison")
    
    if args.analysis in ['a2', 'both']:
        answer_analysis2_questions(small_results, large_results)
        create_comparison_plot(small_results, large_results, args.data_dir, "Advantage Standardization Comparison")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
