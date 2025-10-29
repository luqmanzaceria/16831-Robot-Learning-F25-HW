#!/usr/bin/env python3
"""
Script to plot DQN vs DDQN comparison with error bars across three seeds.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path

def parse_stats_file(stats_file):
    """Parse stats JSON file to get episode rewards."""
    with open(stats_file, 'r') as f:
        data = json.load(f)
    
    episode_rewards = data['episode_rewards']
    episode_lengths = data['episode_lengths']
    
    # Calculate cumulative sum of timesteps (using episode lengths)
    timesteps = np.cumsum(episode_lengths)
    return timesteps, episode_rewards


def load_all_runs(data_dir, exp_prefix):
    """Load all runs matching an experiment prefix."""
    # Find all directories matching the pattern
    pattern = os.path.join(data_dir, f'{exp_prefix}_*')
    run_dirs = sorted(glob(pattern))
    
    all_timesteps = []
    all_rewards = []
    
    for run_dir in run_dirs:
        stats_files = glob(os.path.join(run_dir, 'gym', 'openai*.stats.json'))
        if stats_files:
            timesteps, rewards = parse_stats_file(stats_files[0])
            all_timesteps.append(timesteps)
            all_rewards.append(rewards)
    
    return all_timesteps, all_rewards


def interpolate_to_common_timesteps(timestep_arrays, reward_arrays):
    """Interpolate all runs to a common set of timesteps."""
    # Find the minimum max timesteps (to align all runs)
    min_timesteps = min(arr[-1] for arr in timestep_arrays)
    
    # Create a common grid from 0 to min_timesteps with 500 points
    common_timesteps = np.linspace(0, min_timesteps, 500)
    
    interpolated_rewards = []
    for timesteps, rewards in zip(timestep_arrays, reward_arrays):
        # Clip timesteps to the common range
        mask = timesteps <= min_timesteps
        clipped_timesteps = timesteps[mask]
        clipped_rewards = np.array(rewards)[mask]
        
        # Interpolate to common grid
        interp_rewards = np.interp(common_timesteps, clipped_timesteps, clipped_rewards)
        interpolated_rewards.append(interp_rewards)
    
    return common_timesteps, np.array(interpolated_rewards)


def plot_comparison(data_dir, output_file):
    """Create comparison plot for DQN vs DDQN."""
    # Load DQN runs
    dqn_timesteps, dqn_rewards = load_all_runs(data_dir, 'q1_dqn')
    ddqn_timesteps, ddqn_rewards = load_all_runs(data_dir, 'q1_doubledqn')
    
    print(f"Found {len(dqn_timesteps)} DQN runs")
    print(f"Found {len(ddqn_timesteps)} DDQN runs")
    
    # Find the overall minimum max timesteps across all runs
    all_timesteps = dqn_timesteps + ddqn_timesteps
    min_max_timesteps = min(arr[-1] for arr in all_timesteps)
    
    # Create a common grid
    common_timesteps = np.linspace(0, min_max_timesteps, 500)
    
    # Interpolate all runs to common timesteps
    dqn_interp = []
    for timesteps, rewards in zip(dqn_timesteps, dqn_rewards):
        mask = timesteps <= min_max_timesteps
        interp_rewards = np.interp(common_timesteps, timesteps[mask], np.array(rewards)[mask])
        dqn_interp.append(interp_rewards)
    
    ddqn_interp = []
    for timesteps, rewards in zip(ddqn_timesteps, ddqn_rewards):
        mask = timesteps <= min_max_timesteps
        interp_rewards = np.interp(common_timesteps, timesteps[mask], np.array(rewards)[mask])
        ddqn_interp.append(interp_rewards)
    
    dqn_interp = np.array(dqn_interp)
    ddqn_interp = np.array(ddqn_interp)
    
    # Calculate mean and std across seeds
    dqn_mean = np.mean(dqn_interp, axis=0)
    dqn_std = np.std(dqn_interp, axis=0)
    
    ddqn_mean = np.mean(ddqn_interp, axis=0)
    ddqn_std = np.std(ddqn_interp, axis=0)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Downsample slightly to reduce clutter - show every Nth point
    step = 20
    ts_downsampled = common_timesteps[::step]
    dqn_mean_ds = dqn_mean[::step]
    dqn_std_ds = dqn_std[::step]
    ddqn_mean_ds = ddqn_mean[::step]
    ddqn_std_ds = ddqn_std[::step]
    
    # Plot DQN with error bars
    plt.errorbar(ts_downsampled, dqn_mean_ds, yerr=dqn_std_ds, 
                 label='DQN', linewidth=2, color='blue', alpha=0.8,
                 capsize=3, capthick=2)
    
    # Plot DDQN with error bars
    plt.errorbar(ts_downsampled, ddqn_mean_ds, yerr=ddqn_std_ds, 
                 label='Double DQN', linewidth=2, color='red', alpha=0.8,
                 capsize=3, capthick=2)
    
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Average Per-Episode Reward', fontsize=12)
    plt.title('DQN vs Double DQN on LunarLander-v3', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Use scientific notation for x-axis if needed
    if common_timesteps[-1] > 100000:
        ax = plt.gca()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0e}'))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Also save a PDF version
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"Plot saved to {pdf_file}")
    
    plt.show()


if __name__ == '__main__':
    data_dir = '/Users/luqmanzaceria/mscv/fall25/CMU-16-831-Robot-Learning/16831-Robot-Learning-F25-HW/hw3/data'
    output_file = 'dqn_vs_ddqn_comparison.png'
    
    plot_comparison(data_dir, output_file)

