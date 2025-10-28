#!/usr/bin/env python3

import pickle
import numpy as np
import os

def analyze_expert_data():
    """Analyze expert data for all environments and calculate mean/std of returns."""
    
    environments = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Walker2d-v2']
    expert_data_dir = './hw1/rob831/expert_data'
    
    results = {}
    
    for env in environments:
        data_file = os.path.join(expert_data_dir, f'expert_data_{env}.pkl')
        
        if os.path.exists(data_file):
            print(f"\nAnalyzing {env}...")
            
            with open(data_file, 'rb') as f:
                expert_data = pickle.load(f)
            
            # Extract returns from trajectories
            returns = []
            for path in expert_data:
                if 'reward' in path:
                    # Sum rewards for each trajectory to get return
                    trajectory_return = np.sum(path['reward'])
                    returns.append(trajectory_return)
            
            if returns:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                num_trajectories = len(returns)
                
                results[env] = {
                    'mean': mean_return,
                    'std': std_return,
                    'num_trajectories': num_trajectories,
                    'returns': returns
                }
                
                print(f"  Number of trajectories: {num_trajectories}")
                print(f"  Mean return: {mean_return:.2f}")
                print(f"  Std return: {std_return:.2f}")
                print(f"  Individual returns: {returns[:5]}...")  # Show first 5
            else:
                print(f"  No reward data found in {env}")
        else:
            print(f"Data file not found: {data_file}")
    
    # Print summary table
    print("\n" + "="*60)
    print("EXPERT DATA ANALYSIS - TABLE 1")
    print("="*60)
    print(f"{'Environment':<15} {'Mean Return':<12} {'Std Return':<12} {'# Traj':<8}")
    print("-"*60)
    
    for env in environments:
        if env in results:
            mean = results[env]['mean']
            std = results[env]['std']
            num_traj = results[env]['num_trajectories']
            print(f"{env:<15} {mean:<12.2f} {std:<12.2f} {num_traj:<8}")
        else:
            print(f"{env:<15} {'N/A':<12} {'N/A':<12} {'N/A':<8}")
    
    return results

if __name__ == "__main__":
    results = analyze_expert_data()