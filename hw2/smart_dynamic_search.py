#!/usr/bin/env python3
"""
Smart dynamic search for optimal batch size and learning rate for InvertedPendulum-v4
Goal: Find smallest batch size b* and largest learning rate r* that reaches 1000 in <100 iterations
Strategy: Binary search with validation that b* and r* work together
"""

import os
import subprocess
import time
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse

def run_experiment(batch_size, learning_rate, exp_name):
    """Run a single experiment and return success status."""
    print(f"Testing: batch={batch_size}, lr={learning_rate}")
    
    cmd = [
        'python', 'rob831/scripts/run_hw2.py',
        '--env_name', 'InvertedPendulum-v4',
        '--ep_len', '1000',
        '--discount', '0.9',
        '-n', '100',
        '-l', '2',
        '-s', '64',
        '-b', str(batch_size),
        '-lr', str(learning_rate),
        '-rtg',
        '--exp_name', exp_name
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        if result.returncode == 0:
            print(f"✓ Experiment {exp_name} completed successfully")
            return True
        else:
            print(f"✗ Experiment {exp_name} failed with return code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ Experiment {exp_name} timed out")
        return False
    except Exception as e:
        print(f"✗ Experiment {exp_name} failed with error: {e}")
        return False

def check_success(exp_name, threshold=1000, max_iters=100):
    """Check if experiment reached threshold within max iterations."""
    data_dir = 'data'
    exp_dirs = glob.glob(os.path.join(data_dir, f"{exp_name}_*"))
    
    if not exp_dirs:
        print(f"No experiment directory found for {exp_name}")
        return False, None
    
    exp_dir = exp_dirs[0]
    event_files = glob.glob(os.path.join(exp_dir, "events.out.tfevents.*"))
    
    if not event_files:
        print(f"No TensorBoard log found for {exp_name}")
        return False, None
    
    try:
        event_file = event_files[0]
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        scalar_tags = ea.Tags()['scalars']
        
        # Try different possible scalar names
        scalar_name = None
        for name in ['Eval_AverageReturn', 'Train_AverageReturn', 'AverageReturn']:
            if name in scalar_tags:
                scalar_name = name
                break
        
        if scalar_name is None:
            print(f"No return scalar found in {exp_name}")
            return False, None
        
        scalar_events = ea.Scalars(scalar_name)
        values = [s.value for s in scalar_events]
        
        if len(values) == 0:
            print(f"No data points found for {exp_name}")
            return False, None
        
        # Check if we reached threshold within max_iters
        for i, val in enumerate(values[:max_iters]):
            if val >= threshold:
                print(f"✓ {exp_name} reached {threshold} at iteration {i+1}")
                return True, i+1
        
        max_val = max(values[:max_iters]) if len(values) >= max_iters else max(values)
        print(f"✗ {exp_name} max value: {max_val:.1f} (target: {threshold})")
        return False, None
        
    except Exception as e:
        print(f"Error parsing {exp_name}: {e}")
        return False, None

def binary_search_batch_size(min_batch, max_batch, learning_rate):
    """Binary search for smallest working batch size."""
    print(f"\nBinary searching batch size from {min_batch} to {max_batch} with lr={learning_rate}")
    
    optimal_batch = None
    
    while min_batch <= max_batch:
        mid_batch = (min_batch + max_batch) // 2
        exp_name = f"q2_b{mid_batch}_r{learning_rate}_batch_search"
        
        print(f"\nTesting batch size: {mid_batch}")
        if run_experiment(mid_batch, learning_rate, exp_name):
            success, convergence_iter = check_success(exp_name)
            if success:
                optimal_batch = mid_batch
                max_batch = mid_batch - 1
                print(f"✓ Batch size {mid_batch} works! (converged at iter {convergence_iter})")
            else:
                min_batch = mid_batch + 1
                print(f"✗ Batch size {mid_batch} failed to reach target")
        else:
            min_batch = mid_batch + 1
            print(f"✗ Batch size {mid_batch} experiment failed")
    
    return optimal_batch

def binary_search_learning_rate(batch_size, min_lr, max_lr, precision=0.001):
    """Binary search for largest working learning rate."""
    print(f"\nBinary searching learning rate from {min_lr} to {max_lr} with batch={batch_size}")
    
    optimal_lr = None
    
    while max_lr - min_lr > precision:
        mid_lr = (min_lr + max_lr) / 2
        exp_name = f"q2_b{batch_size}_r{mid_lr}_lr_search"
        
        print(f"\nTesting learning rate: {mid_lr:.4f}")
        if run_experiment(batch_size, mid_lr, exp_name):
            success, convergence_iter = check_success(exp_name)
            if success:
                optimal_lr = mid_lr
                min_lr = mid_lr + precision
                print(f"✓ Learning rate {mid_lr:.4f} works! (converged at iter {convergence_iter})")
            else:
                max_lr = mid_lr - precision
                print(f"✗ Learning rate {mid_lr:.4f} failed to reach target")
        else:
            max_lr = mid_lr - precision
            print(f"✗ Learning rate {mid_lr:.4f} experiment failed")
    
    return optimal_lr

def validate_combination(batch_size, learning_rate):
    """Validate that the batch size and learning rate work together."""
    exp_name = f"q2_b{batch_size}_r{learning_rate}_validation"
    print(f"\nValidating combination: batch={batch_size}, lr={learning_rate}")
    
    if run_experiment(batch_size, learning_rate, exp_name):
        success, convergence_iter = check_success(exp_name)
        if success:
            print(f"✓ Combination validated! Converged at iteration {convergence_iter}")
            return True, convergence_iter
        else:
            print(f"✗ Combination failed to reach target")
            return False, None
    else:
        print(f"✗ Combination experiment failed")
        return False, None

def main():
    print("Smart Dynamic Search for InvertedPendulum-v4")
    print("=" * 50)
    print("Goal: Find smallest batch size b* and largest learning rate r*")
    print("that reaches 1000 average return in <100 iterations")
    print("Search range: batch_size [1, 1000], learning_rate [0.001, 1.0]")
    print("")
    
    # Step 1: Find smallest batch size with a reasonable learning rate
    print("Step 1: Finding smallest batch size (b*)")
    print("=" * 40)
    print("Using learning rate = 0.01 as reference")
    
    test_lr = 0.01
    optimal_batch = binary_search_batch_size(1, 1000, test_lr)
    
    if optimal_batch is None:
        print("No working batch size found in range [1, 1000]!")
        print("Trying with a smaller learning rate...")
        test_lr = 0.001
        optimal_batch = binary_search_batch_size(1, 1000, test_lr)
        
        if optimal_batch is None:
            print("Still no working batch size found! The environment might be too difficult.")
            return
    
    print(f"\n✓ Smallest working batch size (b*): {optimal_batch}")
    
    # Step 2: Find largest learning rate with the optimal batch size
    print(f"\nStep 2: Finding largest learning rate (r*) with batch={optimal_batch}")
    print("=" * 60)
    
    # Search from 0.001 to 1.0 (very large learning rate)
    optimal_lr = binary_search_learning_rate(optimal_batch, 0.001, 1.0)
    
    if optimal_lr is None:
        print("No working learning rate found! Trying smaller range...")
        optimal_lr = binary_search_learning_rate(optimal_batch, 0.001, 0.1)
        
        if optimal_lr is None:
            print("Still no working learning rate found!")
            return
    
    print(f"\n✓ Largest working learning rate (r*): {optimal_lr:.4f}")
    
    # Step 3: Validate that b* and r* work together
    print(f"\nStep 3: Validating that b*={optimal_batch} and r*={optimal_lr:.4f} work together")
    print("=" * 70)
    
    validation_success, convergence_iter = validate_combination(optimal_batch, optimal_lr)
    
    if validation_success:
        print(f"\n🎉 SUCCESS! Optimal configuration validated!")
        print(f"   Batch size (b*): {optimal_batch}")
        print(f"   Learning rate (r*): {optimal_lr:.4f}")
        print(f"   Convergence: iteration {convergence_iter}")
        
        print(f"\nCommand line for optimal configuration:")
        print(f"python rob831/scripts/run_hw2.py \\")
        print(f"    --env_name InvertedPendulum-v4 \\")
        print(f"    --ep_len 1000 \\")
        print(f"    --discount 0.9 \\")
        print(f"    -n 100 \\")
        print(f"    -l 2 \\")
        print(f"    -s 64 \\")
        print(f"    -b {optimal_batch} \\")
        print(f"    -lr {optimal_lr:.4f} \\")
        print(f"    -rtg \\")
        print(f"    --exp_name q2_b{optimal_batch}_r{optimal_lr:.4f}_optimal")
    else:
        print(f"\n⚠️  Validation failed! The combination b*={optimal_batch}, r*={optimal_lr:.4f} doesn't work together.")
        print("This suggests the search found individual working values but they're incompatible.")
        print("You may need to run a more comprehensive search or adjust the search strategy.")

if __name__ == "__main__":
    main()