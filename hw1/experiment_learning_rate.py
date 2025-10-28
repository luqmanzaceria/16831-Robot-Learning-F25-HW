#!/usr/bin/env python3

import os
import subprocess
import numpy as np
import json
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, will skip plotting")

def run_experiment(learning_rate, env_name, expert_data, expert_policy_file, num_runs=5):
    """Run BC experiment with specified learning rate for multiple runs"""
    returns = []

    for run in range(num_runs):
        # Create experiment name
        exp_name = f"lr_experiment_lr{learning_rate}_run{run}"

        # Run the training command
        cmd = [
            "python", "rob831/scripts/run_hw1.py",
            "--expert_policy_file", expert_policy_file,
            "--expert_data", expert_data,
            "--env_name", env_name,
            "--exp_name", exp_name,
            "--n_iter", "1",
            "--learning_rate", str(learning_rate),
            "--num_agent_train_steps_per_iter", "1000",
            "--batch_size", "1000",
            "--train_batch_size", "100"
        ]

        print(f"Running experiment {run+1}/{num_runs} with learning rate {learning_rate}")
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/ubuntu/robotlearning/CMU-16-831-Robot-Learning/hw1")
            if result.returncode != 0:
                print(f"Error in run {run}: {result.stderr}")
                continue

            # Find the log directory
            data_dir = Path("/home/ubuntu/robotlearning/CMU-16-831-Robot-Learning/hw1/data")
            log_dirs = [d for d in data_dir.glob(f"q1_{exp_name}_*") if d.is_dir()]

            if not log_dirs:
                print(f"No log directory found for run {run}")
                continue

            # Get the most recent log directory
            log_dir = max(log_dirs, key=os.path.getctime)

            # Read the performance from the log file
            log_files = list(log_dir.glob("events.out.tfevents*"))

            if not log_files:
                print(f"No tensorboard log found for run {run}")
                continue

            # Parse tensorboard logs to get eval return
            try:
                import tensorflow as tf
                eval_return = None
                for event_file in log_files:
                    for event in tf.compat.v1.train.summary_iterator(str(event_file)):
                        for value in event.summary.value:
                            if value.tag == 'Eval_AverageReturn':
                                eval_return = value.simple_value

                if eval_return is not None:
                    returns.append(eval_return)
                    print(f"Run {run+1}: Return = {eval_return}")
                else:
                    print(f"Could not find eval return for run {run}")

            except ImportError:
                print("TensorFlow not available, trying alternative method...")
                # Alternative: try to read JSON logs if they exist
                json_files = list(log_dir.glob("*.json"))
                if json_files:
                    try:
                        with open(json_files[0], 'r') as f:
                            data = json.load(f)
                            if 'Eval_AverageReturn' in data:
                                eval_return = data['Eval_AverageReturn']
                                returns.append(eval_return)
                                print(f"Run {run+1}: Return = {eval_return}")
                    except:
                        pass

        except Exception as e:
            print(f"Exception in run {run}: {e}")
            continue

    return returns

def plot_results(learning_rates, results, env_name):
    """Plot the results with error bars"""
    means = []
    stds = []

    for lr_results in results:
        if len(lr_results) > 0:
            means.append(np.mean(lr_results))
            stds.append(np.std(lr_results))
        else:
            means.append(0)
            stds.append(0)

    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot generation")
        return means, stds

    plt.figure(figsize=(10, 6))
    plt.errorbar(learning_rates, means, yerr=stds, marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Return')
    plt.title(f'Behavioral Cloning Performance vs Learning Rate ({env_name})')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)

    # Add text annotations for each point
    for lr, mean, std in zip(learning_rates, means, stds):
        plt.annotate(f'{mean:.1f}±{std:.1f}',
                    (lr, mean),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')

    plt.tight_layout()
    plt.savefig('learning_rate_experiment.png', dpi=300, bbox_inches='tight')
    plt.savefig('learning_rate_experiment.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return means, stds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v2',
                       help='Environment name (e.g., Hopper-v2)')
    parser.add_argument('--expert_data', type=str, required=True,
                       help='Path to expert data')
    parser.add_argument('--expert_policy_file', type=str, required=True,
                       help='Path to expert policy file')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of runs per learning rate')

    args = parser.parse_args()

    # Learning rates to test (logarithmic scale)
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

    print(f"Starting learning rate experiment for {args.env_name}")
    print(f"Learning rates to test: {learning_rates}")
    print(f"Number of runs per learning rate: {args.num_runs}")

    all_results = []

    for lr in learning_rates:
        print(f"\n{'='*50}")
        print(f"Testing learning rate: {lr}")
        print(f"{'='*50}")

        results = run_experiment(
            learning_rate=lr,
            env_name=args.env_name,
            expert_data=args.expert_data,
            expert_policy_file=args.expert_policy_file,
            num_runs=args.num_runs
        )

        all_results.append(results)
        print(f"Results for lr={lr}: {results}")

        if len(results) > 0:
            mean_return = np.mean(results)
            std_return = np.std(results)
            print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")
        else:
            print("No successful runs for this learning rate")

    # Plot results
    print(f"\n{'='*50}")
    print("Plotting results...")
    print(f"{'='*50}")

    means, stds = plot_results(learning_rates, all_results, args.env_name)

    # Save results to JSON
    results_data = {
        'learning_rates': learning_rates,
        'all_results': all_results,
        'means': means,
        'stds': stds,
        'env_name': args.env_name
    }

    with open('learning_rate_experiment_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    print("Results saved to learning_rate_experiment_results.json")
    print("Plot saved as learning_rate_experiment.png and learning_rate_experiment.pdf")

if __name__ == "__main__":
    main()