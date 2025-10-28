#!/bin/bash

# HW2 Section 5 Analysis 1: Value Estimator Comparison
# This script runs experiments to compare trajectory-centric vs reward-to-go estimators

echo "HW2 Section 5 Analysis 1: Value Estimator Comparison"
echo "=================================================="

# Change to the hw2 directory
cd "$(dirname "$0")"

# Install required packages
echo "Installing required packages..."
pip install tensorboard matplotlib pandas numpy

# Run Analysis 1 experiments
echo ""
echo "=================================================="
echo "RUNNING ANALYSIS 1 EXPERIMENTS"
echo "=================================================="

echo ""
echo "[1/4] Running q1_sb_rtg_na (small batch, rtg, no adv std)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 1500 -rtg --exp_name q1_sb_rtg_na

echo ""
echo "[2/4] Running q1_sb_na (small batch, no rtg, no adv std)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 1500 --exp_name q1_sb_na

echo ""
echo "[3/4] Running q1_lb_rtg_na (large batch, rtg, no adv std)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 6000 -rtg --exp_name q1_lb_rtg_na

echo ""
echo "[4/4] Running q1_lb_na (large batch, no rtg, no adv std)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 6000 --exp_name q1_lb_na

echo ""
echo "=================================================="
echo "CREATING ANALYSIS 1 PLOTS"
echo "=================================================="

# Create plots for Analysis 1
python extract_analysis1_data.py --data_dir data --output_dir plots_a1

echo ""
echo "=================================================="
echo "ANALYSIS 1 COMPLETED!"
echo "=================================================="
echo "Learning curve plots have been created in the 'plots_a1' directory:"
echo "- small_batch_learning_curves.pdf/png"
echo "- large_batch_learning_curves.pdf/png"
echo ""
echo "Raw data saved in JSON format:"
echo "- small_batch_data.json"
echo "- large_batch_data.json"
