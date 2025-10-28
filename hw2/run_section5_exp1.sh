#!/bin/bash

# HW2 Experiment Runner and Plot Generator
# This script runs all 6 experiments and creates learning curve plots

echo "HW2 Complete Experiment Runner and Plot Generator"
echo "=============================================="

# Change to the hw2 directory
cd "$(dirname "$0")"

# Install required packages
echo "Installing required packages..."
pip install tensorboard matplotlib pandas numpy

# Run all experiments
echo ""
echo "=============================================="
echo "RUNNING ALL EXPERIMENTS"
echo "=============================================="

echo ""
echo "[1/6] Running q1_sb_no_rtg_dsa (small batch, no rtg, dsa)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 1500 -dsa --exp_name q1_sb_no_rtg_dsa

echo ""
echo "[2/6] Running q1_sb_rtg_dsa (small batch, rtg, dsa)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 1500 -rtg -dsa --exp_name q1_sb_rtg_dsa

echo ""
echo "[3/6] Running q1_sb_rtg_na (small batch, rtg, no dsa)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 1500 -rtg --exp_name q1_sb_rtg_na

echo ""
echo "[4/6] Running q1_lb_no_rtg_dsa (large batch, no rtg, dsa)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 6000 -dsa --exp_name q1_lb_no_rtg_dsa

echo ""
echo "[5/6] Running q1_lb_rtg_dsa (large batch, rtg, dsa)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 6000 -rtg -dsa --exp_name q1_lb_rtg_dsa

echo ""
echo "[6/6] Running q1_lb_rtg_na (large batch, rtg, no dsa)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 6000 -rtg --exp_name q1_lb_rtg_na

echo ""
echo "=============================================="
echo "CREATING LEARNING CURVE PLOTS"
echo "=============================================="

# Create plots using the Python script
python extract_tensorboard_data.py

echo ""
echo "=============================================="
echo "COMPLETED!"
echo "=============================================="
echo "Learning curve plots have been created in the 'plots' directory:"
echo "- small_batch_learning_curves.pdf/png"
echo "- large_batch_learning_curves.pdf/png"
echo ""
echo "Raw data saved in JSON format:"
echo "- small_batch_data.json"
echo "- large_batch_data.json"
