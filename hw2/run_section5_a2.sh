#!/bin/bash

# HW2 Section 5 Analysis 2: Advantage Standardization Comparison
# This script runs experiments to compare with and without advantage standardization

echo "HW2 Section 5 Analysis 2: Advantage Standardization Comparison"
echo "============================================================="

# Change to the hw2 directory
cd "$(dirname "$0")"

# Install required packages
echo "Installing required packages..."
pip install tensorboard matplotlib pandas numpy

# Run Analysis 2 experiments
echo ""
echo "============================================================="
echo "RUNNING ANALYSIS 2 EXPERIMENTS"
echo "============================================================="

echo ""
echo "[1/8] Running q1_sb_rtg_na (small batch, rtg, no adv std)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 1500 -rtg --exp_name q1_sb_rtg_na

echo ""
echo "[2/8] Running q1_sb_rtg_dsa (small batch, rtg, with adv std)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 1500 -rtg -dsa --exp_name q1_sb_rtg_dsa

echo ""
echo "[3/8] Running q1_sb_na (small batch, no rtg, no adv std)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 1500 --exp_name q1_sb_na

echo ""
echo "[4/8] Running q1_sb_dsa (small batch, no rtg, with adv std)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 1500 -dsa --exp_name q1_sb_dsa

echo ""
echo "[5/8] Running q1_lb_rtg_na (large batch, rtg, no adv std)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 6000 -rtg --exp_name q1_lb_rtg_na

echo ""
echo "[6/8] Running q1_lb_rtg_dsa (large batch, rtg, with adv std)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 6000 -rtg -dsa --exp_name q1_lb_rtg_dsa

echo ""
echo "[7/8] Running q1_lb_na (large batch, no rtg, no adv std)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 6000 --exp_name q1_lb_na

echo ""
echo "[8/8] Running q1_lb_dsa (large batch, no rtg, with adv std)"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 6000 -dsa --exp_name q1_lb_dsa

echo ""
echo "============================================================="
echo "CREATING ANALYSIS 2 PLOTS"
echo "============================================================="

# Create plots for Analysis 2
python extract_analysis2_data.py --data_dir data --output_dir plots_a2

echo ""
echo "============================================================="
echo "ANALYSIS 2 COMPLETED!"
echo "============================================================="
echo "Learning curve plots have been created in the 'plots_a2' directory:"
echo "- small_batch_learning_curves.pdf/png"
echo "- large_batch_learning_curves.pdf/png"
echo ""
echo "Raw data saved in JSON format:"
echo "- small_batch_data.json"
echo "- large_batch_data.json"
