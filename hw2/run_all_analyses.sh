#!/bin/bash

# HW2 Complete Analysis Runner
# This script runs both Analysis 1 and Analysis 2, then generates all plots

echo "HW2 Complete Analysis Runner"
echo "============================"

# Change to the hw2 directory
cd "$(dirname "$0")"

# Install required packages
echo "Installing required packages..."
pip install tensorboard matplotlib pandas numpy

echo ""
echo "============================"
echo "RUNNING ANALYSIS 1"
echo "============================"
./run_section5_a1.sh

echo ""
echo "============================"
echo "RUNNING ANALYSIS 2"
echo "============================"
./run_section5_a2.sh

echo ""
echo "============================"
echo "GENERATING ANALYSIS PLOTS"
echo "============================"

# Generate analysis plots for both
echo "Generating Analysis 1 plots..."
python analyze_results_v2.py --data_dir plots_a1 --analysis a1

echo ""
echo "Generating Analysis 2 plots..."
python analyze_results_v2.py --data_dir plots_a2 --analysis a2

echo ""
echo "============================"
echo "ALL ANALYSES COMPLETE!"
echo "============================"
echo "Results are available in:"
echo "- plots_a1/ (Analysis 1: Value Estimator Comparison)"
echo "- plots_a2/ (Analysis 2: Advantage Standardization Comparison)"
echo ""
echo "Each directory contains:"
echo "- Learning curve plots (PDF and PNG)"
echo "- Performance comparison plots"
echo "- Raw data in JSON format"
