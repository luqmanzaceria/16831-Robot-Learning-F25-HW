#!/bin/bash
# Quick search strategy: Start with known good configs and refine
# Based on typical PG performance on InvertedPendulum

# Change to hw2 directory
cd "$(dirname "$0")"

# Use system Python with gymnasium (InvertedPendulum-v5)
PYTHON=python

echo "Quick search for InvertedPendulum-v5 optimal hyperparameters"
echo "=============================================================="
echo ""
echo "Using gymnasium with InvertedPendulum-v5 (newer mujoco bindings)"
echo ""

# Phase 1: Find a working baseline with moderate settings
echo "Phase 1: Testing baseline configurations..."
echo ""

# Test moderate batch with moderate LR
echo "Running: b=1000, lr=0.01"
$PYTHON -m rob831.scripts.run_hw2 \
    --env_name InvertedPendulum-v5 \
    --ep_len 1000 \
    --discount 0.9 \
    -n 100 \
    -l 2 \
    -s 64 \
    -b 1000 \
    -lr 0.01 \
    -rtg \
    --exp_name q2_b1000_r0.01
echo "Completed: b=1000, lr=0.01"

# Phase 2: Try smaller batches with same LR
echo ""
echo "Phase 2: Testing smaller batch sizes..."
echo ""

for batch in 500 200 100 50; do
    echo "Running: b=${batch}, lr=0.01"
    $PYTHON -m rob831.scripts.run_hw2 \
        --env_name InvertedPendulum-v5 \
        --ep_len 1000 \
        --discount 0.9 \
        -n 100 \
        -l 2 \
        -s 64 \
        -b ${batch} \
        -lr 0.01 \
        -rtg \
        --exp_name q2_b${batch}_r0.01
    echo "Completed: b=${batch}, lr=0.01"
    echo ""
done

# Phase 3: Try larger learning rates with promising batch sizes
echo ""
echo "Phase 3: Testing larger learning rates..."
echo ""

for lr in 0.015 0.02 0.025 0.03; do
    echo "Running: b=100, lr=${lr}"
    $PYTHON -m rob831.scripts.run_hw2 \
        --env_name InvertedPendulum-v5 \
        --ep_len 1000 \
        --discount 0.9 \
        -n 100 \
        -l 2 \
        -s 64 \
        -b 100 \
        -lr ${lr} \
        -rtg \
        --exp_name q2_b100_r${lr}
    echo "Completed: b=100, lr=${lr}"
    echo ""
done

echo ""
echo "=============================================================="
echo "Quick search complete!"
echo "=============================================================="
echo ""
echo "Now run: python analyze_inverted_pendulum.py"
echo ""
