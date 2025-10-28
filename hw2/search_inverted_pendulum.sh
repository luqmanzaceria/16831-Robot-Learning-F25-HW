#!/bin/bash
# Grid search for optimal batch size and learning rate for InvertedPendulum-v4
# Goal: Find smallest batch size b* and largest learning rate r* that reaches 1000 in <100 iterations

# Strategy: Start with reasonable ranges and refine
# Batch sizes to try (smallest first)
BATCH_SIZES=(100 200 500 1000 2000 5000)

# Learning rates to try (largest first)
LEARNING_RATES=(0.02 0.01 0.005 0.001 0.0005)

echo "Starting grid search for InvertedPendulum-v4"
echo "=============================================="
echo ""

# Try different combinations
for lr in "${LEARNING_RATES[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
        exp_name="q2_b${batch}_r${lr}"
        echo "Running: batch=${batch}, lr=${lr}"

        python rob831/scripts/run_hw2.py \
            --env_name InvertedPendulum-v4 \
            --ep_len 1000 \
            --discount 0.9 \
            -n 100 \
            -l 2 \
            -s 64 \
            -b ${batch} \
            -lr ${lr} \
            -rtg \
            --exp_name ${exp_name}

        echo "Completed: ${exp_name}"
        echo ""
    done
done

echo "Grid search complete!"
echo "Run the analysis script to find the best configuration."
