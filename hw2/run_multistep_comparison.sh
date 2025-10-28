#!/bin/bash

# Experiment: Multi-step Policy Gradient Comparison
# Comparing single-step PG vs multi-step PG on HalfCheetah

# Environment settings
ENV_NAME="HalfCheetah-v4"
N_ITER=100
BATCH_SIZE=5000
EVAL_BATCH_SIZE=5000
LEARNING_RATE=0.02
DISCOUNT=0.95
N_LAYERS=2
SIZE=64
SEEDS=(1)

# Multi-step configurations to test
GRADIENT_STEPS=(1 3 5 10)

echo "Starting Multi-step PG Comparison Experiments"
echo "Environment: $ENV_NAME"
echo "Testing gradient steps: ${GRADIENT_STEPS[*]}"
echo ""

# Run experiments for each configuration
for STEPS in "${GRADIENT_STEPS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="multistep_comparison_steps${STEPS}_seed${SEED}"

        echo "Running: $EXP_NAME"

        python rob831/scripts/run_hw2.py \
            --env_name "$ENV_NAME" \
            --exp_name "$EXP_NAME" \
            --n_iter $N_ITER \
            --batch_size $BATCH_SIZE \
            --eval_batch_size $EVAL_BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --discount $DISCOUNT \
            --n_layers $N_LAYERS \
            --size $SIZE \
            --seed $SEED \
            --num_policy_gradient_steps_per_batch $STEPS \
            --reward_to_go \
            --nn_baseline \
            --scalar_log_freq 1

        echo "Completed: $EXP_NAME"
        echo ""
    done
done

echo "All experiments completed!"
echo ""
echo "To view results, run:"
echo "tensorboard --logdir rob831/data"
