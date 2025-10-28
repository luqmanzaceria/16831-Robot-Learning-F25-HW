#!/bin/bash

# Batch sizes and learning rates to search over
batch_sizes=(15000 35000 55000)
learning_rates=(0.005 0.01 0.02)

# Loop over all combinations
for b in "${batch_sizes[@]}"; do
  for r in "${learning_rates[@]}"; do
    echo "Running experiment with batch_size=$b and learning_rate=$r"

    python rob831/scripts/run_hw2.py \
      --env_name HalfCheetah-v4 \
      --ep_len 150 \
      --discount 0.95 \
      -n 100 \
      -l 2 \
      -s 32 \
      -b $b \
      -lr $r \
      -rtg \
      --nn_baseline \
      --exp_name q4_search_b${b}_lr${r}_rtg_nnbaseline
  done
done
