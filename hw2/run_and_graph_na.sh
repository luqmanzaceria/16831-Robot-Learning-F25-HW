#!/bin/bash

# Run small batch experiment without reward-to-go
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 1500 --exp_name q1_sb_na_na

# Run large batch experiment without reward-to-go
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 150 -b 6000 --exp_name q1_lb_na_na

echo "Experiments complete. Run your plotting script to visualize the results."
