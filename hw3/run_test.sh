#!/bin/zsh

# Source the shell config to get proper PATH
source ~/.zshrc

# Set MuJoCo environment variables
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export DYLD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$DYLD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH

# Run the test
cd /Users/luqmanzaceria/mscv/fall25/CMU-16-831-Robot-Learning/16831-Robot-Learning-F25-HW/hw3

python rob831/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v4 \
  --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 \
  --exp_name q3_10_10 -ntu 10 -ngsptu 10 --no_gpu
