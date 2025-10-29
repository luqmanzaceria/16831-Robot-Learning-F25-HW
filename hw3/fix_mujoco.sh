#!/bin/bash

set -e

echo "==================================================================="
echo "MuJoCo-py Fix for Apple Silicon (M1/M2/M3)"
echo "==================================================================="

echo ""
echo "Step 1: Installing libomp (OpenMP support for clang)..."
brew install libomp

echo ""
echo "Step 2: Cleaning old mujoco-py build cache..."
rm -rf ~/.mujoco/mujoco-py-*
rm -rf ~/Library/Caches/mujoco_py
find ~/miniconda3/envs/rob831/lib/python3.10/site-packages/mujoco_py -name "*.so" -delete 2>/dev/null || true
find ~/miniconda3/envs/rob831/lib/python3.10/site-packages/mujoco_py -name "*.pyc" -delete 2>/dev/null || true
rm -rf ~/miniconda3/envs/rob831/lib/python3.10/site-packages/mujoco_py/generated 2>/dev/null || true

echo ""
echo "Step 3: Setting up environment variables..."
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$DYLD_LIBRARY_PATH

# Compiler settings for OpenMP support
export CC=clang
export CXX=clang++
export CPPFLAGS="-I/opt/homebrew/include"
export LDFLAGS="-L/opt/homebrew/lib -L/opt/homebrew/opt/libomp/lib"
export CFLAGS="-I/opt/homebrew/opt/libomp/include"
export CXXFLAGS="-I/opt/homebrew/opt/libomp/include"

# Disable OpenMP in mujoco-py (clang doesn't support -fopenmp flag)
export MUJOCO_PY_FORCE_CPU=1

echo ""
echo "Step 4: Reinstalling mujoco-py..."
pip uninstall -y mujoco-py
pip install mujoco-py --no-cache-dir --force-reinstall

echo ""
echo "Step 5: Testing mujoco-py import..."
python -c "import mujoco_py; print('✅ MuJoCo-py imported successfully!')"

echo ""
echo "==================================================================="
echo "✅ MuJoCo setup complete!"
echo "==================================================================="
echo ""
echo "Now reload your shell and try running your script:"
echo "  source ~/.zshrc"
echo "  python rob831/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v4 ..."
