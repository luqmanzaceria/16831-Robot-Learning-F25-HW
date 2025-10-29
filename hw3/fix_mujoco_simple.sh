#!/bin/bash

set -e

echo "==================================================================="
echo "Simple MuJoCo-py Fix for Apple Silicon"
echo "==================================================================="

echo ""
echo "Step 1: Cleaning old mujoco-py build..."
rm -rf ~/miniconda3/envs/rob831/lib/python3.10/site-packages/mujoco_py/generated

echo ""
echo "Step 2: Patching mujoco-py to disable OpenMP..."

# Find mujoco-py builder.py location
BUILDER_FILE=~/miniconda3/envs/rob831/lib/python3.10/site-packages/mujoco_py/builder.py

if [ ! -f "$BUILDER_FILE" ]; then
    echo "Error: mujoco-py not found. Install it first with: pip install mujoco-py"
    exit 1
fi

# Backup original
cp "$BUILDER_FILE" "${BUILDER_FILE}.backup"

# Remove -fopenmp flag from builder.py
sed -i.tmp "s/'-fopenmp'/# '-fopenmp' # Removed for macOS compatibility/g" "$BUILDER_FILE"
rm "${BUILDER_FILE}.tmp"

echo "✅ Patched mujoco-py builder.py"

echo ""
echo "Step 3: Setting environment variables..."
source ~/.zshrc

echo ""
echo "Step 4: Testing import (this will trigger compilation)..."
python -c "import mujoco_py; print('✅ MuJoCo-py works!')" || {
    echo "❌ Import failed. Trying alternative fix..."

    # Alternative: use gcc instead of clang
    export CC=gcc-13
    export CXX=g++-13

    # Clean and retry
    rm -rf ~/miniconda3/envs/rob831/lib/python3.10/site-packages/mujoco_py/generated
    python -c "import mujoco_py; print('✅ MuJoCo-py works with gcc!')"
}

echo ""
echo "==================================================================="
echo "✅ MuJoCo setup complete!"
echo "==================================================================="
