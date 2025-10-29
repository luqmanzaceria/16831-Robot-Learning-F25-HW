#!/bin/bash

# Installation script for Gymnasium migration
# This script will uninstall gym and install gymnasium with MuJoCo support

echo "Uninstalling old gym and mujoco-py packages..."
pip uninstall -y gym mujoco-py free-mujoco-py gym-notebook-wrapper

echo "Installing gymnasium and dependencies..."
pip install -r requirements.txt

echo "Installation complete!"
echo "You can now run your scripts with Gymnasium."
