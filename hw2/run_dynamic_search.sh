#!/bin/bash
# Wrapper script to run the smart dynamic search

echo "Starting Smart Dynamic Search for InvertedPendulum-v4"
echo "===================================================="
echo "Search ranges:"
echo "  - Batch size: 1 to 1000"
echo "  - Learning rate: 0.001 to 1.0 (very large)"
echo "  - Goal: Find smallest b* and largest r* that work together"
echo ""

# Check if required packages are available
python -c "import tensorboard" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install tensorboard
fi

# Run the smart dynamic search
python smart_dynamic_search.py

echo ""
echo "Search complete! Check the results above."
echo "You can also run 'python analyze_inverted_pendulum.py' for detailed analysis."
