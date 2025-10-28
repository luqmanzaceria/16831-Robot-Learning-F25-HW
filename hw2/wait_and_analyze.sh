#!/bin/bash
# Wait for experiments to complete and then analyze

echo "Waiting for InvertedPendulum experiments to complete..."
echo "This script will check every 2 minutes."
echo ""
echo "You can also run 'python analyze_inverted_pendulum.py' manually when ready."
echo ""

target_count=30  # We expect 30 experiments (6 batch sizes × 5 learning rates)

while true; do
    # Count completed experiments (those with event files)
    completed=$(find data/q2_* -name "events.out.tfevents.*" 2>/dev/null | wc -l | tr -d ' ')

    echo "[$(date +%H:%M:%S)] Progress: $completed / $target_count experiments have data"

    if [ "$completed" -ge "$target_count" ]; then
        echo ""
        echo "All experiments appear to have started! Waiting 30 more seconds for completion..."
        sleep 30
        echo ""
        echo "Running analysis..."
        python analyze_inverted_pendulum.py
        break
    fi

    sleep 120  # Check every 2 minutes
done
