#!/bin/bash
# Quick check of experiment progress

echo "Checking InvertedPendulum experiment progress..."
echo "================================================"
echo ""

# Count completed experiments
completed=$(ls -d data/q2_* 2>/dev/null | wc -l | tr -d ' ')
echo "Experiments found: $completed"

echo ""
echo "Most recent experiments:"
ls -lt data/q2_* 2>/dev/null | head -5

echo ""
echo "Experiment sizes (larger = more complete):"
du -sh data/q2_* 2>/dev/null | tail -5
