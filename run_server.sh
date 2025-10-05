#!/bin/bash

# Run the Musequill server with automatic restart on failure

echo "Starting Musequill server with memory-safe configuration..."
echo "Using model profile: ${MODEL_PROFILE:-small}"
echo ""

# Set default to small models for memory safety
export MODEL_PROFILE=${MODEL_PROFILE:-small}

while true; do
    echo "[$(date)] Starting server..."
    python main.py
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Server exited normally."
        break
    else
        echo "[$(date)] Server crashed with exit code $EXIT_CODE"
        echo "[$(date)] Restarting in 5 seconds..."
        sleep 5
    fi
done