#!/bin/bash
# Empty the log file before starting
> backend.log

echo "Starting FastAPI backend... All output is being logged to backend.log"
echo "Press Ctrl+C to stop."

# Run uvicorn and pipe both stdout and stderr to tee, which prints to terminal and writes to backend.log
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 2>&1 | tee backend.log
