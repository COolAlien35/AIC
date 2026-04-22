#!/bin/bash
# Run hackathon tasks
cd "$(dirname "$0")"
exec python3 run_hackathon.py "$@"
