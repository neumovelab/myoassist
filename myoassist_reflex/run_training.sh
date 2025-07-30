#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: ./run_training.sh <config_name>"
    echo "Available configurations:"
    echo "  basic_11muscle    - Basic 11-muscle model optimization"
    echo "  exo_legacy       - Exoskeleton with legacy 4-point spline"
    echo "  exo_npoint       - Exoskeleton with n-point spline"
    echo "  debug           - Quick debug run"
    exit 1
fi

config_name=$1
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_file="${script_dir}/training_configs/${config_name}.sh"

if [ ! -f "$config_file" ]; then
    echo "Error: Configuration '$config_name' not found"
    exit 1
fi

# Store original directory
ORIGINAL_DIR=$(pwd)

# Change to script directory
cd "$script_dir"

# Set up Python path
ROOT_DIR="$(cd .. && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

# Make config file executable and run it
chmod +x "$config_file"
"$config_file"

# Return to original directory
cd "$ORIGINAL_DIR"