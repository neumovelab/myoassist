#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: ./run_training.sh <config_name>"
    echo "Available configurations:"
    
    # Get the directory of this script
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    config_dir="${script_dir}/training_configs"
    
    # Check if training_configs directory exists
    if [ ! -d "$config_dir" ]; then
        echo "  No training_configs directory found at: $config_dir"
        exit 1
    fi
    
    # List all .sh files in training_configs directory
    if [ -z "$(ls -A "$config_dir"/*.sh 2>/dev/null)" ]; then
        echo "  No .sh configuration files found"
    else
        for config_file in "$config_dir"/*.sh; do
            if [ -f "$config_file" ]; then
                filename=$(basename "$config_file" .sh)
                echo "  $filename"
            fi
        done
    fi
    
    exit 1
fi

config_name=$1
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_file="${script_dir}/training_configs/${config_name}.sh"

if [ ! -f "$config_file" ]; then
    echo "Error: Configuration '$config_name' not found"
    echo
    echo "Available configurations:"
    
    # Get the directory of this script
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    config_dir="${script_dir}/training_configs"
    
    # List all .sh files in training_configs directory
    if [ -z "$(ls -A "$config_dir"/*.sh 2>/dev/null)" ]; then
        echo "  No .sh configuration files found"
    else
        for config_file in "$config_dir"/*.sh; do
            if [ -f "$config_file" ]; then
                filename=$(basename "$config_file" .sh)
                echo "  $filename"
            fi
        done
    fi
    
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