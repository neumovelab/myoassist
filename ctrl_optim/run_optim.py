#!/usr/bin/env python3
"""
Python script to execute training configurations the same way as run_training.bat and run_training.sh
"""

import os
import sys
import subprocess
import glob
import platform
import shutil
from pathlib import Path

env = os.environ.copy()
env['PYTHON_EXECUTABLE'] = sys.executable


def get_script_directory():
    """Get the directory where this script is located"""
    return Path(__file__).parent.absolute()


def list_available_configs():
    """List all available configuration files"""
    script_dir = get_script_directory()
    config_dir = script_dir / "optim" / "training_configs"
    
    if not config_dir.exists():
        print(f"  No training_configs directory found at: {config_dir}")
        return []
    
    # Get all .bat and .sh files
    config_files = []
    for ext in ["*.bat", "*.sh"]:
        config_files.extend(glob.glob(str(config_dir / ext)))
    
    if not config_files:
        print("  No configuration files found")
        return []
    
    # Extract just the names without extension
    config_names = []
    for config_file in config_files:
        name = Path(config_file).stem
        if name not in config_names:
            config_names.append(name)
    
    return sorted(config_names)


def setup_environment():
    """Set up the environment variables and working directory"""
    script_dir = get_script_directory()
    
    # Store original directory
    original_dir = os.getcwd()
    
    # Change to parent directory to set ROOT_DIR
    os.chdir(script_dir.parent)
    root_dir = os.getcwd()
    
    # Set PYTHONPATH
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if current_pythonpath:
        new_pythonpath = f"{root_dir}{os.pathsep}{current_pythonpath}"
    else:
        new_pythonpath = str(root_dir)
    
    os.environ['PYTHONPATH'] = new_pythonpath
    os.environ['ROOT_DIR'] = str(root_dir)
    
    # Change to script directory
    os.chdir(script_dir)
    
    return original_dir


def execute_config(config_name, original_dir):
    """Execute the specified configuration file"""
    script_dir = get_script_directory()
    config_dir = script_dir / "optim" / "training_configs"
    
    # Select script type based on OS
    is_windows = platform.system() == "Windows"
    
    # Determine which script to use
    if is_windows:
        primary_config = config_dir / f"{config_name}.bat"
        fallback_config = config_dir / f"{config_name}.sh"
    else:
        primary_config = config_dir / f"{config_name}.sh"
        fallback_config = config_dir / f"{config_name}.bat"
    
    config_file = None
    if primary_config.exists():
        config_file = primary_config
    elif fallback_config.exists():
        print(f"Warning: Preferred script type for your OS ({primary_config.suffix}) not found.")
        print(f"Attempting to use {fallback_config.suffix} script instead.")
        config_file = fallback_config
    
    if not config_file:
        print(f"Error: Configuration '{config_name}' not found")
        print()
        print("Available configurations:")
        for config in list_available_configs():
            print(f"  {config}")
        return False
    
    # Change to optim directory
    optim_dir = script_dir / "optim"
    os.chdir(optim_dir)
    
    try:
        # Execute the configuration file
        if platform.system() == "Windows":
            if config_file.suffix == ".bat":
                result = subprocess.run(["cmd", "/c", str(config_file)], 
                                      shell=True, check=True)
            else:
                # For .sh files on Windows, we need bash
                if not shutil.which('bash'):
                    print("Error: bash is required to run .sh files on Windows")
                    return False
                result = subprocess.run(["bash", str(config_file)], 
                                      shell=True, check=True)
        else:
            # On Unix-like systems
            if config_file.suffix == ".sh":
                # Make executable and run
                os.chmod(config_file, 0o755)
                result = subprocess.run([str(config_file)], 
                                      shell=True, check=True)
            else:
                print("Error: .bat files are not supported on Unix-like systems")
                print("Please use the equivalent .sh file instead")
                return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing configuration: {e}")
        return False
    finally:
        # Return to original directory
        os.chdir(original_dir)


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python run_training.py <config_name>")
        print("Available configurations:")
        
        configs = list_available_configs()
        for config in configs:
            print(f"  {config}")
        
        sys.exit(1)
    
    config_name = sys.argv[1]
    
    # Set up environment
    original_dir = setup_environment()
    
    try:
        # Execute the configuration
        success = execute_config(config_name, original_dir)
        if not success:
            sys.exit(1)
    finally:
        # Ensure we return to original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main() 