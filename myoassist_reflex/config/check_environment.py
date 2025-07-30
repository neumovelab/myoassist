#!/usr/bin/env python3
"""
Environment Checker for MyoAssist

This script checks if all required dependencies are installed and
helps install missing ones.
"""

import importlib.util
import sys
import subprocess
import os

def check_module(module_name):
    """Check if a module is installed."""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def install_module(module_name):
    """Install a module using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    # Required modules
    required_modules = {
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "cma": "cma",
        "pickle": "pickle-mixin",  # Standard library but check anyway
    }
    
    missing_modules = []
    
    print("Checking dependencies for MyoAssist...")
    
    # Check each module
    for module_name, pip_name in required_modules.items():
        sys.stdout.write(f"Checking for {module_name}... ")
        if check_module(module_name):
            print("OK")
        else:
            print("Missing")
            missing_modules.append((module_name, pip_name))
    
    # Install missing modules if any
    if missing_modules:
        print("\nThe following dependencies are missing:")
        for module_name, pip_name in missing_modules:
            print(f"  - {module_name}")
        
        install = input("\nDo you want to install them now? (y/n): ")
        if install.lower() == 'y':
            for module_name, pip_name in missing_modules:
                print(f"\nInstalling {module_name}...")
                if install_module(pip_name):
                    print(f"{module_name} installed successfully.")
                else:
                    print(f"Failed to install {module_name}. Please install it manually.")
        else:
            print("\nPlease install the missing dependencies manually.")
    else:
        print("\nAll dependencies are already installed!")
    
    # Check if the modular package is installable
    if os.path.exists("setup.py"):
        print("\nYou can install the MyoAssist package in development mode with:")
        print("pip install -e .")
    else:
        print("\nSetup file not found. Please make sure you're in the project root directory.")

if __name__ == "__main__":
    main() 