#!/usr/bin/env python3
"""
MyoAssist ReflexProcessing Pipeline Launcher
This script allows running the processing pipeline directly from the myoassist_reflex directory.
"""

import os
import sys

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run the processing pipeline
from processing.processing import main

if __name__ == "__main__":
    main() 