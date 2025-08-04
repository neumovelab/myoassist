"""
Entry point for running the processing module directly.
"""

import os
import sys

# Add the parent directory to the path for processing import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ctrl_optim.results.processing.processing import main

if __name__ == '__main__':
    main() 