"""
Path resolution utilities for the ctrl_optim package.

This module provides utilities for resolving file paths consistently
across different execution contexts (evaluation, optimization, etc.).
"""

import os


def get_project_root() -> str:
    """
    Get the absolute path to the project root directory.

    Returns:
        str: Absolute path to the project root
    """
    # Start from this file's location and navigate to project root
    current_file = os.path.abspath(__file__)
    # Navigate from ctrl_optim/optim/optim_utils/resolve_path.py to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), "..", "..", ".."))
    return project_root


def resolve_reference_data_path(filename: str) -> str:
    """
    Resolve path to reference data files.

    Args:
        filename (str): Name of the reference data file

    Returns:
        str: Absolute path to the reference data file

    Raises:
        FileNotFoundError: If reference data file doesn't exist
    """
    project_root = get_project_root()
    ref_data_path = os.path.join(project_root, "ctrl_optim", "optim", "ref_data", filename)

    if not os.path.exists(ref_data_path):
        raise FileNotFoundError(f"Reference data file not found: {ref_data_path}")

    return ref_data_path


def resolve_results_path(relative_path: str) -> str:
    """
    Resolve path to results directory.

    Args:
        relative_path (str): Relative path from project root to results

    Returns:
        str: Absolute path to the results directory
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)
