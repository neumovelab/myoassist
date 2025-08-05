"""
Path resolution utilities for the ctrl_optim package.

This module provides utilities for resolving file paths consistently
across different execution contexts (evaluation, optimization, etc.).
"""

import os
from typing import Optional


def get_project_root() -> str:
    """
    Get the absolute path to the project root directory.
    
    Returns:
        str: Absolute path to the project root
    """
    # Start from this file's location and navigate to project root
    current_file = os.path.abspath(__file__)
    # Navigate from ctrl_optim/optim/optim_utils/path_utils.py to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '..', '..', '..'))
    return project_root


def resolve_model_path(model: str, mode: str, model_path: Optional[str] = None) -> str:
    """
    Resolve the model path consistently regardless of execution context.
    
    Args:
        model (str): Model type (baseline, dephy, hmedi, humotech, tutorial, custom)
        mode (str): Movement mode ('2D' or '3D')
        model_path (str, optional): Custom model path for custom models
        
    Returns:
        str: Absolute path to the model file
        
    Raises:
        ValueError: If model type is invalid
        FileNotFoundError: If model file doesn't exist
    """
    # If a custom model path is provided, use it directly
    if model == "custom" and model_path:
        if os.path.isabs(model_path):
            return model_path
        else:
            # Resolve relative to project root
            project_root = get_project_root()
            return os.path.join(project_root, model_path)
    
    # Get project root
    project_root = get_project_root()
    
    # Construct model path based on mode and model type
    if mode == '2D':
        model_dir = '22muscle_2D'
        model_prefix = 'myoLeg22_2D'
    else:  # 3D
        model_dir = '26muscle_3D'
        model_prefix = 'myoLeg26'
    
    # Map model types to file names
    model_mapping = {
        "baseline": "BASELINE",
        "dephy": "DEPHY", 
        "hmedi": "HMEDI",
        "humotech": "HUMOTECH",
        "tutorial": "TUTORIAL"
    }
    
    if model not in model_mapping:
        raise ValueError(f"Invalid model type '{model}'. Valid types: {list(model_mapping.keys())}")
    
    model_suffix = model_mapping[model]
    model_filename = f"{model_prefix}_{model_suffix}.xml"
    
    # Construct full path
    full_model_path = os.path.join(project_root, 'models', model_dir, model_filename)
    
    # Verify the file exists
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Model file not found: {full_model_path}")
    
    return full_model_path


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
    ref_data_path = os.path.join(project_root, 'ctrl_optim', 'optim', 'ref_data', filename)
    
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


def get_available_models() -> dict:
    """
    Get a dictionary of available models organized by mode.
    
    Returns:
        dict: Dictionary with '2D' and '3D' keys, each containing list of available models
    """
    return {
        '2D': ['baseline', 'dephy', 'hmedi', 'humotech', 'tutorial'],
        '3D': ['baseline', 'dephy', 'hmedi', 'humotech', 'tutorial']
    }


def validate_model_config(model: str, mode: str) -> bool:
    """
    Validate that a model configuration is valid.
    
    Args:
        model (str): Model type
        mode (str): Movement mode ('2D' or '3D')
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    available_models = get_available_models()
    
    if mode not in available_models:
        raise ValueError(f"Invalid mode '{mode}'. Valid modes: {list(available_models.keys())}")
    
    if model not in available_models[mode]:
        raise ValueError(f"Invalid model '{model}' for mode '{mode}'. Valid models: {available_models[mode]}")
    
    return True