"""
Optimization related modules.

This package contains modules for optimization parameters, 
bounds, and tracking.
"""

from .tracker import OptimizationTracker
from .bounds import (
    get_bounds,
    getBounds_22_26_mus,
    getBounds_80mus,
    getBounds_expanded_80mus
)
from .plotting import create_combined_plot
from .resolve_path import (
    resolve_model_path,
    resolve_reference_data_path,
    resolve_results_path,
    get_project_root,
    get_available_models,
    validate_model_config
)

__all__ = [
    'OptimizationTracker',
    'get_bounds',
    'getBounds_22_26_mus',
    'getBounds_80mus',
    'getBounds_expanded_80mus',
    'create_combined_plot',
    'resolve_model_path',
    'resolve_reference_data_path',
    'resolve_results_path',
    'get_project_root',
    'get_available_models',
    'validate_model_config'
]