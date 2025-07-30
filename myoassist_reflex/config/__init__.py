"""
Configuration modules for myoassist.

This package contains modules for configuration, argument parsing, 
and environment setup.
"""

from .arg_parser import initParser
from .environment import create_environment_dict, get_optimization_type, get_optimization_suffix

__all__ = [
    'initParser', 
    'create_environment_dict', 
    'get_optimization_type', 
    'get_optimization_suffix'
] 