"""
Configuration parser utility for loading TestEnv from .bat files.

This module provides functionality to parse training configuration .bat files
and automatically configure TestEnv instances with the same parameters used
during optimization.
"""

import re
import os
from typing import Dict, Any, Optional, Tuple
import numpy as np


def parse_bat_config(bat_file_path: str) -> Dict[str, Any]:
    """
    Parse a .bat file and extract configuration parameters for TestEnv.
    
    Args:
        bat_file_path (str): Path to the .bat file
        
    Returns:
        Dict[str, Any]: Configuration dictionary for TestEnv
        
    Raises:
        FileNotFoundError: If the .bat file doesn't exist
        ValueError: If the .bat file cannot be parsed
    """
    
    if not os.path.exists(bat_file_path):
        raise FileNotFoundError(f"Configuration file not found: {bat_file_path}")
    
    # Default values based on TestEnv constructor
    config = {
        'sim_time': 20,
        'mode': '2D',
        'init_pose': 'walk_left',
        'slope_deg': 0,
        'delayed': False,
        'exo_bool': False,
        'fixed_exo': False,
        'n_points': 0,
        'use_4param_spline': False,
        'max_torque': 0.0,
        'model': 'baseline',
        'model_path': None
    }
    
    try:
        # Read .bat configuration file
        with open(bat_file_path, 'r') as f:
            content = f.read()
        
        content = content.replace(' ^\r\n', ' ').replace(' ^\n', ' ').replace('\r\n', ' ').replace('\n', ' ')
        
        patterns = {
            '--sim_time': (r'--sim_time\s+(\d+)', int),
            '--move_dim': (r'--move_dim\s+(\d+)', int),
            '--tgt_slope': (r'--tgt_slope\s+([\d.-]+)', float),
            '--delayed': (r'--delayed\s+(\d+)', int),
            '--ExoOn': (r'--ExoOn\s+(\d+)', int),
            '--n_points': (r'--n_points\s+(\d+)', int),
            '--max_torque': (r'--max_torque\s+([\d.]+)', float),
            '--model': (r'--model\s+(\w+)', str),
            '--pose_key': (r'--pose_key\s+(\w+)', str),
        }
        
        # Check for flags
        flag_patterns = [
            '--use_4param_spline',
            '--fixed_exo'
        ]
        
        # Parse parameter values
        for param, (pattern, type_func) in patterns.items():
            match = re.search(pattern, content)
            if match:
                value = type_func(match.group(1))
                
                # Map to TestEnv parameter names
                if param == '--sim_time':
                    config['sim_time'] = value
                elif param == '--move_dim':
                    config['mode'] = '2D' if value == 2 else '3D'
                elif param == '--tgt_slope':
                    config['slope_deg'] = value
                elif param == '--delayed':
                    config['delayed'] = bool(value)
                elif param == '--ExoOn':
                    config['exo_bool'] = bool(value)
                elif param == '--n_points':
                    config['n_points'] = value
                elif param == '--max_torque':
                    config['max_torque'] = value
                elif param == '--model':
                    config['model'] = value
                elif param == '--pose_key':
                    # Map pose_key to init_pose
                    if value == 'walk':
                        config['init_pose'] = 'walk_left'
                    else:
                        config['init_pose'] = value
        
        # Check for flags
        for flag in flag_patterns:
            if flag in content:
                if flag == '--use_4param_spline':
                    config['use_4param_spline'] = True
                elif flag == '--fixed_exo':
                    config['fixed_exo'] = True
        
        return config
        
    except Exception as e:
        raise ValueError(f"Error parsing .bat file {bat_file_path}: {str(e)}")


def create_testenv_from_bat(bat_file_path: str, params: np.ndarray, **override_kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Create a TestEnv instance from a .bat configuration file.
    
    Args:
        bat_file_path (str): Path to the .bat configuration file
        params (np.ndarray): Control parameters to use
        **override_kwargs: Any parameters to override from the .bat file
        
    Returns:
        Tuple[TestEnv, Dict]: TestEnv instance and configuration dictionary
        
    Raises:
        ImportError: If required modules cannot be imported
        FileNotFoundError: If the .bat file doesn't exist
        ValueError: If the configuration is invalid
    """
    
    try:
        # Import here to avoid circular imports
        from myoassist_reflex.reflex import reflex_interface
        
        # Parse the .bat file
        config = parse_bat_config(bat_file_path)
        config.update(override_kwargs)
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        # Construct absolute model path
        model_name = f"myoLeg22_2D_{config['model'].upper()}.xml"
        model_path = os.path.join(workspace_root, 'models', model_name)
        
        # Create the TestEnv instance with absolute model path
        TestEnv = reflex_interface.myoLeg_reflex(
            sim_time=config['sim_time'],
            mode=config['mode'],
            init_pose=config['init_pose'],
            control_params=params,
            slope_deg=config['slope_deg'],
            delayed=config['delayed'],
            exo_bool=config['exo_bool'],
            fixed_exo=config['fixed_exo'],
            n_points=config['n_points'],
            use_4param_spline=config['use_4param_spline'],
            max_torque=config['max_torque'],
            model=config['model'],
            model_path=model_path  # Pass the absolute path
        )
        
        return TestEnv, config
        
    except Exception as e:
        raise ValueError(f"Error creating TestEnv with configuration: {str(e)}")


def load_params_and_create_testenv(results_dir: str, filename: str, bat_file_path: str, **override_kwargs) -> Tuple[Any, Dict[str, Any], np.ndarray]:
    """
    Convenience function to load parameters and create TestEnv in one step.
    
    Args:
        results_dir (str): Directory containing the parameter file
        filename (str): Name of the parameter file
        bat_file_path (str): Path to the .bat configuration file
        **override_kwargs: Any parameters to override from the .bat file
        
    Returns:
        Tuple[TestEnv, Dict, np.ndarray]: TestEnv instance, configuration dictionary, and loaded parameters
        
    Raises:
        FileNotFoundError: If parameter file or .bat file doesn't exist
        ValueError: If files cannot be loaded or parsed
    """
    
    # Load parameters
    param_path = os.path.join(results_dir, filename)
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Parameter file not found: {param_path}")
    
    try:
        params = np.loadtxt(param_path)
    except Exception as e:
        raise ValueError(f"Error loading parameter file {param_path}: {str(e)}")
    
    # Create TestEnv from .bat file
    TestEnv, config = create_testenv_from_bat(bat_file_path, params, **override_kwargs)
    
    return TestEnv, config, params


def print_config_summary(config: Dict[str, Any], title: str = "Configuration Summary") -> None:
    """
    Print a formatted summary of the configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        title (str): Title for the summary
    """
    print(f"\n{title}:")
    print("=" * len(title))
    for key, value in config.items():
        print(f"  {key:<20}: {value}")
    print()


def get_available_configs(config_dir: str = "training_configs") -> list:
    """
    Get a list of available .bat configuration files.
    
    Args:
        config_dir (str): Directory containing .bat files
        
    Returns:
        list: List of available .bat file names
    """
    if not os.path.exists(config_dir):
        return []
    
    bat_files = [f for f in os.listdir(config_dir) if f.endswith('.bat')]
    return sorted(bat_files)


# Example usage functions to modify as necessary
def load_exo_4param_kine_config(results_dir: str = "results/exo_4param", 
                               filename: str = "example_results.txt",
                               config_dir: str = "../training_configs") -> Tuple[Any, Dict[str, Any], np.ndarray]:
    """
    Convenience function to load the exo_4params_kine configuration.
    
    Args:
        results_dir (str): Directory containing the parameter file
        filename (str): Name of the parameter file  
        config_dir (str): Directory containing .bat files
        
    Returns:
        Tuple[TestEnv, Dict, np.ndarray]: TestEnv instance, configuration, and parameters
    """
    bat_path = os.path.join(config_dir, "exo_4param_kine.bat")
    return load_params_and_create_testenv(results_dir, filename, bat_path)


if __name__ == "__main__":
    print("Available configurations:")
    configs = get_available_configs("../training_configs")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config}") 