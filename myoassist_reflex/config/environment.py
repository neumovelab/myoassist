"""
Environment configuration for optimization.

This module contains functions for creating and managing environment
configurations used in the optimization process.
"""

from typing import Dict, Any, Optional
import argparse


def create_environment_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Create a dictionary of environment settings from command line arguments.
    
    Args:
        args (argparse.Namespace): Command line arguments
        
    Returns:
        Dict[str, Any]: Environment configuration dictionary
    """
    # Set up unified flag based on reflex mode
    if args.musc_model == 'leg_80':
        # Default to unified if not specified
        if args.reflex_mode is None:
            reflex_mode = 'uni'
        else:
            reflex_mode = args.reflex_mode
            
        isUnified = (reflex_mode == 'uni')
    else:
        # For 11-muscle model, unified is not applicable
        isUnified = False

    # Set control mode based on muscle model
    if args.musc_model in ['22']:
        flag_ctrl_mode = '2D'
    elif args.musc_model in ['26', '80']:
        flag_ctrl_mode = '3D'
    else:
        raise ValueError(f"Invalid muscle model: {args.musc_model}")
    
    # Set up exoskeleton flag
    exo_bool = (args.ExoOn == 1)
    
    # Set up delayed flag
    delayed = (args.delayed == 1)
    
    # Create environment dictionary
    env_dict = {
        'leg_model': args.musc_model,
        'init_pose': args.pose_key,
        'mode': flag_ctrl_mode,
        'sim_time': args.sim_time,
        'seed': 0,  # Fixed seed for reproducibility
        'unified': isUnified,
        'slope_deg': args.tgt_slope,
        'delayed': delayed,
        'exo_bool': exo_bool,
        'n_points': args.n_points,
        'use_4param_spline': args.use_4param_spline,
        'fixed_exo': args.fixed_exo,
        'max_torque': args.max_torque,
        'model': args.model,
        'model_path': args.model_path
    }
    
    return env_dict


def get_optimization_type(args: argparse.Namespace) -> str:
    """
    Determine the optimization type from command line arguments.
    
    Args:
        args (argparse.Namespace): Command line arguments
        
    Returns:
        str: Optimization type identifier
    """
    if args.effort:
        return 'Effort'
    elif args.effort_knee:
        return 'Eff_Knee'
    elif args.classic:
        return 'Classic'
    elif args.kinematics:
        return 'Kine'
    elif args.combined:
        return 'Combined'
    elif args.velocity:
        return 'Velocity'
    elif args.velocity_grf:
        return 'Vel_grf'
    elif args.kinematics_grf:
        return 'Kine_grf'
    elif args.kinematics_grf_musc:
        return 'Kine_grf_musc'
    elif args.vel_musc:
        return 'vel_musc'
    elif args.vel_musc_grf:
        return 'vel_musc_grf'
    else:
        # Default to velocity optimization
        return 'Velocity'


def get_optimization_suffix(optim_type: str) -> str:
    """
    Get a short suffix for the optimization type for file naming.
    
    Args:
        optim_type (str): Optimization type identifier
        
    Returns:
        str: Short suffix for file naming
    """
    suffix_map = {
        'Effort': 'Eff',
        'Eff_Knee': 'Eff_Kne',
        'Classic': 'Class',
        'Kine': 'Kine',
        'Combined': 'Comb',
        'Velocity': 'Vel',
        'Vel_grf': 'Vel_grf',
        'Kine_grf': 'Kine_grf',
        'Kine_grf_musc': 'Kine_grf_musc',
        'vel_musc': 'vel_musc',
        'vel_musc_grf': 'vel_musc_grf'
    }
    
    return suffix_map.get(optim_type, 'Unk') 