"""
Command line argument parser for myoassist.

This module contains the functions to parse command line arguments
for the myoassist tool.
"""

import argparse
from typing import Dict, Any, Optional


def initParser() -> argparse.Namespace:
    """
    Initialize and return the command line argument parser.
    
    Returns:
        argparse.Namespace: The parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="NeuMove MyoReflex Optimization Tool")
    group = parser.add_mutually_exclusive_group()

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--musc_model", 
                           help="(String) The muscle model to use. Currently supports only [leg_80, leg_11] "
                                "for 80 muscles and 11 muscles leg model respectively")
    model_group.add_argument("--delayed", type=int, 
                           help="(int) Delayed mode, 1 to activate")
    model_group.add_argument("--reflex_mode", required=False, 
                           help="(String) Unified or Individual [uni, ind]. Used only for 80 mus reflex model")
    model_group.add_argument("--move_dim", type=int, 
                           help="(int) Whether the model operates in 2D or 3D")
    model_group.add_argument("--model", type=str, default="default", 
                           choices=["baseline", "dephy", "hmedi", "humotech"], 
                           help="Type of model to use for simulation")
    model_group.add_argument("--model_path", type=str, default=None, 
                           help="Optional: Custom path to model XML file. Only used if model_type is 'custom'")

    # Exoskeleton configuration
    exo_group = parser.add_argument_group("Exoskeleton Configuration")
    exo_group.add_argument("--ExoOn", type=int, 
                         help="(int) 1 for Exo on, 0 otherwise")
    exo_group.add_argument("--use_4param_spline", action="store_true", 
                         help="Flag to use legacy 4-point spline controller")
    exo_group.add_argument("--fixed_exo", action="store_true", 
                         help="Keep exoskeleton parameters fixed at initial values during optimization")
    exo_group.add_argument("--n_points", type=int, default=4, required=False, 
                         help="(int) Number of points in exo torque spline (min 2, ignored if use_4param_spline is True)")
    exo_group.add_argument("--max_torque", type=float, default=10.0, required=False,
                         help="(float) Maximum torque allowed in the exoskeleton controller")

    # Optimization configuration
    optim_group = parser.add_argument_group("Optimization Configuration")
    optim_group.add_argument("--optim_mode", 
                           help="(String) Optimization to be done. Currently supports only "
                                "[evaluate, single, multispeed, multislope]")
    optim_group.add_argument("--optim_params", type=float, nargs='+', 
                           help="(float) List of target velocities or slopes")
    optim_group.add_argument("--popsize", type=int, 
                           help="(int) Population size for CMA-ES")
    optim_group.add_argument("--maxiter", type=int, 
                           help="(int) Max iteration to run")
    optim_group.add_argument("--threads", type=int, 
                           help="(int) Number of threads for CMA-ES")
    optim_group.add_argument("--sigma_gain", type=int, 
                           help="(int) Multipliers for initial sigma value of 0.01")

    # Simulation parameters
    sim_group = parser.add_argument_group("Simulation Parameters")
    sim_group.add_argument("--sim_time", type=int, 
                         help="(int) Max simulation time (in seconds)")
    sim_group.add_argument("--num_strides", type=int, 
                         help="(int) Number of minimum strides to calculate cost")
    sim_group.add_argument("--tgt_vel", type=float, 
                         help="(float) Target velocity to optimize for")
    sim_group.add_argument("--tgt_slope", type=float, 
                         help="(float) Target slope (in degrees) to optimize for")
    sim_group.add_argument("--pose_key", required=False, 
                         help="(String) Initial keypose of model")
    sim_group.add_argument("--tgt_sym_th", type=float, 
                         help="(float) Threshold difference for symmetry")
    sim_group.add_argument("--tgt_grf_th", type=float, 
                         help="(float) Threshold for normalized GRF")
    
    # Cost function configuration
    cost_group = parser.add_argument_group("Cost Function Configuration")
    cost_group.add_argument("--trunk_err_type", 
                          help="(String) type of trunk error, from ['ref_diff','zero_diff','vel_square']")

    # Cost function types (mutually exclusive)
    group.add_argument("-eff", "--effort", action="store_true", 
                     help="Flag for Effort (Cost of Transport) optimization")
    group.add_argument("-eff_knee", "--effort_knee", action="store_true", 
                     help="Flag CoT and Knee pain")
    group.add_argument("-vel", "--velocity", action="store_true", 
                     help="Velocity Only")
    group.add_argument("-vel_grf", "--velocity_grf", action="store_true", 
                     help="Velocity and GRF threshold")
    group.add_argument("-class", "--classic", action="store_true", 
                     help="Flag for Effort+Velocity+KneeOver optimization")
    group.add_argument("-kine", "--kinematics", action="store_true", 
                     help="Flag for Kinematics optimization")
    group.add_argument("-kine_grf", "--kinematics_grf", action="store_true", 
                     help="Flag for Kinematics optimization")
    group.add_argument("-combined", "--combined", action="store_true", 
                     help="Flag to combine both effort and kinematics cost")
    group.add_argument("-kine_grf_musc", "--kinematics_grf_musc", action="store_true", 
                     help="Flag for Kinematics optimization")
    group.add_argument("-vel_musc", "--vel_musc", action="store_true", 
                     help="Velocity Muscle profile")
    group.add_argument("-vel_musc_grf", "--vel_musc_grf", action="store_true", 
                     help="Vel Musc Profile GRF")

    # Output and misc options
    output_group = parser.add_argument_group("Output and Misc Options")
    output_group.add_argument("--runSuffix", 
                            help="(String) Suffix added to the end of the savefile")
    output_group.add_argument("-clu", "--cluster", action="store_true", 
                            help="Flag for script on cluster or local machine")
    output_group.add_argument("--cost_print", action="store_true", 
                            help="Flag to determine evaluation mode of cost function")
    output_group.add_argument("--param_path", required=False, 
                            help="(String) Path of param file, takes the first file in the directory")
    output_group.add_argument("--save_path", required=False, 
                            help="(String) Path to save outputs")
    output_group.add_argument("--pickle_path", required=False, 
                            help="(String) Path of pickle file, takes the first file in the directory")

    return parser.parse_args()


def get_optimization_type(args: argparse.Namespace) -> str:
    """
    Determine the optimization type based on command line arguments.
    
    Args:
        args (argparse.Namespace): The parsed command line arguments
        
    Returns:
        str: The optimization type
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
        return 'Effort'  # Default


def create_environment_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Create environment dictionary from parsed arguments.
    
    Args:
        args (argparse.Namespace): The parsed command line arguments
        
    Returns:
        Dict[str, Any]: Dictionary with environment configuration
    """
    # Set control mode based on move_dim
    flag_ctrl_mode = '2D' if args.move_dim == 2 else '3D'
    
    # Set delayed mode
    delayed = True if args.delayed == 1 else False
    
    # Set exo_bool
    exo_bool = True if args.ExoOn == 1 else False
    
    # Set reflex_mode (default to unified)
    reflex_mode = args.reflex_mode if args.reflex_mode else 'uni'
    isUnified = True if reflex_mode == 'uni' else False
    
    # Create environment dictionary
    env_dict = {
        'leg_model': args.musc_model,
        'init_pose': args.pose_key,
        'mode': flag_ctrl_mode,
        'sim_time': args.sim_time,
        'seed': 0,  # Default seed
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