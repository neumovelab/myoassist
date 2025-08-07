"""
Walking cost function.

This module contains the main walking cost function used to evaluate
the performance of the neuromuscular reflex controller.
"""

import os
import numpy as np
from typing import Dict, List, Union, Any, Optional, Tuple
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium.core')

from .evaluate_cost import evaluateCost
from ctrl_optim.ctrl.reflex.reflex_interface import myoLeg_reflex


def func_Walk_FitCost(
    params: np.ndarray, 
    optim_type: str, 
    one_step: np.ndarray, 
    one_EMG: np.ndarray, 
    trunk_err_type: str, 
    input_tgt_vel: float, 
    stride_num: int, 
    tgt_sym: float, 
    tgt_grf: float, 
    env_dict: Optional[Dict[str, Any]] = None, 
    cost_print: bool = False
) -> Union[float, Dict[str, Any]]:
    """
    Evaluate the walking fitness cost of a set of parameters.
    
    This function creates a simulation environment, runs the simulation with the
    provided parameters, and evaluates the cost of the resulting motion.
    
    Args:
        params (np.ndarray): Controller parameters to evaluate
        optim_type (str): Type of optimization (e.g., 'Velocity', 'Effort', etc.)
        one_step (np.ndarray): Reference kinematic data for one step
        one_EMG (np.ndarray): Reference EMG data
        trunk_err_type (str): Type of trunk error to calculate
        input_tgt_vel (float): Target velocity
        stride_num (int): Number of strides to evaluate
        tgt_sym (float): Target symmetry threshold
        tgt_grf (float): Target ground reaction force threshold
        env_dict (Dict[str, Any], optional): Environment configuration
        cost_print (bool): Whether to print detailed cost information
        
    Returns:
        Union[float, Dict[str, Any]]: Cost value or dictionary of cost components
    """
    # Initialize environment based on model type
    if env_dict['leg_model'] in ['80']:
        from optim.reflex import ReflexInterface_11mus_80mus # NOT CURRENTLY CONFIGURED FOR MYOASSIST

        Myo_env = ReflexInterface_11mus_80mus.MyoLegReflex(
            init_pose=env_dict['init_pose'], 
            mode=env_dict['mode'], 
            sim_time=env_dict['sim_time'],
            seed=env_dict['seed'], 
            unified=env_dict['unified'], 
            control_params=params, 
            slope_deg=env_dict['slope_deg'], 
            delayed=env_dict['delayed']
        )
    elif env_dict['leg_model'] in ['22', '26']:

        Myo_env = myoLeg_reflex(
            init_pose=env_dict['init_pose'], 
            mode=env_dict['mode'], 
            sim_time=env_dict['sim_time'],
            seed=env_dict['seed'], 
            control_params=params, 
            slope_deg=env_dict['slope_deg'], 
            delayed=env_dict['delayed'], 
            exo_bool=env_dict['exo_bool'], 
            n_points=env_dict['n_points'], 
            use_4param_spline=env_dict['use_4param_spline'], 
            fixed_exo=env_dict['fixed_exo'], 
            max_torque=env_dict['max_torque'],
            model=env_dict['model'],
            model_path=env_dict['model_path'],
            leg_model=env_dict['leg_model']
        )
    else:
        raise ValueError(f"Unsupported leg model: {env_dict['leg_model']}")

    try:
        Myo_env.reset(params)
        pose_valid = Myo_env.check_pose_validity()
        
        if not pose_valid:
            return 120 * 10000

    except Exception as e_msg:
        print("Simulation error!!!")
        print(e_msg)
        
        if cost_print:
            print("Simulation Error")
            out_dict = {
                'Solution': 'None',
                'Cost': 120 * 10000
            }
            return out_dict
        else:
            return 120 * 10000

    data_store = []
    
    for i in range(Myo_env.timestep_limit):
        cost_dict, time, done_flag = Myo_env.run_reflex_step_Cost()
        
        data_store.append({
            'obj_func_out': cost_dict.copy(),
            'sim_time': time
        })
        
        if done_flag:
            break
    
    muslen_param = params[-1 * len(Myo_env.mus_len_key):]
    
    final_cost = evaluateCost(
        data_store, 
        Myo_env.dt, 
        Myo_env.mode, 
        Myo_env.timestep_limit, 
        Myo_env.slope_deg, 
        Myo_env.muscles_dict, 
        optim_type, 
        one_step, 
        one_EMG, 
        trunk_err_type, 
        input_tgt_vel, 
        stride_num, 
        tgt_sym, 
        tgt_grf, 
        muslen_param, 
        eval_mode=cost_print
    )
    
    return final_cost 