"""
Effort cost functions.

This module contains functions for calculating metabolic and effort costs
during gait optimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional


def calculate_effort_cost(
    muscle_activations: np.ndarray,
    step_indices: np.ndarray,
    distance_traveled: float,
    mass: float,
    muscle_indices: Optional[List[int]] = None
) -> float:
    """
    Calculate metabolic effort cost normalized by distance and mass.
    
    The cost is based on the sum of squared muscle activations over the
    evaluation period, divided by the distance traveled and the model's mass.
    Matches original implementation exactly.
    
    Args:
        muscle_activations (np.ndarray): Muscle activation values over time
                                        Shape: [timesteps, num_muscles]
        step_indices (np.ndarray): Indices marking the start and end of evaluation period
        distance_traveled (float): Total distance traveled in meters
        mass (float): Mass of the model in kg
        muscle_indices (List[int], optional): Indices of muscles to include in calculation
                                             If None, all muscles are included
    
    Returns:
        float: Calculated effort cost
    """
    start_idx = step_indices[0]
    end_idx = step_indices[-1] + 1
    
    if muscle_indices is not None:
        activations = muscle_activations[start_idx:end_idx, muscle_indices]
    else:
        activations = muscle_activations[start_idx:end_idx, :]
    
    # Match original implementation exactly:
    # effort_cost = np.sum(np.square(unpacked_act[index_vector[0]:index_vector[-1]+1, musc_idx])) / (cost_mass*np.linalg.norm(steps_dist))
    squared_activations = np.sum(np.square(activations))
    effort_cost = squared_activations / (mass * np.linalg.norm([distance_traveled]))
    return effort_cost


def calculate_emg_profile_cost(
    muscle_activations: np.ndarray,
    ref_emg: np.ndarray,
    muscles_dict: Dict[str, Dict[str, List[int]]],
    eval_ctrl_mode: str,
    step_indices: np.ndarray,
    n_strides: int
) -> float:
    """
    Calculate cost based on matching EMG profiles to reference data.
    
    Args:
        muscle_activations (np.ndarray): Muscle activation values over time
                                        Shape: [timesteps, num_muscles]
        ref_emg (np.ndarray): Reference EMG data
                             Shape: [100, num_ref_muscles]
        muscles_dict (Dict): Dictionary mapping muscle names to indices for both legs
        eval_ctrl_mode (str): Control mode ('2D' or '3D')
        step_indices (np.ndarray): Indices marking stride boundaries
        n_strides (int): Number of strides to evaluate
    
    Returns:
        float: EMG profile matching cost
    """
    # Select muscles based on control mode - exact match with original
    if eval_ctrl_mode == '2D':
        musc_fields = ['HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']
    else:  # 3D mode
        musc_fields = ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']
    
    # Create muscle name to index mapping for EMG data
    musc_map = dict(zip(['HAB','HAD','HFL','GLU','HAM','RF','VAS','BFSH','GAS','SOL','TA'], 
                       range(len(['HAB','HAD','HFL','GLU','HAM','RF','VAS','BFSH','GAS','SOL','TA']))))
    
    # Initialize interpolation matrix
    interp_emg = np.zeros((100, muscle_activations.shape[1], n_strides))
    
    # Interpolate each stride to 100 points
    stride_indices = np.arange(0, len(step_indices) - 2, 2)  # Use every other step index for full strides
    counter = 0
    
    for stride_idx in stride_indices:
        if counter >= n_strides:
            break
            
        start_idx = step_indices[stride_idx]
        end_idx = step_indices[stride_idx + 2]  # Use two steps ahead for full stride
        
        extract_act = muscle_activations[start_idx:end_idx, :]
        
        # Interpolate each muscle's activation to 100 points
        for mus_idx in range(muscle_activations.shape[1]):
            interp_emg[:, mus_idx, counter] = np.interp(
                np.linspace(0, extract_act.shape[0], 100),
                np.arange(0, extract_act.shape[0]),
                extract_act[:, mus_idx]
            )
        counter += 1
    
    # Calculate EMG profile cost - match original code exactly
    musc_profile_cost = 0
    
    for musc in musc_fields:
        if musc in muscles_dict['l_leg']:  # Only compare muscles that have EMG data
            # Get indices for this muscle type from both legs
            musc_indices = []
            if musc in muscles_dict['l_leg']:
                musc_indices.extend(muscles_dict['l_leg'][musc])
            if musc in muscles_dict['r_leg']:
                musc_indices.extend(muscles_dict['r_leg'][musc])
            
            # Compare each muscle's activation to the EMG profile
            for stride in range(n_strides):
                for idx in musc_indices:
                    # Use squared difference between interpolated activation and reference EMG
                    musc_profile_cost += np.sum(
                        np.square(interp_emg[:, idx, stride] - ref_emg[:, musc_map[musc]])
                    ) / n_strides
    
    return musc_profile_cost


def calculate_joint_limit_cost(
    joint_limit_torques: np.ndarray,
    step_indices: np.ndarray,
    joint_groups: Optional[Dict[str, List[int]]] = None
) -> Dict[str, float]:
    """
    Calculate costs related to joint limit violations.
    
    Args:
        joint_limit_torques (np.ndarray): Joint limit torques over time
                                         Shape: [timesteps, num_joints]
        step_indices (np.ndarray): Indices marking the start and end of evaluation period
        joint_groups (Dict[str, List[int]], optional): Dictionary mapping joint groups to indices
                                                     Default groups: hip, knee, ankle
    
    Returns:
        Dict[str, float]: Dictionary of joint limit costs by group
    """
    start_idx = step_indices[0]
    end_idx = step_indices[-1] + 1
    
    torques = joint_limit_torques[start_idx:end_idx, :]
    
    if joint_groups is None:
        joint_groups = {
            'hip': [0, 1],    # Left and right hip
            'knee': [2, 3],   # Left and right knee
            'ankle': [4, 5]   # Left and right ankle
        }
    
    costs = {}
    duration = end_idx - start_idx
    
    # Calculate costs for each joint group
    for group_name, indices in joint_groups.items():
        group_torques = torques[:, indices]
        # Use absolute sum normalized by duration, matching original implementation
        costs[group_name] = np.sum(np.abs(group_torques)) / duration
    
    # Add total cost
    costs['total'] = sum(costs.values())
    return costs 