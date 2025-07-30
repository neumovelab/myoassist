"""
Kinematic cost functions.

This module contains functions for calculating kinematic costs
for comparing simulated gait to reference data.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional


def calculate_kinematic_costs(
    unpacked_angles: np.ndarray,
    one_step: np.ndarray,
    n_strides: int,
    index_vector: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate kinematic costs by comparing simulated and reference joint angles.
    This version uses event-based stride segmentation to match the original framework.
    
    Args:
        unpacked_angles (np.ndarray): Joint angles from simulation
                                     Shape: [timesteps, num_angles]
        one_step (np.ndarray): Reference joint angles from normal gait
                              Shape: [100, num_angles]
        n_strides (int): The target number of strides to evaluate.
        index_vector (np.ndarray): Array of timestamps for detected heel strikes.
        
    Returns:
        Tuple[float, float]: (kinematic_cost, trunk_cost)
    """
    num_angles = unpacked_angles.shape[1]

    # Determine stride boundaries from the event-based index_vector
    stride_indices = np.arange(0, len(index_vector) - 2, 2)
    actual_n_strides = len(stride_indices)

    if actual_n_strides == 0:
        return 0.0, 0.0  # No full strides to evaluate

    # Initialize interpolation matrix
    interp_mat = np.zeros((100, num_angles, actual_n_strides))

    # Interpolate each stride to 100 points
    for i, stride_start_idx in enumerate(stride_indices):
        start_idx = index_vector[stride_start_idx]
        end_idx = index_vector[stride_start_idx + 2]  # Full stride is two steps
        
        stride_data = unpacked_angles[start_idx:end_idx]
        
        if stride_data.shape[0] == 0:
            continue  # Skip empty strides

        for ang_idx in range(num_angles):
            interp_mat[:, ang_idx, i] = np.interp(
                np.linspace(0, stride_data.shape[0], 100),
                np.arange(stride_data.shape[0]),
                stride_data[:, ang_idx]
            )

    # Calculate squared differences from reference
    err_gains = np.ones((100, num_angles))
    squared_diff = np.zeros_like(interp_mat)
    
    for ang_idx in range(num_angles):
        # Tile reference data for the number of actual strides found
        tiled_step = np.tile(one_step[:, ang_idx], (actual_n_strides, 1)).T
        squared_diff[:, ang_idx, :] = err_gains[:, ang_idx].reshape(-1, 1) * (
            np.sqrt(np.square(interp_mat[:, ang_idx, :] - tiled_step)) / actual_n_strides
        )
    
    # Sum up costs excluding trunk
    summed = np.sum(squared_diff, axis=2)  # Sum across strides
    kinematic_cost = np.sum(summed[:, 1:])  # Sum all joints except trunk
    
    # Calculate trunk cost
    trunk_cost = np.sum(summed[:, 0])
    
    return kinematic_cost, trunk_cost


def calculate_trunk_cost(
    unpacked_angles: np.ndarray,
    trunk_err_type: str,
    const_theta_tgt: float = 0.0
) -> float:
    """
    Calculate trunk-specific cost based on specified error type.
    
    Args:
        unpacked_angles (np.ndarray): Joint angles including trunk
        trunk_err_type (str): Type of trunk error to calculate
                             One of: 'ref_diff', 'tgt_diff', 'zero_diff',
                                    'vel_square', 'no_trunk'
        const_theta_tgt (float): Target trunk angle for 'tgt_diff' mode
        
    Returns:
        float: Calculated trunk cost
    """
    trunk_angles = unpacked_angles[:, 0]  # First column is trunk
    trunk_gain = 1.0
    
    if trunk_err_type == 'ref_diff':
        # Difference from reference
        return np.sum(np.abs(trunk_angles))
    elif trunk_err_type == 'tgt_diff':
        # Difference from target
        return np.sum(trunk_gain * np.square(trunk_angles - const_theta_tgt))
    elif trunk_err_type == 'zero_diff':
        # Difference from zero
        return np.sum(trunk_gain * np.square(trunk_angles))
    elif trunk_err_type == 'vel_square':
        # Square of velocity (requires velocity data)
        trunk_vel = np.diff(trunk_angles)
        return np.sum(np.square(trunk_vel))
    elif trunk_err_type == 'no_trunk':
        return 0.0
    else:
        return 0.0


def interpolate_gait_cycle(
    data: np.ndarray,
    n_points: int = 100
) -> np.ndarray:
    """
    Interpolate data to specified number of points (usually 100 for % gait cycle).
    
    Args:
        data (np.ndarray): Data to interpolate, shape [timesteps, channels]
        n_points (int): Number of points to interpolate to
        
    Returns:
        np.ndarray: Interpolated data, shape [n_points, channels]
    """
    timesteps = data.shape[0]
    old_x = np.arange(timesteps)
    new_x = np.linspace(0, timesteps-1, n_points)
    
    if data.ndim == 1:
        return np.interp(new_x, old_x, data)
    else:
        result = np.zeros((n_points, data.shape[1]))
        for i in range(data.shape[1]):
            result[:, i] = np.interp(new_x, old_x, data[:, i])
        return result 