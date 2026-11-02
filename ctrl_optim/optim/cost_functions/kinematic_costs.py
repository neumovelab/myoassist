"""
Kinematic cost functions.

This module contains functions for calculating kinematic costs
for comparing simulated gait to reference data.
"""

import numpy as np
from typing import Optional, Tuple


TRUNK_ERR_TYPES = ("ref_diff", "tgt_diff", "zero_diff", "vel_square", "no_trunk")


def resample_strides(series: np.ndarray, index_vector: np.ndarray) -> np.ndarray:
    """Resample each stride of `series` [timesteps, channels] to 100 samples.

    Strides run between every second heel strike of the event-based index_vector.
    Returns [100, channels, n_strides]; an empty stride stays zero.
    """
    stride_indices = np.arange(0, len(index_vector) - 2, 2)
    out = np.zeros((100, series.shape[1], len(stride_indices)))
    for i, stride_start_idx in enumerate(stride_indices):
        start_idx = index_vector[stride_start_idx]
        end_idx = index_vector[stride_start_idx + 2]  # Full stride is two steps
        stride_data = series[start_idx:end_idx]
        if stride_data.shape[0] == 0:
            continue  # Skip empty strides
        for ch in range(series.shape[1]):
            out[:, ch, i] = np.interp(
                np.linspace(0, stride_data.shape[0], 100), np.arange(stride_data.shape[0]), stride_data[:, ch]
            )
    return out


def calculate_kinematic_costs(
    unpacked_angles: np.ndarray,
    one_step: np.ndarray,
    n_strides: int,
    index_vector: np.ndarray,
    trunk_err_type: str = "ref_diff",
    trunk_vel: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Calculate kinematic costs by comparing simulated and reference joint angles.
    This version uses event-based stride segmentation to match the original framework.

    The kinematic cost is the absolute error from the reference, averaged over strides
    and summed over the 100 samples of the gait cycle and the hip, knee, and ankle of
    both legs. The trunk cost is set by trunk_err_type (see calculate_trunk_cost).

    Args:
        unpacked_angles (np.ndarray): Joint angles from simulation
                                     Shape: [timesteps, num_angles]; column 0 is trunk pitch
        one_step (np.ndarray): Reference joint angles from normal gait
                              Shape: [100, num_angles]
        n_strides (int): The target number of strides to evaluate.
        index_vector (np.ndarray): Array of timestamps for detected heel strikes.
        trunk_err_type (str): Trunk term, one of TRUNK_ERR_TYPES (None: 'ref_diff').
        trunk_vel (np.ndarray, optional): Trunk pitch rate (rad/s) per timestep, for 'vel_square'.

    Returns:
        Tuple[float, float]: (kinematic_cost, trunk_cost)
    """
    interp_mat = resample_strides(unpacked_angles, index_vector)
    actual_n_strides = interp_mat.shape[2]

    if actual_n_strides == 0:
        return 0.0, 0.0  # No full strides to evaluate

    # Absolute error from the reference, averaged over strides
    abs_err = np.abs(interp_mat - one_step[:, :, None]) / actual_n_strides
    summed = np.sum(abs_err, axis=2)  # Sum across strides
    kinematic_cost = np.sum(summed[:, 1:])  # Sum all joints except trunk

    trunk_vel_mat = None if trunk_vel is None else resample_strides(trunk_vel[:, None], index_vector)[:, 0, :]
    trunk_cost = calculate_trunk_cost(interp_mat[:, 0, :], one_step[:, 0], trunk_err_type, trunk_vel=trunk_vel_mat)

    return kinematic_cost, trunk_cost


def calculate_trunk_cost(
    trunk_strides: np.ndarray,
    trunk_ref: np.ndarray,
    trunk_err_type: Optional[str],
    const_theta_tgt: float = 0.0,
    trunk_vel: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate the trunk cost for the selected error type. Every type is summed over
    the 100 samples of the gait cycle and averaged over strides.

    Args:
        trunk_strides (np.ndarray): Trunk pitch (rad) per stride, [100, n_strides]
        trunk_ref (np.ndarray): Reference trunk pitch (rad), [100]
        trunk_err_type (str): One of TRUNK_ERR_TYPES; None means 'ref_diff'.
                              'ref_diff': absolute error from the reference (rad)
                              'tgt_diff': squared error from const_theta_tgt (rad^2)
                              'zero_diff': squared trunk pitch (rad^2)
                              'vel_square': squared trunk pitch rate ((rad/s)^2)
                              'no_trunk': 0
        const_theta_tgt (float): Target trunk angle for 'tgt_diff' mode
        trunk_vel (np.ndarray, optional): Trunk pitch rate per stride, [100, n_strides]

    Returns:
        float: Calculated trunk cost
    """
    n_strides = trunk_strides.shape[1]
    if trunk_err_type is None or trunk_err_type == "ref_diff":
        return float(np.sum(np.abs(trunk_strides - trunk_ref[:, None])) / n_strides)
    if trunk_err_type == "tgt_diff":
        return float(np.sum(np.square(trunk_strides - const_theta_tgt)) / n_strides)
    if trunk_err_type == "zero_diff":
        return float(np.sum(np.square(trunk_strides)) / n_strides)
    if trunk_err_type == "vel_square":
        if trunk_vel is None:
            raise ValueError("trunk_err_type 'vel_square' needs the trunk pitch rate")
        return float(np.sum(np.square(trunk_vel)) / n_strides)
    if trunk_err_type == "no_trunk":
        return 0.0
    raise ValueError(f"unknown trunk_err_type {trunk_err_type!r}; expected one of {TRUNK_ERR_TYPES}")


def interpolate_gait_cycle(data: np.ndarray, n_points: int = 100) -> np.ndarray:
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
    new_x = np.linspace(0, timesteps - 1, n_points)

    if data.ndim == 1:
        return np.interp(new_x, old_x, data)
    else:
        result = np.zeros((n_points, data.shape[1]))
        for i in range(data.shape[1]):
            result[:, i] = np.interp(new_x, old_x, data[:, i])
        return result
