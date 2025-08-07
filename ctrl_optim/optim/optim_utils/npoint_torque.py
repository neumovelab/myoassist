"""
N-point torque utilities.

This module provides functions for calculating and manipulating
n-point torque profiles for exoskeletons.
"""

import numpy as np
from typing import List, Union


def calculate_npoint_torques(n_points: int) -> np.ndarray:
    """Legacy geometric-decay torque initialisation.

    Mirrors the behaviour used in the original train_EXO_MyoReflex_11mus_80mus.py
    script: a peak of 0.5 at the centre (or just past the centre for even *n*)
    with surrounding points reduced by powers of two.

    Examples
    --------
    >>> calculate_npoint_torques(1)
    array([0.5])

    >>> calculate_npoint_torques(2)
    array([0.25, 0.5 ])

    >>> calculate_npoint_torques(4)
    array([0.125, 0.25 , 0.5  , 0.25 ])
    """

    if n_points <= 0:
        raise ValueError("n_points must be positive")

    # Handle 1-point case
    if n_points == 1:
        return np.array([0.5])

    # Determine peak index (4param rule: middle for odd, latter middle for even)
    peak_idx = n_points // 2 if n_points % 2 == 0 else (n_points - 1) // 2

    torques = np.zeros(n_points)
    torques[peak_idx] = 0.5

    # Geometric decay
    for i in range(n_points):
        if i == peak_idx:
            continue
        distance = abs(i - peak_idx)
        torques[i] = 0.5 / (2 ** distance)

    return torques


def interpolate_torque_profile(torque_points: np.ndarray, time_points: np.ndarray, 
                              num_samples: int = 100) -> tuple:
    """
    Interpolate an n-point torque profile to a smooth curve.
    
    Args:
        torque_points (np.ndarray): Array of torque magnitude values
        time_points (np.ndarray): Array of time points (normalized 0-1)
        num_samples (int): Number of points to sample in the interpolated curve
    
    Returns:
        tuple: (time_samples, torque_samples) interpolated arrays
    """
    import scipy.interpolate as interp
    
    # Create normalized time points
    if time_points is None:
        time_points = np.linspace(0, 1, len(torque_points))
    
    # Create interpolation function
    f = interp.CubicSpline(time_points, torque_points)
    
    # Generate samples
    time_samples = np.linspace(0, 1, num_samples)
    torque_samples = f(time_samples)
    
    return time_samples, torque_samples 