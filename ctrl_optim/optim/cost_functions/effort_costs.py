"""
Effort cost functions.

This module contains the muscle-effort, EMG-tracking, and joint-limit cost terms of
the gait optimization.

The cumulative terms (effort, joint-limit reliance, and the GRF limit in
evaluate_cost) are time integrals over the evaluation window divided by the
distance traveled in the window, so they measure the accumulated penalty per meter.
The integrals use the control period dt, so their values do not depend on it.
"""

import numpy as np
from typing import Dict, List, Optional

from .kinematic_costs import resample_strides

GRAVITY = 9.81  # m/s^2


def calculate_effort_cost(
    muscle_activations: np.ndarray,
    step_indices: np.ndarray,
    distance_traveled: float,
    dt: float,
    muscle_indices: Optional[List[int]] = None,
) -> float:
    """
    Muscle effort: the time integral of the squared activations, summed over the
    muscles and divided by the distance traveled and the number of muscles.

        C_E = sum_m int_T a_m(t)^2 dt / (D * n_m)      (s/m)

    C_E equals the mean squared activation divided by the mean speed, so C_E = 1
    means that every muscle is fully active at 1 m/s.

    Args:
        muscle_activations (np.ndarray): Muscle activations, [timesteps, num_muscles]
        step_indices (np.ndarray): Heel-strike indices; the window runs from the first to the last
        distance_traveled (float): Distance traveled in the window (m)
        dt (float): Control period (s), the time between rows of muscle_activations
        muscle_indices (List[int], optional): Muscles to include; None includes all

    Returns:
        float: Effort cost (s/m)
    """
    start_idx = step_indices[0]
    end_idx = step_indices[-1]

    if muscle_indices is not None:
        activations = muscle_activations[start_idx:end_idx, muscle_indices]
    else:
        activations = muscle_activations[start_idx:end_idx, :]

    n_muscles = activations.shape[1]
    integral = dt * np.sum(np.square(activations))
    return float(integral / (abs(distance_traveled) * n_muscles))


def calculate_emg_profile_cost(
    muscle_activations: np.ndarray,
    ref_emg: np.ndarray,
    muscles_dict: Dict[str, Dict[str, List[int]]],
    eval_ctrl_mode: str,
    step_indices: np.ndarray,
) -> float:
    """
    EMG tracking: the squared difference between the activation of each muscle and
    the reference EMG of its group, after each stride is resampled to 100 samples.
    The difference is averaged over the samples, the strides, and the muscles of each
    group (both legs), then over the groups. The value is dimensionless; the gait
    duration does not change it.

    Args:
        muscle_activations (np.ndarray): Muscle activations, [timesteps, num_muscles]
        ref_emg (np.ndarray): Reference EMG, [100, 11] (groups in the order of emg_groups)
        muscles_dict (Dict): Muscle indices of each group, for both legs
        eval_ctrl_mode (str): Control mode ('2D' or '3D')
        step_indices (np.ndarray): Heel-strike indices; strides run between every second one

    Returns:
        float: EMG tracking cost (dimensionless)
    """
    emg_groups = ["HAB", "HAD", "HFL", "GLU", "HAM", "RF", "VAS", "BFSH", "GAS", "SOL", "TA"]
    if eval_ctrl_mode == "2D":
        musc_fields = ["HFL", "GLU", "HAM", "RF", "VAS", "BFSH", "GAS", "SOL", "TA"]
    else:
        musc_fields = emg_groups

    interp_act = resample_strides(muscle_activations, step_indices)  # [100, muscles, strides]
    if interp_act.shape[2] == 0:
        return 0.0

    group_costs = []
    for musc in musc_fields:
        if musc not in muscles_dict["l_leg"]:  # only the groups the model has
            continue
        musc_indices = list(muscles_dict["l_leg"][musc]) + list(muscles_dict["r_leg"].get(musc, []))
        ref = ref_emg[:, emg_groups.index(musc)]
        group_costs.append(np.mean(np.square(interp_act[:, musc_indices, :] - ref[:, None, None])))

    return float(np.mean(group_costs)) if group_costs else 0.0


def calculate_joint_limit_cost(
    joint_limit_torques: np.ndarray,
    step_indices: np.ndarray,
    dt: float,
    distance_traveled: float,
    mass: float,
    leg_length: float,
    joint_groups: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, float]:
    """
    Joint-limit reliance: the time integral of the magnitude of the joint-limit
    constraint torques, divided by the distance traveled, the number of joints, and
    the body weight times the leg length.

        P = sum_j int_T |tau_j(t)| dt / (D * n_j * m g L)      (s/m)

    Args:
        joint_limit_torques (np.ndarray): Joint-limit torques (N m), [timesteps, joints]
        step_indices (np.ndarray): Heel-strike indices; the window runs from the first to the last
        dt (float): Control period (s)
        distance_traveled (float): Distance traveled in the window (m)
        mass (float): Total mass (kg), device included
        leg_length (float): Leg length (m), hip to ankle
        joint_groups (Dict[str, List[int]], optional): Columns of each joint group.
                                                     Default: hip, knee, ankle, each left then right

    Returns:
        Dict[str, float]: P of each group (over its joints) and "total" (over all joints)
    """
    start_idx = step_indices[0]
    end_idx = step_indices[-1]
    torques = np.abs(joint_limit_torques[start_idx:end_idx, :])

    if joint_groups is None:
        joint_groups = {
            "hip": [0, 1],
            "knee": [2, 3],
            "ankle": [4, 5],
        }

    scale = abs(distance_traveled) * mass * GRAVITY * leg_length

    costs = {}
    all_indices = []
    for group_name, indices in joint_groups.items():
        costs[group_name] = float(dt * np.sum(torques[:, indices]) / (scale * len(indices)))
        all_indices.extend(indices)

    costs["total"] = float(dt * np.sum(torques[:, all_indices]) / (scale * len(all_indices)))
    return costs
