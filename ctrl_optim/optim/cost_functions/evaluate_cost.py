"""
Cost evaluation functions.

This module contains functions for evaluating the total cost
of a simulation based on various metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any

from .kinematic_costs import calculate_kinematic_costs
from . import effort_costs

# Muscle groups of the effort cost, by control mode: the group names of
# reflex_interface.muscles_dict (every group the reflex controller drives).
EFFORT_GROUPS = {
    "2D": ["HFL", "GLU", "HAM", "RF", "VAS", "BFSH", "GAS", "SOL", "TA", "FDL", "EDL"],
    "3D": ["HAB", "HAD", "HFL", "GLU", "HAM", "RF", "VAS", "BFSH", "GAS", "SOL", "TA", "FDL", "EDL"],
}

# Velocity tolerance of the constraint stage: the velocity error is relative to the target speed.
VEL_TOL = 0.01

# Terms of the final-stage cost of each optimization type. The final cost is the weighted sum
# sum_k w_k C_k of these terms, with the weights of --cost_weights (1 by default).
FINAL_TERMS = {
    "Effort": ("effort",),
    "Eff_Knee": ("effort", "joint_limit"),
    "Vel_grf": ("effort", "grf", "trunk"),
    "vel_musc": ("effort", "emg"),
    "vel_musc_grf": ("effort", "trunk", "grf", "emg"),
    "Classic": ("effort", "trunk", "joint_limit"),
    "Kine": ("effort", "kinematic", "trunk"),
    "Kine_grf": ("effort", "kinematic", "trunk", "grf"),
    "Kine_grf_musc": ("effort", "kinematic", "trunk", "grf", "emg"),
}
COST_TERMS = ("effort", "kinematic", "trunk", "grf", "emg", "joint_limit")
DEFAULT_COST_WEIGHTS = {term: 1.0 for term in COST_TERMS}


def resolve_cost_weights(cost_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Return the weights of all final-stage terms: the defaults (1) updated by cost_weights."""
    weights = dict(DEFAULT_COST_WEIGHTS)
    if cost_weights:
        unknown = set(cost_weights) - set(COST_TERMS)
        if unknown:
            raise ValueError(f"unknown cost terms {sorted(unknown)}; expected some of {COST_TERMS}")
        weights.update({k: float(v) for k, v in cost_weights.items()})
    return weights


def leg_length_from_frame(obj_out: Dict) -> float:
    """Leg length (m): hip-to-knee plus knee-to-ankle joint distance, mean of both legs."""
    lengths = []
    for leg in ("l_leg", "r_leg"):
        joint = obj_out[leg]["joint"]
        hip, knee, ankle = (np.asarray(joint[k]) for k in ("hip_pos", "knee_pos", "ankle_pos"))
        lengths.append(np.linalg.norm(hip - knee) + np.linalg.norm(knee - ankle))
    return float(np.mean(lengths))


def evaluateCost(
    data_store: List[Dict],
    step_size: float,
    eval_ctrl_mode: str,
    time_limit: int,
    target_slope: float,
    muscles_dict: Dict,
    optim_type: str,
    one_step: np.ndarray,
    one_EMG: np.ndarray,
    trunk_err_type: str,
    input_tgt_vel: float,
    stride_num: int,
    tgt_sym: float,
    tgt_grf: float,
    eval_mode: bool = False,
    cost_weights: Optional[Dict[str, float]] = None,
) -> Union[float, Dict[str, Any]]:
    """
    Evaluate the cost of a walking simulation.

    This function analyzes simulation data to calculate various cost metrics
    including effort, symmetry, velocity matching, kinematics, and more.

    Args:
        data_store: List of dictionaries containing simulation data
        step_size: Control period (s), the time between frames of data_store
        eval_ctrl_mode: Control mode ('2D' or '3D')
        time_limit: Maximum number of timesteps
        target_slope: Target slope in degrees
        muscles_dict: Dictionary of muscle information
        optim_type: Type of optimization (e.g., 'Effort', 'Velocity', etc.)
        one_step: Reference kinematic data for one step
        one_EMG: Reference EMG data
        trunk_err_type: Type of trunk error calculation
        input_tgt_vel: Target velocity
        stride_num: Number of strides to evaluate
        tgt_sym: Symmetry threshold, in leg lengths
        tgt_grf: GRF threshold, in body weights
        eval_mode: If True, return detailed cost breakdown
        cost_weights: Weights of the final-stage terms (see FINAL_TERMS); missing terms get 1

    Returns:
        Union[float, Dict[str, Any]]: Cost value or dictionary of cost components
    """
    # Early termination check
    if len(data_store) < (time_limit - 1):
        return calculate_early_cost(
            cost_const=data_store[0]["obj_func_out"]["const"],
            data_store=data_store,
            left_stance_foot=[],
            right_stance_foot=[],
            failure_mode=99,
        )

    # Extract basic data
    cost_const = data_store[0]["obj_func_out"]["const"]
    cost_mass = data_store[0]["obj_func_out"]["mass"]
    leg_length = leg_length_from_frame(data_store[0]["obj_func_out"])
    weights = resolve_cost_weights(cost_weights)

    # Initialize arrays
    unpacked_cost = np.zeros((len(data_store), 6))
    unpacked_angles = np.zeros((len(data_store), 7))
    unpacked_velocities = np.zeros((len(data_store), 3))
    unpacked_grf = np.zeros((len(data_store), 2))
    unpacked_act = np.zeros((len(data_store), data_store[0]["obj_func_out"]["mus_act"].shape[0]))
    # Joint-limit constraint torques (jointlimitfrc sensors): hip, knee, ankle, each left then right.
    unpacked_torque = np.zeros((len(data_store), 6))
    unpacked_euclidDist = np.zeros((len(data_store), 3))
    unpacked_spinal_phases_r = np.zeros((len(data_store), 11))
    unpacked_spinal_phases_l = np.zeros((len(data_store), 11))

    # Initialize 3D-specific arrays if needed
    if eval_ctrl_mode == "3D":
        unpacked_3d_angles = np.zeros((len(data_store), 4))  # [hip_add_r, hip_add_l, hip_rot_r, hip_rot_l]
        unpacked_pelvis = np.zeros((len(data_store), 2))  # [list, rotation]

    # Extract data from each timestep
    for idx, frame in enumerate(data_store):
        obj_out = frame["obj_func_out"]
        # Extract basic cost data
        unpacked_cost[idx, 0] = data_store[idx]["obj_func_out"]["GRF"]["r_leg"]
        unpacked_cost[idx, 1] = data_store[idx]["obj_func_out"]["GRF"]["l_leg"]
        unpacked_cost[idx, 2] = data_store[idx]["obj_func_out"]["sim_time"]
        unpacked_cost[idx, 3] = data_store[idx]["obj_func_out"]["new_step"]

        # Extract joint angles
        unpacked_angles[idx, 0] = data_store[idx]["obj_func_out"]["torso"]["pitch"]
        unpacked_angles[idx, 1] = data_store[idx]["obj_func_out"]["l_leg"]["joint"]["hip"]
        unpacked_angles[idx, 2] = data_store[idx]["obj_func_out"]["l_leg"]["joint"]["knee"]
        unpacked_angles[idx, 3] = data_store[idx]["obj_func_out"]["l_leg"]["joint"]["ankle"]
        unpacked_angles[idx, 4] = data_store[idx]["obj_func_out"]["r_leg"]["joint"]["hip"]
        unpacked_angles[idx, 5] = data_store[idx]["obj_func_out"]["r_leg"]["joint"]["knee"]
        unpacked_angles[idx, 6] = data_store[idx]["obj_func_out"]["r_leg"]["joint"]["ankle"]

        # Extract 3D-specific data if needed
        if eval_ctrl_mode == "3D":
            unpacked_3d_angles[idx, 0] = data_store[idx]["obj_func_out"]["r_leg"]["joint"]["hip_adduction"]
            unpacked_3d_angles[idx, 1] = data_store[idx]["obj_func_out"]["l_leg"]["joint"]["hip_adduction"]
            unpacked_3d_angles[idx, 2] = data_store[idx]["obj_func_out"]["r_leg"]["joint"]["hip_rotation"]
            unpacked_3d_angles[idx, 3] = data_store[idx]["obj_func_out"]["l_leg"]["joint"]["hip_rotation"]
            unpacked_pelvis[idx, 0] = data_store[idx]["obj_func_out"]["pelvis"]["list"]
            unpacked_pelvis[idx, 1] = data_store[idx]["obj_func_out"]["pelvis"]["rotation"]

        # Extract GRF and other data
        unpacked_grf[idx, 0] = data_store[idx]["obj_func_out"]["GRF"]["l_leg"]
        unpacked_grf[idx, 1] = data_store[idx]["obj_func_out"]["GRF"]["r_leg"]
        unpacked_act[idx, :] = data_store[idx]["obj_func_out"]["mus_act"]
        for col, joint in enumerate(("hip", "knee", "ankle")):
            unpacked_torque[idx, 2 * col] = obj_out["l_leg"]["joint"][f"{joint}_limit_sens"]
            unpacked_torque[idx, 2 * col + 1] = obj_out["r_leg"]["joint"][f"{joint}_limit_sens"]
        unpacked_euclidDist[idx, :] = data_store[idx]["obj_func_out"]["pelvis"]["x_pos"]
        unpacked_velocities[idx, 0] = data_store[idx]["obj_func_out"]["torso"]["dpitch"]

        unpacked_spinal_phases_r[idx, :] = np.array(list(obj_out["r_leg"]["spinal_control_phase"].values()))
        unpacked_spinal_phases_l[idx, :] = np.array(list(obj_out["l_leg"]["spinal_control_phase"].values()))

    # Find step indices and stance phases
    step_index = np.where(unpacked_cost[:, 3] == 1)[0]
    step_index = filter_step_indices(step_index, step_size)
    left_stance_foot, right_stance_foot = find_stance_feet(step_index.tolist(), unpacked_angles, unpacked_grf)

    # Check if we have enough strides
    if (
        not left_stance_foot
        or len(left_stance_foot) < stride_num + 1
        or not right_stance_foot
        or len(right_stance_foot) < stride_num + 1
    ):
        return calculate_early_cost(cost_const, data_store, left_stance_foot, right_stance_foot)

    # Use the last N strides for evaluation (matching original behavior)
    temp_index_vector = np.sort(np.concatenate([left_stance_foot, right_stance_foot]))

    firstStep = left_stance_foot[-(stride_num + 1)]
    lastStep = left_stance_foot[-1]

    startIdx_mask = np.where(temp_index_vector == firstStep)[0]
    endIdx_mask = np.where(temp_index_vector == lastStep)[0]

    # Check if steps were found in the combined vector
    if startIdx_mask.size == 0 or endIdx_mask.size == 0:
        return calculate_early_cost(cost_const, data_store, left_stance_foot, right_stance_foot, failure_mode=98)

    startIdx = startIdx_mask[0]
    endIdx = endIdx_mask[0]

    index_vector = temp_index_vector[startIdx : endIdx + 1]

    # Calculate basic costs
    sym_cost = calculate_symmetry_cost(
        left_stance_foot, right_stance_foot, stride_num, unpacked_angles, unpacked_grf, data_store, index_vector, leg_length
    )

    # If symmetry calculation failed, return early termination cost
    if sym_cost is None:
        return calculate_early_cost(
            cost_const=data_store[0]["obj_func_out"]["const"],
            data_store=data_store,
            left_stance_foot=left_stance_foot,
            right_stance_foot=right_stance_foot,
            failure_mode=98,  # Using 98 to indicate symmetry calculation failure
        )

    velocity_cost, steps_dist, steps_duration = calculate_velocity_cost(
        unpacked_euclidDist, unpacked_cost, index_vector, target_slope, input_tgt_vel
    )
    distance = np.linalg.norm(steps_dist)  # distance traveled in the window (m)

    GRF_cost = calculate_grf_cost(unpacked_grf, index_vector, tgt_grf, step_size, distance)

    # Joint-limit reliance per distance traveled, per joint group (both legs) and over all joints
    pain_costs = effort_costs.calculate_joint_limit_cost(
        unpacked_torque, index_vector, step_size, distance, cost_mass, leg_length
    )
    pain_cost = pain_costs["total"]
    knee_pain = pain_costs["knee"]
    hip_pain = pain_costs["hip"]
    ankle_pain = pain_costs["ankle"]

    # --- Effort and Profile Costs ---
    filter_musc = EFFORT_GROUPS["2D" if eval_ctrl_mode == "2D" else "3D"]

    # Extract muscle indices from the dictionary
    muscle_indices = []
    for leg in muscles_dict.values():
        for musc_group, indices in leg.items():
            if musc_group in filter_musc:
                muscle_indices.extend(indices)
    muscle_indices = sorted(list(set(muscle_indices)))

    # Calculate effort cost
    effort_cost = effort_costs.calculate_effort_cost(
        muscle_activations=unpacked_act,
        step_indices=index_vector,
        distance_traveled=distance,
        dt=step_size,
        muscle_indices=muscle_indices,
    )

    # Calculate EMG profile cost
    musc_profile_cost = effort_costs.calculate_emg_profile_cost(
        unpacked_act, one_EMG, muscles_dict, eval_ctrl_mode, index_vector
    )

    # Calculate kinematic costs (trunk_err_type selects the trunk term)
    kinematic_cost, trunk_cost = calculate_kinematic_costs(
        unpacked_angles, one_step, stride_num, index_vector, trunk_err_type, trunk_vel=unpacked_velocities[:, 0]
    )

    # Calculate 3D-specific costs if needed
    if eval_ctrl_mode == "3D":
        start_idx = index_vector[0]
        end_idx = index_vector[-1]

        # Hip adduction cost (neutral = -2 deg)
        add_diff = unpacked_3d_angles[start_idx : end_idx + 1, :2] - (-2 * np.pi / 180)
        add_cost = np.sum(10 * (-np.exp(-10 * np.square(add_diff)) + 1)) / (end_idx - start_idx)

        # Hip rotation cost (neutral = -28 deg)
        rot_diff = unpacked_3d_angles[start_idx : end_idx + 1, 2:] - (-28 * np.pi / 180)
        rot_cost = np.sum(10 * (-np.exp(-10 * np.square(rot_diff)) + 1)) / (end_idx - start_idx)

        # Pelvis orientation cost
        pelvis_cost = np.mean(np.abs(unpacked_pelvis[start_idx : end_idx + 1]))
    else:
        add_cost = 0
        rot_cost = 0
        pelvis_cost = 0

    # Final-stage terms and their weighted sum for the optimization type
    term_values = {
        "effort": effort_cost,
        "kinematic": kinematic_cost,
        "trunk": trunk_cost,
        "grf": GRF_cost,
        "emg": musc_profile_cost,
        "joint_limit": pain_cost,
    }
    final_cost = None
    if optim_type in FINAL_TERMS:
        final_cost = sum(weights[term] * term_values[term] for term in FINAL_TERMS[optim_type])

    # Constraint stage: velocity (relative to the target speed), symmetry (leg lengths), and,
    # except in the Vel_grf, vel_musc, and vel_musc_grf modes, pelvis orientation
    vel_bad = velocity_cost > VEL_TOL
    sym_bad = sym_cost > tgt_sym
    pelvis_bad = pelvis_cost > 20
    w_vel = 1000 if optim_type == "Classic" else 100
    constraint_cost = 10 * cost_const + w_vel * velocity_cost * vel_bad + 100 * sym_cost * sym_bad

    if optim_type in ("Effort", "Eff_Knee"):
        total_cost = final_cost
    elif optim_type in ("Vel_grf", "vel_musc", "vel_musc_grf"):
        total_cost = final_cost if not (vel_bad or sym_bad) else constraint_cost
    elif optim_type in ("Velocity", "Classic", "Kine", "Kine_grf", "Kine_grf_musc"):
        passed = not (vel_bad or sym_bad or pelvis_bad)
        constraint_cost += pelvis_cost * pelvis_bad
        if optim_type == "Velocity" or not passed:
            total_cost = constraint_cost
        else:
            total_cost = final_cost
    else:
        raise ValueError(f"unknown optim_type {optim_type!r}")

    if eval_mode:
        return create_cost_dictionary(
            eval_ctrl_mode=eval_ctrl_mode,
            effort_cost=effort_cost,
            sym_cost=sym_cost,
            velocity_cost=velocity_cost,
            GRF_cost=GRF_cost,
            pain_cost=pain_cost,
            knee_pain=knee_pain,
            hip_pain=hip_pain,
            ankle_pain=ankle_pain,
            kinematic_cost=kinematic_cost,
            trunk_cost=trunk_cost,
            steps_dist=steps_dist,
            steps_duration=steps_duration,
            musc_profile_cost=musc_profile_cost,
            add_cost=add_cost if eval_ctrl_mode == "3D" else None,
            rot_cost=rot_cost if eval_ctrl_mode == "3D" else None,
            pelvis_cost=pelvis_cost if eval_ctrl_mode == "3D" else None,
            leg_length=leg_length,
            final_cost=final_cost,
            total_cost=total_cost,
        )
    else:
        return total_cost


# Helper functions to extract and process simulation data
def extract_simulation_data(data_store):
    """Extract basic data arrays from simulation data"""
    unpacked_cost = np.zeros((len(data_store), 6))
    unpacked_angles = np.zeros((len(data_store), 7))
    unpacked_velocities = np.zeros((len(data_store), 3))
    unpacked_grf = np.zeros((len(data_store), 2))
    return unpacked_cost, unpacked_angles, unpacked_velocities, unpacked_grf


def filter_step_indices(step_index, step_size):
    """Remove steps that are too close together"""
    # Filter steps that are too close together (300ms)
    for smallSteps in np.arange(len(step_index)):
        if smallSteps >= len(step_index):
            break
        test_bool = (np.abs(step_index[smallSteps] - step_index) > (0.3 / step_size)) | (
            np.abs(step_index[smallSteps] - step_index) == 0
        )
        step_index = step_index[np.where(test_bool)[0]]
    return step_index


def find_stance_feet(step_index, unpacked_angles, unpacked_grf):
    """Identify stance feet based on ankle joint angles."""
    left_stance_foot = [idx for idx in step_index if unpacked_angles[idx, 3] < unpacked_angles[idx, 6]]
    right_stance_foot = [idx for idx in step_index if unpacked_angles[idx, 3] > unpacked_angles[idx, 6]]
    return left_stance_foot, right_stance_foot


def enough_strides(left_stance_foot, right_stance_foot, stride_num):
    """Check if there are enough strides for evaluation."""
    return (len(left_stance_foot) >= stride_num + 1) and (len(right_stance_foot) >= stride_num + 1)


def calculate_early_cost(
    cost_const: float, data_store: List[Dict], left_stance_foot: List[int], right_stance_foot: List[int], failure_mode: int = 99
) -> float:
    """Calculate cost for early termination cases."""
    total_cost = (
        failure_mode * cost_const
        - (data_store[len(data_store) - 1]["obj_func_out"]["pelvis_dist"][0])
        - (0.5 * (len(left_stance_foot) + len(right_stance_foot)))
    )
    return total_cost


def calculate_symmetry_cost(
    left_stance_foot, right_stance_foot, stride_num, unpacked_angles, unpacked_grf, data_store, index_vector, leg_length
):
    """Symmetry cost (leg lengths): distances of the left knee and ankle from the mirrored right knee
    and ankle (relative to the pelvis) at matched heel strikes, summed, divided by the number of
    strides and by the leg length."""
    # Get the evaluation window from index_vector
    start_idx = index_vector[0]
    end_idx = index_vector[-1]

    # Filter stance indices to only those within our evaluation window
    left_stance_idx = np.array([idx for idx in left_stance_foot if start_idx <= idx <= end_idx])
    right_stance_idx = np.array([idx for idx in right_stance_foot if start_idx <= idx <= end_idx])

    # Ensure equal number of left and right stances by truncating the longer array
    if len(left_stance_idx) > len(right_stance_idx):
        left_stance_idx = left_stance_idx[: len(right_stance_idx)]
    elif len(right_stance_idx) > len(left_stance_idx):
        right_stance_idx = right_stance_idx[: len(left_stance_idx)]

    # After equalizing, check if we still have enough strides
    if len(left_stance_idx) < stride_num or len(right_stance_idx) < stride_num:
        return None

    # Calculate joint position differences
    relative_pos = np.zeros((len(data_store), 3, 7))
    for idx in range(len(data_store)):
        relative_pos[idx, :, 0] = data_store[idx]["obj_func_out"]["pelvis"]["x_pos"]
        relative_pos[idx, :, 1] = data_store[idx]["obj_func_out"]["l_leg"]["joint"]["hip_pos"]
        relative_pos[idx, :, 2] = data_store[idx]["obj_func_out"]["l_leg"]["joint"]["knee_pos"]
        relative_pos[idx, :, 3] = data_store[idx]["obj_func_out"]["l_leg"]["joint"]["ankle_pos"]
        relative_pos[idx, :, 4] = data_store[idx]["obj_func_out"]["r_leg"]["joint"]["hip_pos"]
        relative_pos[idx, :, 5] = data_store[idx]["obj_func_out"]["r_leg"]["joint"]["knee_pos"]
        relative_pos[idx, :, 6] = data_store[idx]["obj_func_out"]["r_leg"]["joint"]["ankle_pos"]

    # Calculate symmetry costs for knee and ankle
    knee_diff = np.linalg.norm(
        np.array([1, -1, 1]) * (relative_pos[right_stance_idx, :, 5] - relative_pos[right_stance_idx, :, 0])
        - (relative_pos[left_stance_idx, :, 2] - relative_pos[left_stance_idx, :, 0]),
        axis=1,
    )

    ank_diff = np.linalg.norm(
        np.array([1, -1, 1]) * (relative_pos[right_stance_idx, :, 6] - relative_pos[right_stance_idx, :, 0])
        - (relative_pos[left_stance_idx, :, 3] - relative_pos[left_stance_idx, :, 0]),
        axis=1,
    )

    # Total symmetry cost (normalized by number of strides and leg length)
    sym_cost = np.sum(knee_diff + ank_diff) / (stride_num * leg_length)

    return sym_cost


def calculate_velocity_cost(unpacked_euclidDist, unpacked_cost, index_vector, target_slope, input_tgt_vel):
    """Velocity cost (dimensionless): norm of the mean horizontal velocity error in the window,
    divided by the target speed."""
    if input_tgt_vel <= 0:
        raise ValueError(f"the target velocity must be positive, got {input_tgt_vel}")
    if len(index_vector) < 2:
        return None, np.zeros(2), 0

    # Legacy behaviour: use the *last* `stride_num` strides detected (more stable after gait has settled)
    # First index is the (n+1)-th last left stance, last index is the very last left stance
    l_idx_start = index_vector[0]
    l_idx_end = index_vector[-1]

    # Calculate distances - use left_stance_foot indices directly
    start_pos = unpacked_euclidDist[l_idx_start, :]
    end_pos = unpacked_euclidDist[l_idx_end, :]

    # Calculate forward and lateral distances
    if target_slope != 0:
        # For sloped ground, consider x and z components
        forward_dist = np.linalg.norm(end_pos[[0, 2]] - start_pos[[0, 2]])
        lateral_dist = end_pos[1] - start_pos[1]
        steps_dist = np.array([forward_dist, lateral_dist])
    else:
        # For level ground, just use x and y
        steps_dist = end_pos[0:2] - start_pos[0:2]

    # Calculate duration using unpacked_cost array (legacy behaviour)
    steps_duration = unpacked_cost[l_idx_end, 2] - unpacked_cost[l_idx_start, 2]

    # Calculate velocity
    steps_vel = steps_dist / steps_duration

    # Calculate velocity cost
    v_tgt = np.array([input_tgt_vel, 0])  # Target velocity (forward only)
    velocity_cost = np.linalg.norm(steps_vel - v_tgt) / input_tgt_vel

    return velocity_cost, steps_dist, steps_duration


def calculate_grf_cost(unpacked_grf, index_vector, tgt_grf, dt, distance_traveled):
    """GRF limit (s/m): time integral of the excess of the total vertical GRF (both feet, in body
    weights) over the threshold tgt_grf, divided by the distance traveled in the window."""
    if len(index_vector) < 2:
        return None

    l_idx_start = index_vector[0]
    l_idx_end = index_vector[-1]

    # Total GRF (body weights), left plus right
    total_GRF = np.sum(unpacked_grf[l_idx_start:l_idx_end, :], axis=1)

    excess = np.clip(total_GRF - tgt_grf, 0.0, None)
    return float(dt * np.sum(excess) / abs(distance_traveled))


def create_cost_dictionary(
    eval_ctrl_mode,
    effort_cost,
    sym_cost,
    velocity_cost,
    GRF_cost,
    pain_cost,
    knee_pain,
    hip_pain,
    ankle_pain,
    kinematic_cost,
    trunk_cost,
    steps_dist,
    steps_duration,
    musc_profile_cost,
    add_cost=None,
    rot_cost=None,
    pelvis_cost=None,
    leg_length=None,
    final_cost=None,
    total_cost=None,
):
    """Create detailed cost dictionary for evaluation mode (units in the comments)"""
    out = {
        "Total_Cost": total_cost,  # value returned to the optimizer
        "Final_Cost": final_cost,  # weighted final-stage sum, whether or not the constraints pass
        "Effort_Cost": effort_cost,  # s/m
        "Symmetry_Cost": sym_cost,  # leg lengths
        "Velocity_Cost": velocity_cost,  # fraction of the target speed
        "GRF_Cost": GRF_cost,  # s/m
        "JointPain_Cost": pain_cost,  # s/m, mean over all joints
        "HipPain_Cost": hip_pain,
        "KneePain_Cost": knee_pain,
        "AnklePain_Cost": ankle_pain,
        "Kinematic_Cost": kinematic_cost,  # rad
        "Trunk_Cost": trunk_cost,
        "EMG_Cost": musc_profile_cost,
    }
    if eval_ctrl_mode == "3D":
        out["Hip Rotation Cost"] = rot_cost
        out["Hip Adduction Cost"] = add_cost
    out.update(
        {
            "Actual_velocity": steps_dist / steps_duration,
            "Actual_dist": steps_dist,
            "Actual_time": steps_duration,
            "Pelvis_Cost": pelvis_cost,
            "Leg_length": leg_length,
        }
    )
    return out


def calculate_scruff_cost(
    left_stance_foot, right_stance_foot, stride_num, unpacked_grf, unpacked_spinal_phases_l, unpacked_spinal_phases_r
):
    """Calculate cost to prevent foot scuffing during swing phase"""
    left_scruff_idx = left_stance_foot[-(stride_num + 1) : len(left_stance_foot)]
    right_scruff_idx = right_stance_foot[-(stride_num + 1) : len(right_stance_foot)]

    scruff_cost = 0
    for idx in range(len(left_scruff_idx) - 1):
        # Get spinal phases for swing detection - phase 4 in legacy code is now phase 10 (swing phase)
        scruff_phase_l = unpacked_spinal_phases_l[left_scruff_idx[idx] : left_scruff_idx[idx + 1], 10]
        scruff_phase_r = unpacked_spinal_phases_r[right_scruff_idx[idx] : right_scruff_idx[idx + 1], 10]

        swing_start_l = np.array([value for value in np.where(scruff_phase_l == 1)[0] if value >= 5])
        swing_start_r = np.array([value for value in np.where(scruff_phase_r == 1)[0] if value >= 5])

        if len(swing_start_l) == 0 or len(swing_start_r) == 0:
            return None  # Signal to use early termination cost

        left_swing_start = left_scruff_idx[idx] + swing_start_l[0]
        right_swing_start = right_scruff_idx[idx] + swing_start_r[0]

        scruff_cost += np.sum(unpacked_grf[left_swing_start : left_scruff_idx[idx + 1], 0]) + np.sum(
            unpacked_grf[right_swing_start : right_scruff_idx[idx + 1], 1]
        )

    return scruff_cost
