"""
Cost evaluation functions.

This module contains functions for evaluating the total cost
of a simulation based on various metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional

from .kinematic_costs import calculate_kinematic_costs, calculate_trunk_cost
from . import effort_costs


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
    muslen_param: np.ndarray, 
    eval_mode: bool = False
) -> Union[float, Dict[str, Any]]:
    """
    Evaluate the cost of a walking simulation.
    
    This function analyzes simulation data to calculate various cost metrics
    including effort, symmetry, velocity matching, kinematics, and more.
    
    Args:
        data_store: List of dictionaries containing simulation data
        step_size: Time step size of the simulation
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
        tgt_sym: Target symmetry threshold
        tgt_grf: Target ground reaction force threshold
        muslen_param: Muscle length parameters
        eval_mode: If True, return detailed cost breakdown
        
    Returns:
        Union[float, Dict[str, Any]]: Cost value or dictionary of cost components
    """
    # Early termination check
    if len(data_store) < (time_limit-1):
        return calculate_early_cost(
            cost_const=data_store[0]['obj_func_out']['const'],
            data_store=data_store,
            left_stance_foot=[],
            right_stance_foot=[],
            failure_mode=99
        )

    # Extract basic data
    cost_const = data_store[0]['obj_func_out']['const']
    cost_mass = data_store[0]['obj_func_out']['mass']
    const_theta_tgt = data_store[0]['obj_func_out']['pelvis']['theta_tgt']
    
    # Initialize arrays
    unpacked_cost = np.zeros((len(data_store), 6))
    unpacked_angles = np.zeros((len(data_store), 7))
    unpacked_velocities = np.zeros((len(data_store), 3))
    unpacked_grf = np.zeros((len(data_store), 2))
    unpacked_act = np.zeros((len(data_store), data_store[0]['obj_func_out']['mus_act'].shape[0]))
    unpacked_torque = np.zeros((len(data_store), 8))
    unpacked_euclidDist = np.zeros((len(data_store), 3))
    unpacked_spinal_phases_r = np.zeros((len(data_store), 11))
    unpacked_spinal_phases_l = np.zeros((len(data_store), 11))
    
    # Initialize 3D-specific arrays if needed
    if eval_ctrl_mode == '3D':
        unpacked_3d_angles = np.zeros((len(data_store), 4))  # [hip_add_r, hip_add_l, hip_rot_r, hip_rot_l]
        unpacked_pelvis = np.zeros((len(data_store), 2))     # [list, rotation]

    # Extract data from each timestep
    for idx, frame in enumerate(data_store):
        obj_out = frame['obj_func_out']
        # Extract basic cost data
        unpacked_cost[idx, 0] = data_store[idx]['obj_func_out']['GRF']['r_leg']
        unpacked_cost[idx, 1] = data_store[idx]['obj_func_out']['GRF']['l_leg']
        unpacked_cost[idx, 2] = data_store[idx]['obj_func_out']['sim_time']
        unpacked_cost[idx, 3] = data_store[idx]['obj_func_out']['new_step']
        
        # Extract joint angles
        unpacked_angles[idx, 0] = data_store[idx]['obj_func_out']['torso']['pitch']
        unpacked_angles[idx, 1] = data_store[idx]['obj_func_out']['l_leg']['joint']['hip']
        unpacked_angles[idx, 2] = data_store[idx]['obj_func_out']['l_leg']['joint']['knee']
        unpacked_angles[idx, 3] = data_store[idx]['obj_func_out']['l_leg']['joint']['ankle']
        unpacked_angles[idx, 4] = data_store[idx]['obj_func_out']['r_leg']['joint']['hip']
        unpacked_angles[idx, 5] = data_store[idx]['obj_func_out']['r_leg']['joint']['knee']
        unpacked_angles[idx, 6] = data_store[idx]['obj_func_out']['r_leg']['joint']['ankle']
        
        # Extract 3D-specific data if needed
        if eval_ctrl_mode == '3D':
            unpacked_3d_angles[idx, 0] = data_store[idx]['obj_func_out']['r_leg']['joint']['hip_adduction']
            unpacked_3d_angles[idx, 1] = data_store[idx]['obj_func_out']['l_leg']['joint']['hip_adduction']
            unpacked_3d_angles[idx, 2] = data_store[idx]['obj_func_out']['r_leg']['joint']['hip_rotation']
            unpacked_3d_angles[idx, 3] = data_store[idx]['obj_func_out']['l_leg']['joint']['hip_rotation']
            unpacked_pelvis[idx, 0] = data_store[idx]['obj_func_out']['pelvis']['list']
            unpacked_pelvis[idx, 1] = data_store[idx]['obj_func_out']['pelvis']['rotation']
        
        # Extract GRF and other data
        unpacked_grf[idx, 0] = data_store[idx]['obj_func_out']['GRF']['l_leg']
        unpacked_grf[idx, 1] = data_store[idx]['obj_func_out']['GRF']['r_leg']
        unpacked_act[idx, :] = data_store[idx]['obj_func_out']['mus_act']
        unpacked_euclidDist[idx, :] = data_store[idx]['obj_func_out']['pelvis']['x_pos']
        unpacked_velocities[idx, 0] = data_store[idx]['obj_func_out']['torso']['dpitch']

        unpacked_spinal_phases_r[idx, :] = np.array(list(obj_out['r_leg']['spinal_control_phase'].values()))
        unpacked_spinal_phases_l[idx, :] = np.array(list(obj_out['l_leg']['spinal_control_phase'].values()))
    
    # Find step indices and stance phases
    step_index = np.where(unpacked_cost[:,3] == 1)[0]
    step_index = filter_step_indices(step_index, step_size)
    left_stance_foot, right_stance_foot = find_stance_feet(step_index.tolist(), unpacked_angles, unpacked_grf)
    
    # Check if we have enough strides
    if not left_stance_foot or len(left_stance_foot) < stride_num + 1 or \
       not right_stance_foot or len(right_stance_foot) < stride_num + 1:
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
    
    index_vector = temp_index_vector[startIdx:endIdx+1]

    # Calculate basic costs
    sym_cost = calculate_symmetry_cost(left_stance_foot, right_stance_foot, stride_num, 
                                 unpacked_angles, unpacked_grf, data_store, index_vector)
    
    # If symmetry calculation failed, return early termination cost
    if sym_cost is None:
        return calculate_early_cost(
            cost_const=data_store[0]['obj_func_out']['const'],
            data_store=data_store,
            left_stance_foot=left_stance_foot,
            right_stance_foot=right_stance_foot,
            failure_mode=98  # Using 98 to indicate symmetry calculation failure
        )

    velocity_cost, steps_dist, steps_duration = calculate_velocity_cost(unpacked_euclidDist, unpacked_cost,
                                                                      index_vector, target_slope,
                                                                      input_tgt_vel)
    
    GRF_cost = calculate_grf_cost(unpacked_grf, index_vector, tgt_grf)

    scruff_cost = calculate_scruff_cost(left_stance_foot, right_stance_foot, stride_num, 
                                        unpacked_grf, unpacked_spinal_phases_l, unpacked_spinal_phases_r)
    
    # Calculate joint pain costs
    pain_costs = effort_costs.calculate_joint_limit_cost(unpacked_torque, index_vector)
    pain_cost = pain_costs['total']
    knee_pain = pain_costs['knee']
    hip_pain = pain_costs['hip']
    ankle_pain = pain_costs['ankle']

    # --- Effort and Profile Costs ---
    # Define muscles to use based on control mode
    if eval_ctrl_mode == '2D':
        filter_musc = ['HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA', 'TOFL', 'TOEX']
    else:
        filter_musc = ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA', 'TOFL', 'TOEX']
    
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
        distance_traveled=np.linalg.norm(steps_dist), 
        mass=cost_mass, 
        muscle_indices=muscle_indices
    )

    # Calculate EMG profile cost
    musc_profile_cost = effort_costs.calculate_emg_profile_cost(
        unpacked_act, 
        one_EMG, 
        muscles_dict, 
        eval_ctrl_mode, 
        index_vector, 
        stride_num
    )

    # Calculate kinematic costs
    kinematic_cost, trunk_cost = calculate_kinematic_costs(unpacked_angles, one_step, stride_num, index_vector)
    
    # Calculate 3D-specific costs if needed
    if eval_ctrl_mode == '3D':
        start_idx = index_vector[0]
        end_idx = index_vector[-1]
        
        # Hip adduction cost (neutral = -2 deg)
        add_diff = unpacked_3d_angles[start_idx:end_idx+1, :2] - (-2*np.pi/180)
        add_cost = np.sum(10*(-np.exp(-10*np.square(add_diff))+1)) / (end_idx - start_idx)
        
        # Hip rotation cost (neutral = -28 deg)
        rot_diff = unpacked_3d_angles[start_idx:end_idx+1, 2:] - (-28*np.pi/180)
        rot_cost = np.sum(10*(-np.exp(-10*np.square(rot_diff))+1)) / (end_idx - start_idx)
        
        # Pelvis orientation cost
        pelvis_cost = np.mean(np.abs(unpacked_pelvis[start_idx:end_idx+1]))
    else:
        add_cost = 0
        rot_cost = 0
        pelvis_cost = 0
    
    # Calculate total cost based on optimization type
    pass_flag = False
    if optim_type == 'Effort':
        pass_flag = True
        total_cost = effort_cost
    elif optim_type == 'Eff_Knee':
        pass_flag = pain_cost < 0.01
        total_cost = effort_cost + pain_cost
    elif optim_type == 'Velocity':
        pass_flag = velocity_cost < 0.01 and sym_cost < tgt_sym and pelvis_cost < 20
        total_cost = 10*cost_const + (100*velocity_cost * (velocity_cost > 0.01)) + (100*sym_cost*(sym_cost > tgt_sym)) + (pelvis_cost*(pelvis_cost > 20))
    elif optim_type in ['Vel_grf', 'vel_musc', 'vel_musc_grf']:
        pass_flag = velocity_cost < 0.01 and sym_cost < tgt_sym
        total_cost = 10*cost_const + (100*velocity_cost * (velocity_cost > 0.01)) + (100*sym_cost*(sym_cost > tgt_sym))
        if pass_flag:
            if optim_type == 'Vel_grf':
                total_cost = effort_cost + GRF_cost + trunk_cost
            elif optim_type == 'vel_musc':
                total_cost = effort_cost + musc_profile_cost
            elif optim_type == 'vel_musc_grf':
                total_cost = effort_cost + trunk_cost + GRF_cost + musc_profile_cost
    elif optim_type == 'Classic':
        pass_flag = velocity_cost < 0.01 and sym_cost < tgt_sym and pelvis_cost < 20
        total_cost = 10*cost_const + (1000*velocity_cost * (velocity_cost > 0.01)) + (100*sym_cost*(sym_cost > tgt_sym)) + (pelvis_cost*(pelvis_cost > 20))
        if pass_flag:
            total_cost = effort_cost + trunk_cost + knee_pain + hip_pain + ankle_pain
    elif optim_type in ['Kine', 'Kine_grf', 'Kine_grf_musc']:
        pass_flag = velocity_cost < 0.01 and sym_cost < tgt_sym and pelvis_cost < 20
        total_cost = 10*cost_const + (100*velocity_cost * (velocity_cost > 0.01)) + (100*sym_cost*(sym_cost > tgt_sym)) + (pelvis_cost*(pelvis_cost > 20))
        if pass_flag:
            if optim_type == 'Kine':
                total_cost = effort_cost + kinematic_cost + trunk_cost
            elif optim_type == 'Kine_grf':
                total_cost = effort_cost + kinematic_cost + trunk_cost + GRF_cost
            elif optim_type == 'Kine_grf_musc':
                total_cost = effort_cost + kinematic_cost + trunk_cost + GRF_cost + musc_profile_cost

    # Add muscle length parameter cost if provided
    if len(muslen_param) > 0:
        total_cost += np.sum(np.sqrt(np.square(muslen_param - 1)))

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
            add_cost=add_cost if eval_ctrl_mode == '3D' else None,
            rot_cost=rot_cost if eval_ctrl_mode == '3D' else None,
            pelvis_cost=pelvis_cost if eval_ctrl_mode == '3D' else None
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


def extract_basic_data(data_store, idx, unpacked_cost, unpacked_angles, unpacked_velocities, 
                      unpacked_grf, unpacked_act, unpacked_torque, unpacked_euclidDist):
    """Extract data for a single timestep"""
    if 'obj_func_out' not in data_store[idx]:
        return
        
    # Extract basic cost data
    unpacked_cost[idx, 0] = np.abs(data_store[idx]['obj_func_out']['pelvis_dist'][0])
    unpacked_cost[idx, 1] = np.abs(data_store[idx]['obj_func_out']['pelvis_dist'][1])
    unpacked_cost[idx, 2] = data_store[idx]['obj_func_out']['sim_time']
    unpacked_cost[idx, 3] = data_store[idx]['obj_func_out']['new_step']
    
    # Extract position data
    unpacked_euclidDist[idx, 0] = data_store[idx]['obj_func_out']['pelvis']['x_pos'][0]
    unpacked_euclidDist[idx, 1] = data_store[idx]['obj_func_out']['pelvis']['x_pos'][1]
    unpacked_euclidDist[idx, 2] = data_store[idx]['obj_func_out']['pelvis']['x_pos'][2]

    # Extract muscle activations
    unpacked_act[idx, :] = data_store[idx]['obj_func_out']['mus_act']

    # Extract joint angles
    unpacked_angles[idx, 0] = data_store[idx]['obj_func_out']['torso']['pitch']
    unpacked_angles[idx, 1] = data_store[idx]['obj_func_out']['l_leg']['joint']['hip']
    unpacked_angles[idx, 2] = data_store[idx]['obj_func_out']['l_leg']['joint']['knee']
    unpacked_angles[idx, 3] = data_store[idx]['obj_func_out']['l_leg']['joint']['ankle']
    unpacked_angles[idx, 4] = data_store[idx]['obj_func_out']['r_leg']['joint']['hip']
    unpacked_angles[idx, 5] = data_store[idx]['obj_func_out']['r_leg']['joint']['knee']
    unpacked_angles[idx, 6] = data_store[idx]['obj_func_out']['r_leg']['joint']['ankle']

    try:
        # Extract joint velocities - store in unpacked_velocities[:, 3:7]
        # Left leg velocities
        unpacked_velocities[idx, 3] = data_store[idx]['obj_func_out']['l_leg']['joint']['ankle_vel']
        unpacked_velocities[idx, 4] = data_store[idx]['obj_func_out']['l_leg']['joint']['mtp_vel']
        # Right leg velocities
        unpacked_velocities[idx, 5] = data_store[idx]['obj_func_out']['r_leg']['joint']['ankle_vel']
        unpacked_velocities[idx, 6] = data_store[idx]['obj_func_out']['r_leg']['joint']['mtp_vel']
    except KeyError as e:
        print(f"Warning: Unable to access joint velocity data: {e}")
        print("Joint velocity cost will not be included in kinematic cost calculation")
        unpacked_velocities[idx, 3:7] = 0

    # Extract joint torques
    unpacked_torque[idx, 0] = data_store[idx]['obj_func_out']['l_leg']['joint']['knee_torque']
    unpacked_torque[idx, 1] = data_store[idx]['obj_func_out']['r_leg']['joint']['knee_torque']
    unpacked_torque[idx, 2] = data_store[idx]['obj_func_out']['l_leg']['joint']['knee_limit_sens']
    unpacked_torque[idx, 3] = data_store[idx]['obj_func_out']['r_leg']['joint']['knee_limit_sens']
    unpacked_torque[idx, 4] = data_store[idx]['obj_func_out']['l_leg']['joint']['hip_limit_sens']
    unpacked_torque[idx, 5] = data_store[idx]['obj_func_out']['r_leg']['joint']['hip_limit_sens']
    unpacked_torque[idx, 6] = data_store[idx]['obj_func_out']['l_leg']['joint']['ankle_limit_sens']
    unpacked_torque[idx, 7] = data_store[idx]['obj_func_out']['r_leg']['joint']['ankle_limit_sens']

    # Extract pelvis velocities
    pelvis_vel = data_store[idx]['obj_func_out']['pelvis']['vel']  # This is a 2D array [forward, lateral]
    unpacked_velocities[idx, 0] = pelvis_vel[0]  # Forward velocity (x)
    unpacked_velocities[idx, 1] = pelvis_vel[1]  # Lateral velocity (y)
    unpacked_velocities[idx, 2] = 0  # No z velocity data available, set to 0

    # Extract ground reaction forces
    unpacked_grf[idx, 0] = data_store[idx]['obj_func_out']['GRF']['l_leg']
    unpacked_grf[idx, 1] = data_store[idx]['obj_func_out']['GRF']['r_leg']


def filter_step_indices(step_index, step_size):
    """Remove steps that are too close together"""
    # Filter steps that are too close together (300ms)
    for smallSteps in np.arange(len(step_index)):
        if smallSteps >= len(step_index):
            break
        test_bool = (np.abs(step_index[smallSteps] - step_index) > (0.3 / step_size)) | (np.abs(step_index[smallSteps] - step_index) == 0)
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


def calculate_early_cost(cost_const: float, data_store: List[Dict], left_stance_foot: List[int], right_stance_foot: List[int], failure_mode: int = 99) -> float:
    """Calculate cost for early termination cases."""
    total_cost = (
        failure_mode * cost_const - 
        (data_store[len(data_store)-1]['obj_func_out']['pelvis_dist'][0]) - 
        (0.5 * (len(left_stance_foot) + len(right_stance_foot)))
    )
    return total_cost


def calculate_symmetry_cost(left_stance_foot, right_stance_foot, stride_num, 
                           unpacked_angles, unpacked_grf, data_store, index_vector):
    """Calculate symmetry cost between left and right legs for the specified stride window"""
    # Get the evaluation window from index_vector
    start_idx = index_vector[0]
    end_idx = index_vector[-1]
    
    # Filter stance indices to only those within our evaluation window
    left_stance_idx = np.array([idx for idx in left_stance_foot if start_idx <= idx <= end_idx])
    right_stance_idx = np.array([idx for idx in right_stance_foot if start_idx <= idx <= end_idx])
    
    # Ensure equal number of left and right stances by truncating the longer array
    if len(left_stance_idx) > len(right_stance_idx):
        left_stance_idx = left_stance_idx[:len(right_stance_idx)]
    elif len(right_stance_idx) > len(left_stance_idx):
        right_stance_idx = right_stance_idx[:len(left_stance_idx)]
        
    # After equalizing, check if we still have enough strides
    if len(left_stance_idx) < stride_num or len(right_stance_idx) < stride_num:
        return None
    
    # Calculate joint position differences
    relative_pos = np.zeros((len(data_store), 3, 7))
    for idx in range(len(data_store)):
        relative_pos[idx,:,0] = data_store[idx]['obj_func_out']['pelvis']['x_pos']
        relative_pos[idx,:,1] = data_store[idx]['obj_func_out']['l_leg']['joint']['hip_pos']
        relative_pos[idx,:,2] = data_store[idx]['obj_func_out']['l_leg']['joint']['knee_pos']
        relative_pos[idx,:,3] = data_store[idx]['obj_func_out']['l_leg']['joint']['ankle_pos']
        relative_pos[idx,:,4] = data_store[idx]['obj_func_out']['r_leg']['joint']['hip_pos']
        relative_pos[idx,:,5] = data_store[idx]['obj_func_out']['r_leg']['joint']['knee_pos']
        relative_pos[idx,:,6] = data_store[idx]['obj_func_out']['r_leg']['joint']['ankle_pos']
    
    # Calculate symmetry costs for knee and ankle
    knee_diff = np.linalg.norm(np.array([1,-1,1])*(relative_pos[right_stance_idx,:,5] - relative_pos[right_stance_idx,:,0]) - 
                (relative_pos[left_stance_idx,:,2] - relative_pos[left_stance_idx,:,0]), axis=1)
                
    ank_diff = np.linalg.norm(np.array([1,-1,1])*(relative_pos[right_stance_idx,:,6] - relative_pos[right_stance_idx,:,0]) - 
                (relative_pos[left_stance_idx,:,3] - relative_pos[left_stance_idx,:,0]), axis=1)
    
    # Total symmetry cost (normalized by number of strides)
    sym_cost = np.sum(knee_diff + ank_diff) / stride_num
    
    return sym_cost


def calculate_velocity_cost(unpacked_euclidDist, unpacked_cost, index_vector, target_slope, input_tgt_vel):
    """Calculate velocity matching cost"""
    if len(index_vector) < 2:
        return None, np.zeros(2), 0
        
    # Legacy behaviour: use the *last* `stride_num` strides detected (more stable after gait has settled)
    # First index is the (n+1)-th last left stance, last index is the very last left stance
    l_idx_start = index_vector[0]
    l_idx_end   = index_vector[-1]
    
    # Calculate distances - use left_stance_foot indices directly
    start_pos = unpacked_euclidDist[l_idx_start, :]
    end_pos = unpacked_euclidDist[l_idx_end, :]
    
    # Calculate forward and lateral distances
    if target_slope != 0:
        # For sloped ground, consider x and z components
        forward_dist = np.linalg.norm(end_pos[[0,2]] - start_pos[[0,2]])
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
    velocity_cost = np.linalg.norm(steps_vel - v_tgt)
    
    return velocity_cost, steps_dist, steps_duration


def calculate_grf_cost(unpacked_grf, index_vector, tgt_grf):
    """Calculate ground reaction force cost"""
    if len(index_vector) < 2:
        return None
        
    # --- Legacy stride window (last `stride_num` strides) ---
    l_idx_start = index_vector[0]
    l_idx_end   = index_vector[-1]
    
    # Calculate total GRF by summing left and right forces
    total_GRF = np.sum(unpacked_grf[l_idx_start:l_idx_end,:], axis=1)
    
    # Calculate cost for GRF exceeding target threshold
    GRF_cost = np.sum(total_GRF[total_GRF > tgt_grf]) / (l_idx_end - l_idx_start)
    
    return GRF_cost


def calculate_joint_pain(unpacked_angles, index_vector, left_idx, right_idx):
    """Calculate joint pain cost for a pair of joints."""
    left_pain = np.sum(np.abs(unpacked_angles[index_vector[0]:index_vector[-1], left_idx]))
    right_pain = np.sum(np.abs(unpacked_angles[index_vector[0]:index_vector[-1], right_idx]))
    return left_pain + right_pain


def calculate_effort_and_emg_cost(data_store, step_index, left_stance_foot, stride_num, 
                                  unpacked_act, muscles_dict, cost_mass, steps_dist, one_EMG, eval_ctrl_mode,
                                  index_vector):
    """Calculate effort cost and EMG profile matching cost"""
    if len(left_stance_foot) < stride_num:
        return None, None
    
    # Get muscle indices based on control mode
    if eval_ctrl_mode == '2D':
        filter_musc = ['HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA', 'TOFL', 'TOEX']
    else:
        filter_musc = ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA', 'TOFL', 'TOEX']
    
    # Extract muscle indices from both legs
    muscle_indices = []
    for musc in filter_musc:
        if musc in muscles_dict['l_leg']:
            muscle_indices.extend(muscles_dict['l_leg'][musc])
        if musc in muscles_dict['r_leg']:
            muscle_indices.extend(muscles_dict['r_leg'][musc])
    muscle_indices = sorted(list(set(muscle_indices)))  # Remove duplicates and sort
    
    # Calculate effort cost
    effort_cost = effort_costs.calculate_effort_cost(
        muscle_activations=unpacked_act,
        step_indices=index_vector,
        distance_traveled=steps_dist[0],  # Use forward distance only
        mass=cost_mass,
        muscle_indices=muscle_indices
    )
    
    # Calculate EMG profile cost
    if one_EMG is not None and one_EMG.size > 0:
        musc_profile_cost = effort_costs.calculate_emg_profile_cost(
            muscle_activations=unpacked_act,
            ref_emg=one_EMG,
            muscles_dict=muscles_dict,
            eval_ctrl_mode=eval_ctrl_mode,
            step_indices=index_vector,
            n_strides=stride_num
        )
    else:
        musc_profile_cost = 0
    
    return effort_cost, musc_profile_cost


def check_optimization_constraints(optim_type, velocity_cost, sym_cost, tgt_sym, 
                                 GRF_cost, knee_pain, hip_pain, ankle_pain, pelvis_cost, cost_const, scruff_cost=None):
    """Check constraints based on optimization type"""
    pass_flag = False
    pass_cost = 0
    
    if optim_type == 'Effort':
        pass_flag = True
        pass_cost = effort_cost
    elif optim_type == 'Eff_Knee':
        pass_flag = pain_cost < 0.01
        pass_cost = effort_cost + pain_cost
    elif optim_type == 'Velocity':
        pass_flag = velocity_cost < 0.01 and sym_cost < tgt_sym and pelvis_cost < 20
        pass_cost = 10*cost_const + (100*velocity_cost * (not velocity_cost < 0.01)) + (100*sym_cost*(not sym_cost < tgt_sym)) + (pelvis_cost*(not pelvis_cost < 20))
    elif optim_type in ['Vel_grf', 'vel_musc', 'vel_musc_grf']:
        pass_flag = velocity_cost < 0.01 and sym_cost < tgt_sym
        pass_cost = 10*cost_const + (100*velocity_cost * (not velocity_cost < 0.01)) + (100*sym_cost*(not sym_cost < tgt_sym))
    elif optim_type == 'Classic':
        pass_flag = velocity_cost < 0.01 and sym_cost < tgt_sym and pelvis_cost < 20
        pass_cost = 10*cost_const + (1000*velocity_cost * (not velocity_cost < 0.01)) + (100*sym_cost*(not sym_cost < tgt_sym)) + (pelvis_cost*(not pelvis_cost < 20))
    elif optim_type in ['Kine', 'Kine_grf', 'Kine_grf_musc', 'Monolithic_Kine']:
        pass_flag = velocity_cost < 0.01 and sym_cost < tgt_sym and pelvis_cost < 20
        pass_cost = 10*cost_const + (1000*velocity_cost * (not velocity_cost < 0.01)) + (100*sym_cost*(not sym_cost < tgt_sym)) + (pelvis_cost*(not pelvis_cost < 20))
    
    return pass_flag, pass_cost


def calculate_final_cost(pass_flag, pass_cost, optim_type, effort_cost, kinematic_cost, 
                       trunk_cost, GRF_cost, musc_profile_cost, pain_cost, muslen_param):
    """Calculate final cost based on optimization type"""
    if not pass_flag:
        return pass_cost
        
    total_cost = pass_cost
    
    if optim_type == 'Vel_grf':
        total_cost = effort_cost + GRF_cost + trunk_cost
    elif optim_type == 'Classic':
        total_cost = effort_cost + trunk_cost + pain_cost
    elif optim_type == 'Kine':
        total_cost = effort_cost + kinematic_cost + trunk_cost
    elif optim_type == 'Kine_grf':
        total_cost = effort_cost + kinematic_cost + trunk_cost + GRF_cost
    elif optim_type == 'Kine_grf_musc':
        total_cost = effort_cost + kinematic_cost + trunk_cost + GRF_cost + musc_profile_cost
    elif optim_type == 'vel_musc':
        total_cost = effort_cost + musc_profile_cost
    elif optim_type == 'vel_musc_grf':
        total_cost = effort_cost + trunk_cost + GRF_cost + musc_profile_cost
    else:
        total_cost = effort_cost
        
    # Add muscle length parameter cost if provided
    if len(muslen_param) > 0:
        total_cost += np.sum(np.sqrt(np.square(muslen_param - 1)))
        
    return total_cost


def create_cost_dictionary(eval_ctrl_mode, effort_cost, sym_cost, velocity_cost, 
                          GRF_cost, pain_cost, knee_pain, hip_pain, ankle_pain, 
                          kinematic_cost, trunk_cost, steps_dist, steps_duration, 
                          musc_profile_cost, add_cost=None, rot_cost=None, pelvis_cost=None):
    """Create detailed cost dictionary for evaluation mode"""
    if eval_ctrl_mode == '3D':
        return {
            'Effort_Cost': effort_cost,
            'Symmetry_Cost': sym_cost,
            'Velocity_Cost': velocity_cost,
            'GRF_Cost': GRF_cost,
            'JointPain_Cost': pain_cost,
            'HipPain_Cost': hip_pain,
            'KneePain_Cost': knee_pain,
            'AnklePain_Cost': ankle_pain,
            'Kinematic_Cost': kinematic_cost + trunk_cost,
            'Hip Rotation Cost': rot_cost,
            'Hip Adduction Cost': add_cost,
            'Comb_Eff_Vel_Pain_Scruff': effort_cost + 1000*velocity_cost + trunk_cost + pain_cost,
            'Comb_Kine_Vel_Pain_Scruff': kinematic_cost + 1000*velocity_cost + trunk_cost + pain_cost,
            'Actual_velocity': steps_dist / steps_duration,
            'Actual_dist': steps_dist,
            'Actual_time': steps_duration,
            'Pelvis_Cost': pelvis_cost
        }
    else:
        return {
            'Effort_Cost': effort_cost,
            'Symmetry_Cost': sym_cost,
            'Velocity_Cost': velocity_cost,
            'GRF_Cost': GRF_cost,
            'JointPain_Cost': pain_cost,
            'HipPain_Cost': hip_pain,
            'KneePain_Cost': knee_pain,
            'AnklePain_Cost': ankle_pain,
            'Kinematic_Cost': kinematic_cost + trunk_cost,
            'Comb_Eff_Vel_Pain_Scruff': effort_cost + 1000*velocity_cost + trunk_cost + pain_cost,
            'Comb_Kine_Vel_Pain_Scruff': kinematic_cost + 1000*velocity_cost + trunk_cost + pain_cost,
            'Actual_velocity': steps_dist / steps_duration,
            'Actual_dist': steps_dist,
            'Actual_time': steps_duration,
            'Pelvis_Cost': pelvis_cost
        }


def calculate_scruff_cost(left_stance_foot, right_stance_foot, stride_num, unpacked_grf, unpacked_spinal_phases_l, unpacked_spinal_phases_r):
    """Calculate cost to prevent foot scuffing during swing phase"""
    left_scruff_idx = left_stance_foot[-(stride_num+1):len(left_stance_foot)]
    right_scruff_idx = right_stance_foot[-(stride_num+1):len(right_stance_foot)]
    
    scruff_cost = 0
    for idx in range(len(left_scruff_idx)-1):
        # Get spinal phases for swing detection - phase 4 in legacy code is now phase 10 (swing phase)
        scruff_phase_l = unpacked_spinal_phases_l[left_scruff_idx[idx]:left_scruff_idx[idx+1], 10]
        scruff_phase_r = unpacked_spinal_phases_r[right_scruff_idx[idx]:right_scruff_idx[idx+1], 10]
        
        swing_start_l = np.array([value for value in np.where(scruff_phase_l==1)[0] if value >= 5])
        swing_start_r = np.array([value for value in np.where(scruff_phase_r==1)[0] if value >= 5])
        
        if len(swing_start_l) == 0 or len(swing_start_r) == 0:
            return None  # Signal to use early termination cost
            
        left_swing_start = left_scruff_idx[idx] + swing_start_l[0]
        right_swing_start = right_scruff_idx[idx] + swing_start_r[0]
        
        scruff_cost += (np.sum(unpacked_grf[left_swing_start:left_scruff_idx[idx+1], 0]) + 
                       np.sum(unpacked_grf[right_swing_start:right_scruff_idx[idx+1], 1]))
    
    return scruff_cost 