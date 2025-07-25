# Author(s): Chun Kwang Tan <cktan.neumove@gmail.com>, Calder Robbins <robbins.cal@northeastern.edu>
"""
Parameter bounds for optimization.

This module defines bounds for optimization parameters for different 
muscle models and control modes.
"""

import numpy as np
from typing import List, Tuple, Optional
from ..utils.npoint_torque import calculate_npoint_torques

# Global variable to store input args
input_args = None

def get_bounds(musc_model: str, control_mode: str) -> List[List[float]]:
    """Get bounds for optimization parameters for a given muscle model and control mode."""
    if musc_model == '22' or musc_model == '26':
        return getBounds_22_26_mus(control_mode)
    elif musc_model == '80':
        return getBounds_80mus(control_mode)
    elif musc_model == 'leg_11':
        return getBounds_22_26_mus(control_mode) # Backwards compatibility
    else:
        raise ValueError(f"No bounds defined for muscle model: {musc_model}")

def getBounds_22_26_mus(mode: str) -> List[List[float]]:
    """Get bounds for optimization parameters for 22/26-muscle models.
    
    Args:
        mode: Control mode ('2D' or '3D')
        
    Returns:
        List containing [bound_start, bound_end] where each is a list of bounds
    """
    if not input_args:
        raise ValueError("input_args not set. Call set_input_args() before using this function.")
        
    if mode == '2D':
        # Legacy bounds - DO NOT MODIFY THESE
        bound_vect = [
            [0, np.inf],     # theta_tgt
            [0, np.inf],     # alpha_0
            [0, np.inf],     # alpha_delta
            [0, np.inf],     # C_d
            [0, np.inf],     # C_v
            [0, np.inf],     # Tr_St_sup
            [0, np.inf],     # Tr_Sw
            [-np.inf, np.inf], # knee_tgt # [-100/15, 20/15]
            [-np.inf, np.inf], # knee_sw_tgt # [-3, 3]
            [-np.inf, np.inf], # knee_off_st # [-75/10, 15/10]
            [-np.inf, np.inf], # ankle_tgt # [-1, 3]
            [-np.inf, np.inf], # mtp_tgt # [-1, 3]
            [0, np.inf], # 1_GLU_FG
            [0, np.inf], # 1_VAS_FG
            [0, np.inf], # 1_SOL_FG
            [0, np.inf], # 2_HAM_FG
            [0, np.inf], # 2_BFSH_VAS_PG
            [0, np.inf], # 2_BFSH_PG
            [0, np.inf], # 2_GAS_FG
            [0, np.inf], # 3_HFL_Th
            [0, np.inf], # 3_HFL_d_Th
            [0, np.inf], # 3_GLU_Th
            [0, np.inf], # 3_GLU_d_Th
            [0, np.inf], # 3_GLU_HAM_SG
            [0, np.inf], # 4_C_GLU_HFL_PG
            [0, np.inf], # 4_C_HAM_HFL_PG
            [0, np.inf], # 4_C_HFL_GLU_PG
            [0, np.inf], # 4_C_RF_GLU_PG
            [0, np.inf], # 4_GLU_HAM_PG 
            [0, np.inf], # 5_TA_PG_st
            [0, np.inf], # 5_TA_PG_sw
            [0, np.inf], # 5_TA_SOL_FG
            [0, np.inf], # 6_RF_HFL_PG
            [0, np.inf], # 6_RF_HFL_VG
            [0, np.inf], # 6_HAM_GLU_PG
            [0, np.inf], # 6_HAM_GLU_VG
            [0, np.inf], # 7_RF_BFSH_VG
            [0, np.inf], # 7_BFSH_PG
            [0, np.inf], # 8_RF_VG
            [0, np.inf], # 8_BFSH_PG
            [0, np.inf], # 8_BFSH_VG
            [0, np.inf], # 9_HAM_PG
            [0, np.inf], # 9_HAM_BFSH_SG
            [0, np.inf], # 9_HAM_BFSH_Thr
            [0, np.inf], # 9_HAM_GAS_SG
            [0, np.inf], # 9_HAM_GAS_Thr
            [0, np.inf], # 10_HFL_PG
            [0, np.inf], # 10_GLU_PG
            [0, np.inf], # 10_VAS_PG
            [-np.inf, np.inf], # pelvis_tilt
            [-np.inf, np.inf], # hip_flexion_r
            [-np.inf, np.inf], # hip_flexion_l
            [-np.inf, np.inf], # knee_angle_r
            [-np.inf, np.inf], # knee_angle_l
            [-np.inf, np.inf], # ankle_angle_r
            [-np.inf, np.inf], # mtp_angle_r
            [-np.inf, np.inf], # ankle_angle_l
            [-np.inf, np.inf], # mtp_angle_l
            [-29, 6], # vel_pelvis_tx
            [1, 50], # GLU_r
            [1, 50], # HFL_r
            [1, 50], # HAM_r
            [1, 50], # RF_r
            [1, 50], # BFSH_r
            [1, 50], # GAS_r
            [1, 50], # SOL_r
            [1, 50], # VAS_r
            [1, 50], # TA_r
            [1, 50], # GLU_l
            [1, 50], # HFL_l
            [1, 50], # HAM_l
            [1, 50], # RF_l
            [1, 50], # BFSH_l
            [1, 50], # GAS_l
            [1, 50], # SOL_l
            [1, 50], # VAS_l
            [1, 50], # TA_l
        ]

        if input_args.ExoOn:
            if input_args.use_4param_spline:
                # Legacy spline uses four normalised timing parameters
                for _ in range(4):
                    bound_vect.append([0, 1])
            else:
                # n-point spline – append n torque amplitudes followed by n timing
                n_points = input_args.n_points

                # Normalised torque amplitudes (0-1)
                for _ in range(n_points):
                    bound_vect.append([0, 1])
            
                # Normalised timing parameters (0-1)
                for _ in range(n_points):
                    bound_vect.append([0, 1])

        # Convert to start and end bounds
        bound_start = [i[0] for i in bound_vect]
        bound_end = [i[1] for i in bound_vect]

        return [bound_start, bound_end]
    elif mode == '3D':
        # Add bounds for 3D movement
        bound_vect.extend([
            [-np.inf, np.inf], # hip_adduction_r
            [-np.inf, np.inf], # hip_adduction_l
            [-np.inf, np.inf], # hip_rotation_r
            [-np.inf, np.inf], # hip_rotation_l
            [-4, 4],           # pelvis_list
            [-4, 4],           # pelvis_rotation
        ])
        
        if input_args.ExoOn:
            if input_args.use_4param_spline:
                # Legacy spline uses four normalised timing parameters
                for _ in range(4):
                    bound_vect.append([0, 1])
            else:
                # n-point spline – append n torque amplitudes followed by n timing
                n_points = input_args.n_points
                # Normalised torque amplitudes (0-1)
                for _ in range(n_points):
                    bound_vect.append([0, 1])
                # Normalised timing parameters (0-1)
                for _ in range(n_points):
                    bound_vect.append([0, 1])

        # Convert to start and end bounds
        bound_start = [i[0] for i in bound_vect]
        bound_end = [i[1] for i in bound_vect]

        return [bound_start, bound_end]

def getBounds_80mus(control_mode: str) -> List[List[float]]:
    """Return legacy CMA-ES bounds for the unified 80-muscle reflex model.

    These bounds are copied verbatim from the original
    train_EXO_MyoReflex_11mus_80mus.py implementation so that
    optimization behaviour is identical.
    
    Args:
        control_mode: '2D' or '3D'
    
    Returns:
        (bound_start, bound_end) where each is a list of lower/upper limits.
    """
    if control_mode == '2D':
        bound_vect = [
            [0, np.inf],  # theta_tgt
            [0, np.inf],  # alpha_0
            [0, np.inf],  # alpha_delta
            [0, np.inf],  # C_d
            [0, np.inf],  # C_v
            [0, np.inf],  # Tr_St_sup
            [0, np.inf],  # Tr_Sw
            [-np.inf, np.inf],  # knee_tgt
            [-np.inf, np.inf],  # knee_sw_tgt
            [-np.inf, np.inf],  # knee_off_st
            [-np.inf, np.inf],  # ankle_tgt
            [-np.inf, np.inf],  # mtp_tgt
            [0, np.inf],  # 1_GLU_FG
            [0, np.inf],  # 1_VAS_FG
            [0, np.inf],  # 1_SOL_FG
            [0, np.inf],  # 1_TOFL_FG
            [0, np.inf],  # 2_HAM_FG
            [0, np.inf],  # 2_BFSH_VAS_PG
            [0, np.inf],  # 2_BFSH_PG
            [0, np.inf],  # 2_GAS_FG
            [0, np.inf],  # 3_HFL_Th
            [0, np.inf],  # 3_HFL_d_Th
            [0, np.inf],  # 3_GLU_Th
            [0, np.inf],  # 3_GLU_d_Th
            [0, np.inf],  # 3_GLU_HAM_SG
            [0, np.inf],  # 4_C_GLU_HFL_PG
            [0, np.inf],  # 4_C_HAM_HFL_PG
            [0, np.inf],  # 4_C_HFL_GLU_PG
            [0, np.inf],  # 4_C_RF_GLU_PG
            [0, np.inf],  # 4_GLU_HAM_PG
            [0, np.inf],  # 5_TA_PG_st
            [0, np.inf],  # 5_TA_PG_sw
            [0, np.inf],  # 5_SOL_TA_FG
            [0, np.inf],  # 5_TOEX_PG_st
            [0, np.inf],  # 5_TOEX_PG_sw
            [0, np.inf],  # 5_TOEX_TOFL_FG
            [0, np.inf],  # 6_RF_HFL_PG
            [0, np.inf],  # 6_RF_HFL_VG
            [0, np.inf],  # 6_HAM_GLU_PG
            [0, np.inf],  # 6_HAM_GLU_VG
            [0, np.inf],  # 7_RF_BFSH_VG
            [0, np.inf],  # 7_BFSH_PG
            [0, np.inf],  # 8_RF_VG
            [0, np.inf],  # 8_BFSH_PG
            [0, np.inf],  # 8_BFSH_VG
            [0, np.inf],  # 9_HAM_PG
            [0, np.inf],  # 9_HAM_BFSH_SG
            [0, np.inf],  # 9_HAM_BFSH_Thr
            [0, np.inf],  # 9_HAM_GAS_SG
            [0, np.inf],  # 9_HAM_GAS_Thr
            [0, np.inf],  # 10_HFL_PG
            [0, np.inf],  # 10_GLU_PG
            [0, np.inf],  # 10_VAS_PG
            [-np.inf, np.inf],  # pelvis_tilt
            [-np.inf, np.inf],  # hip_flexion_r
            [-np.inf, np.inf],  # hip_flexion_l
            [-np.inf, np.inf],  # knee_angle_r
            [-np.inf, np.inf],  # knee_angle_l
            [-np.inf, np.inf],  # ankle_angle_r
            [-np.inf, np.inf],  # ankle_angle_l
            [-29, 6],  # vel_pelvis_tx
            [1, 50],  # GLU_r
            [1, 50],  # HFL_r
            [1, 50],  # HAM_r
            [1, 50],  # RF_r
            [1, 50],  # BFSH_r
            [1, 50],  # GAS_r
            [1, 50],  # SOL_r
            [1, 50],  # VAS_r
            [1, 50],  # TA_r
            [1, 50],  # GLU_l
            [1, 50],  # HFL_l
            [1, 50],  # HAM_l
            [1, 50],  # RF_l
            [1, 50],  # BFSH_l
            [1, 50],  # GAS_l
            [1, 50],  # SOL_l
            [1, 50],  # VAS_l
            [1, 50],  # TA_l
        ]
    elif control_mode == '3D':
        bound_vect = [
            [0, np.inf],  # theta_tgt
            [0, np.inf],  # alpha_0
            [0, np.inf],  # alpha_delta
            [0, np.inf],  # C_d
            [0, np.inf],  # C_v
            [0, np.inf],  # Tr_St_sup
            [0, np.inf],  # Tr_Sw
            [-np.inf, np.inf],  # knee_tgt
            [-np.inf, np.inf],  # knee_sw_tgt
            [-np.inf, np.inf],  # knee_off_st
            [-np.inf, np.inf],  # ankle_tgt
            [-np.inf, np.inf],  # mtp_tgt
            [0, np.inf],  # 1_GLU_FG
            [0, np.inf],  # 1_VAS_FG
            [0, np.inf],  # 1_SOL_FG
            [0, np.inf],  # 1_TOFL_FG
            [0, np.inf],  # 2_HAM_FG
            [0, np.inf],  # 2_BFSH_VAS_PG
            [0, np.inf],  # 2_BFSH_PG
            [0, np.inf],  # 2_GAS_FG
            [0, np.inf],  # 3_HFL_Th
            [0, np.inf],  # 3_HFL_d_Th
            [0, np.inf],  # 3_GLU_Th
            [0, np.inf],  # 3_GLU_d_Th
            [0, np.inf],  # 3_GLU_HAM_SG
            [0, np.inf],  # 4_C_GLU_HFL_PG
            [0, np.inf],  # 4_C_HAM_HFL_PG
            [0, np.inf],  # 4_C_HFL_GLU_PG
            [0, np.inf],  # 4_C_RF_GLU_PG
            [0, np.inf],  # 4_GLU_HAM_PG
            [0, np.inf],  # 5_TA_PG_st
            [0, np.inf],  # 5_TA_PG_sw
            [0, np.inf],  # 5_SOL_TA_FG
            [0, np.inf],  # 5_TOEX_PG_st
            [0, np.inf],  # 5_TOEX_PG_sw
            [0, np.inf],  # 5_TOEX_TOFL_FG
            [0, np.inf],  # 6_RF_HFL_PG
            [0, np.inf],  # 6_RF_HFL_VG
            [0, np.inf],  # 6_HAM_GLU_PG
            [0, np.inf],  # 6_HAM_GLU_VG
            [0, np.inf],  # 7_RF_BFSH_VG
            [0, np.inf],  # 7_BFSH_PG
            [0, np.inf],  # 8_RF_VG
            [0, np.inf],  # 8_BFSH_PG
            [0, np.inf],  # 8_BFSH_VG
            [0, np.inf],  # 9_HAM_PG
            [0, np.inf],  # 9_HAM_BFSH_SG
            [0, np.inf],  # 9_HAM_BFSH_Thr
            [0, np.inf],  # 9_HAM_GAS_SG
            [0, np.inf],  # 9_HAM_GAS_Thr
            [0, np.inf],  # 10_HFL_PG
            [0, np.inf],  # 10_GLU_PG
            [0, np.inf],  # 10_VAS_PG
            [-np.inf, np.inf],  # pelvis_tilt
            [-np.inf, np.inf],  # hip_flexion_r
            [-np.inf, np.inf],  # hip_flexion_l
            [-np.inf, np.inf],  # knee_angle_r
            [-np.inf, np.inf],  # knee_angle_l
            [-np.inf, np.inf],  # ankle_angle_r
            [-np.inf, np.inf],  # ankle_angle_l
            [-29, 6],  # vel_pelvis_tx
            [-np.inf, np.inf],  # hip_adduction_r
            [-np.inf, np.inf],  # hip_adduction_l
            [-np.inf, np.inf],  # hip_rotation_r
            [-np.inf, np.inf],  # hip_rotation_l
            [-4, 4],  # pelvis_list
            [-4, 4],  # pelvis_rotation
            [-np.inf, np.inf],  # subtalar_angle_r
            [-np.inf, np.inf],  # subtalar_angle_l
        ]

    bound_start = [i[0] for i in bound_vect]
    bound_end = [i[1] for i in bound_vect]
    return [bound_start, bound_end]


def getBounds_expanded_80mus(control_mode: str) -> List[List[float]]:
    """Return bounds for the expanded 80-muscle reflex model.
    
    Args:
        control_mode: '2D' or '3D'
        
    Returns:
        (bound_start, bound_end) where each is a list of lower/upper limits
    """
    if control_mode == '2D':
        bound_vect = [
            [0, np.inf], # theta_tgt
            [0, np.inf], # alpha_0
            [0, np.inf], # alpha_delta
            [0, np.inf], # C_d
            [0, np.inf], # C_v
            [0, np.inf], # Tr_St_sup
            [0, np.inf], # Tr_Sw
            [-np.inf, np.inf], # knee_tgt # [-100/15, 20/15]
            [-np.inf, np.inf], # knee_sw_tgt # [-3, 3]
            [-np.inf, np.inf], # knee_off_st # [-75/10, 15/10]
            [-np.inf, np.inf], # ankle_tgt # [-1, 3]
            [-np.inf, np.inf], # mtp_tgt # [-1, 3]
            [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], # 1_GLU_FG
            [0, np.inf], [0, np.inf], [0, np.inf], # 1_VAS_FG
            [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], # 1_SOL_FG
            [0, np.inf], [0, np.inf], # 1_TOFL_FG
            [0, np.inf], [0, np.inf], [0, np.inf], # 2_HAM_FG
            [0, np.inf], [0, np.inf], [0, np.inf], # 2_VAS_BFSH_PG
            [0, np.inf], # 2_BFSH_PG
            [0, np.inf], [0, np.inf], # 2_GAS_FG
            [0, np.inf], [0, np.inf], # 3_HFL_Th
            [0, np.inf], [0, np.inf], # 3_HFL_d_Th
            [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], # 3_GLU_Th
            [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], # 3_GLU_d_Th
            [0, np.inf], [0, np.inf], [0, np.inf], # 3_HAM_GLU_SG
            [0, np.inf], [0, np.inf], # 4_HFL_C_GLU_PG
            [0, np.inf], [0, np.inf], # 4_HFL_C_HAM_PG
            [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], # 4_GLU_C_HFL_PG
            [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], # 4_GLU_C_RF_PG
            [0, np.inf], [0, np.inf], [0, np.inf], # 4_HAM_GLU_PG
            [0, np.inf], # 5_TA_PG_st
            [0, np.inf], # 5_TA_PG_sw
            [0, np.inf], # 5_TA_SOL_FG
            [0, np.inf], [0, np.inf], # 5_TOEX_PG_st
            [0, np.inf], [0, np.inf], # 5_TOEX_PG_sw
            [0, np.inf], [0, np.inf], # 5_TOEX_TOFL_FG
            [0, np.inf], [0, np.inf], # 6_HFL_RF_PG
            [0, np.inf], [0, np.inf], # 6_HFL_RF_VG
            [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], # 6_GLU_HAM_PG
            [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], # 6_GLU_HAM_VG
            [0, np.inf], # 7_BFSH_RF_VG
            [0, np.inf], # 7_BFSH_PG
            [0, np.inf], # 8_RF_VG
            [0, np.inf], # 8_BFSH_PG
            [0, np.inf], # 8_BFSH_VG
            [0, np.inf], [0, np.inf], [0, np.inf], # 9_HAM_PG
            [0, np.inf], # 9_BFSH_HAM_SG
            [0, np.inf], # 9_BFSH_HAM_Thr
            [0, np.inf], [0, np.inf], # 9_GAS_HAM_SG
            [0, np.inf], [0, np.inf], # 9_GAS_HAM_Thr
            [0, np.inf], [0, np.inf], # 10_HFL_PG
            [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], # 10_GLU_PG
            [0, np.inf], [0, np.inf], [0, np.inf], # 10_VAS_PG
            [-np.inf, np.inf], # pelvis_tilt
            [-np.inf, np.inf], # hip_flexion_r
            [-np.inf, np.inf], # hip_flexion_l
            [-np.inf, np.inf], # knee_angle_r
            [-np.inf, np.inf], # knee_angle_l
            [-np.inf, np.inf], # ankle_angle_r
            [-np.inf, np.inf], # ankle_angle_l
            [-29, 6], # vel_pelvis_tx
            [1, 50], # GLU_r
            [1, 50], # HFL_r
            [1, 50], # HAM_r
            [1, 50], # RF_r
            [1, 50], # BFSH_r
            [1, 50], # GAS_r
            [1, 50], # SOL_r
            [1, 50], # VAS_r
            [1, 50], # TA_r
            [1, 50], # GLU_l
            [1, 50], # HFL_l
            [1, 50], # HAM_l
            [1, 50], # RF_l
            [1, 50], # BFSH_l
            [1, 50], # GAS_l
            [1, 50], # SOL_l
            [1, 50], # VAS_l
            [1, 50], # TA_l
        ]
    elif control_mode == '3D':
        # Add 3D-specific bounds
        bound_vect = [
            # ... same as 2D bounds ...
            # Add hip adduction/rotation bounds
            [-np.inf, np.inf], # hip_adduction_r
            [-np.inf, np.inf], # hip_adduction_l
            [-np.inf, np.inf], # hip_rotation_r
            [-np.inf, np.inf], # hip_rotation_l
            [-4, 4], # pelvis_list
            [-4, 4], # pelvis_rotation
            [-np.inf, np.inf], # subtalar_angle_r
            [-np.inf, np.inf], # subtalar_angle_l
        ]

    bound_start = [i[0] for i in bound_vect]
    bound_end = [i[1] for i in bound_vect]
    return [bound_start, bound_end] 