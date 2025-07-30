# Author(s): Calder Robbins <robbins.cal@northeastern.edu>
import os
import sys
from datetime import datetime
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from matplotlib.backends.backend_pdf import PdfPages

class MyoReport(object):
    def __init__(self, ref_kinematics_path='ref_kinematics_radians_mod.csv', 
                 ref_emg_path='ref_EMG.csv'):
        """
        Initialize MyoReport with reference data and default muscle labels.
        
        Args:
            ref_kinematics_path: Path to kinematics reference data
            ref_emg_path: Path to EMG reference data
        """
        try:
            # Load reference data
            ref_data = np.loadtxt(ref_kinematics_path, delimiter=',')
            rad_to_deg = 180 / np.pi
            
            # Reference data column mapping:
            # Column 0: Trunk
            # Columns 1-3: Right leg (hip, knee, ankle)
            # Columns 4-6: Left leg (hip, knee, ankle)
            self.ref_kinematics = {
                'l_leg': {
                    'trunk': -1 * (ref_data[:, 0] * rad_to_deg),        # Trunk (col 0)
                    'hip': -1 * (ref_data[:, 4] * rad_to_deg) + 180,    # Left Hip (col 4)
                    'knee': -1 * (ref_data[:, 5] * rad_to_deg) + 180,   # Left Knee (col 5)
                    'ankle': -1 * (ref_data[:, 6] * rad_to_deg) + 90    # Left Ankle (col 6)
                },
                'r_leg': {
                    'trunk': -1 * (ref_data[:, 0] * rad_to_deg),        # Trunk (col 0)
                    'hip': -1 * (ref_data[:, 1] * rad_to_deg) + 180,    # Right Hip (col 1)Fc
                    'knee': -1 * (ref_data[:, 2] * rad_to_deg) + 180,   # Right Knee (col 2)
                    'ankle': -1 * (ref_data[:, 3] * rad_to_deg) + 90    # Right Ankle (col 3)
                }
            }
            
            self.ref_emg = np.loadtxt(ref_emg_path, delimiter=',')
        except FileNotFoundError as e:
            print(f"Error loading reference data: {e}")
            raise

        # Define default muscle labels
        self.default_muscle_labels = {
            'l_leg': {
                'HAB': ['HAB'],
                'HAD': ['HAD'],
                'HFL': ['HFL'],
                'GLU': ['GLU'],
                'HAM': ['HAM'],
                'RF': ['RF'],
                'VAS': ['VAS'],
                'BFSH': ['BFSH'],
                'GAS': ['GAS'],
                'SOL': ['SOL'],
                'TA': ['TA']
            },
            'r_leg': {
                'HAB': ['HAB'],
                'HAD': ['HAD'],
                'HFL': ['HFL'],
                'GLU': ['GLU'],
                'HAM': ['HAM'],
                'RF': ['RF'],
                'VAS': ['VAS'],
                'BFSH': ['BFSH'],
                'GAS': ['GAS'],
                'SOL': ['SOL'],
                'TA': ['TA']
            }
        }
    
    def load_simulation_data(self, prefix="20250312", data_dir="visualization_outputs/data"):
        """
        Load and process all simulation data files with given prefix.
        
        Args:
            prefix: Prefix for JSON files to process
            data_dir: Directory containing simulation data
            
        Returns:
            list: List of dictionaries containing processed data:
                - unpacked_dict: Processed simulation data
                - metadata: Simulation parameters
                - filename: Original JSON filename
        """
        # Find all matching JSON files
        json_files = [f for f in os.listdir(data_dir) if f.startswith(prefix) and f.endswith(".json")]
        if not json_files:
            raise FileNotFoundError(f"No JSON files found with prefix {prefix} in {data_dir}")
        
        processed_data = []
        
        for json_file in json_files:
            file_path = os.path.join(data_dir, json_file)
            print(f"Processing {json_file}...")
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract metadata from simulation parameters
                metadata = {
                    'popsize': data['metadata']['all_flags'].get('popsize', 'N/A'),
                    'maxiter': data['metadata']['all_flags'].get('maxiter', 'N/A'),
                    'threads': data['metadata']['all_flags'].get('threads', 'N/A'),
                    'sigma_gain': data['metadata']['all_flags'].get('sigma_gain', 'N/A'),
                    'tgt_vel': data['metadata']['all_flags'].get('tgt_vel', 'N/A'),
                    'tgt_sym_th': data['metadata']['all_flags'].get('tgt_sym_th', 'N/A'),
                    'tgt_grf_th': data['metadata']['all_flags'].get('tgt_grf_th', 'N/A'),
                    'trunk_err_type': data['metadata']['all_flags'].get('trunk_err_type', 'N/A'),
                    'n_points': data['metadata']['all_flags'].get('n_points', 'N/A'),
                    'use_4param_spline': data['metadata']['all_flags'].get('use_4param_spline', 'N/A'),
                    'max_torque': data['metadata']['all_flags'].get('max_torque', 'N/A')
                }
                
                # Process simulation data
                unpacked_dict = self._process_simulation_data(data)
                
                # Store processed data
                processed_data.append({
                    'unpacked_dict': unpacked_dict,
                    'metadata': metadata,
                    'filename': json_file
                })
                
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
                continue
        
        if not processed_data:
            raise ValueError("No files were successfully processed")
        
        print(f"Successfully processed {len(processed_data)} files")
        return processed_data

    def _detect_strides(self, unpacked_dict, threshold=0.1, dt=0.01, min_time=0.5):
        """Detect heel strikes using vGRF data, with visualization."""
           
        def plot_debug(vgrf, heel_strikes, leg_name):
            """Plot GRF data with detected heel strikes and alternating stride colors"""
            plt.figure(figsize=(15, 5))
            
            # Plot full GRF signal
            plt.plot(vgrf, 'b-', label='GRF', alpha=0.6)
            
            # Plot threshold line
            plt.axhline(y=threshold, color='r', linestyle='--', 
                    label=f'Threshold ({threshold})', alpha=0.5)
            
            # Plot detected heel strikes
            if len(heel_strikes) > 0:
                plt.plot(heel_strikes, vgrf[heel_strikes], 'go', 
                        label='Heel Strikes', markersize=10)
                
                # Highlight stride regions with alternating colors
                colors = ['lightgreen', 'lightblue']  # Alternating colors
                for i in range(len(heel_strikes)-1):
                    plt.axvspan(heel_strikes[i], heel_strikes[i+1], 
                            color=colors[i % 2], 
                            alpha=0.6,
                            label=f'Stride {i+1}' if i == 0 else "")
            
            plt.title(f'{leg_name} GRF and Detected Heel Strikes')
            plt.xlabel('Time Steps')
            plt.ylabel('Ground Reaction Force')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add min_time visualization
            min_samples = int(min_time / dt)
            plt.text(0.02, 0.98, 
                    f'Min time between strikes: {min_time}s ({min_samples} samples)',
                    transform=plt.gca().transAxes,
                    verticalalignment='top')
            
            plt.show()

        def find_heel_strikes(vgrf):
            """Find heel strikes using vertical ground reaction force"""
            heel_strikes = []
            min_samples = int(min_time / dt)
            last_strike = -min_samples
            
            for i in range(1, len(vgrf)):
                if vgrf[i] > threshold and vgrf[i-1] <= threshold:
                    if (i - last_strike) >= min_samples:
                        heel_strikes.append(i)
                        last_strike = i
            
            return np.array(heel_strikes)

        stride_info = {}
        for leg in ['l_leg', 'r_leg']:
            # Get vGRF data
            vgrf = unpacked_dict[leg]['load_ipsi']
            
            # Find heel strikes
            heel_strikes = find_heel_strikes(vgrf)
            
            # Plot debug visualization
            plot_debug(vgrf, heel_strikes, leg)
            
            # Need at least 2 heel strikes to define a stride
            if len(heel_strikes) >= 2:
                stride_info[leg] = {
                    'starts': heel_strikes[:-1],
                    'ends': heel_strikes[1:],
                    'durations': np.diff(heel_strikes),
                    'num_strides': len(heel_strikes) - 1
                }
            else:
                stride_info[leg] = {
                    'starts': np.array([]),
                    'ends': np.array([]),
                    'durations': np.array([]),
                    'num_strides': 0
                }
            
            print(f"{leg}: {stride_info[leg]['num_strides']} strides detected")
            # if stride_info[leg]['num_strides'] > 0:
            #     print(f"Stride durations: {stride_info[leg]['durations']}")

        return stride_info

    def _get_analysis_strides(self, stride_info, num_desired_strides=5):
        """
        Get the strides to use for analysis for both legs, preferring the last 5 strides.
        
        Args:
            stride_info: Dictionary containing stride information for both legs
            num_desired_strides: Number of strides desired for analysis (default 5)
        
        Returns:
            dict: Analysis strides for both legs containing:
                l_leg: {
                    starts: Array of stride start indices for analysis
                    ends: Array of stride end indices for analysis
                    durations: Array of stride durations for analysis
                    num_strides: Number of strides being used
                }
                r_leg: {
                    starts: Array of stride start indices for analysis
                    ends: Array of stride end indices for analysis
                    durations: Array of stride durations for analysis
                    num_strides: Number of strides being used
                }
        """
        analysis_strides = {}
        
        for leg in ['l_leg', 'r_leg']:
            if stride_info[leg]['num_strides'] < 1:
                print(f"Warning: No strides detected for {leg}")
                analysis_strides[leg] = {
                    'starts': np.array([]),
                    'ends': np.array([]),
                    'durations': np.array([]),
                    'num_strides': 0
                }
                continue
                
            num_available = stride_info[leg]['num_strides']
            num_strides = min(num_available, num_desired_strides)
            
            # Take the last n strides
            analysis_strides[leg] = {
                'starts': stride_info[leg]['starts'][-num_strides:],
                'ends': stride_info[leg]['ends'][-num_strides:],
                'durations': stride_info[leg]['durations'][-num_strides:],
                'num_strides': num_strides
            }
            
            print(f"{leg}: Using {num_strides} strides for analysis")
        
        return analysis_strides
    
    def _calculate_gait_phases(self, stride_info, grf_data):
        """
        Calculate stance and swing phases based on GRF data.
        
        Args:
            stride_info: Dictionary containing stride information for a single leg
            grf_data: Ground reaction force data array
            
        Returns:
            tuple: (stance_phase, swing_phase, swing_std)
        """
        if stride_info['num_strides'] < 1:
            return None, None, None
        
        stance_percentages = []
        threshold = 0.1  # GRF threshold for stance detection
        min_stance_samples = 50  # Minimum number of samples to consider valid stance
        
        # Calculate stance duration for each stride
        for i in range(stride_info['num_strides']):
            start_idx = stride_info['starts'][i]
            end_idx = stride_info['ends'][i]
            stride_grf = grf_data[start_idx:end_idx]
            
            # Find potential toe-offs (where GRF drops below threshold)
            toe_off_candidates = np.where(stride_grf < threshold)[0]
            valid_toe_off = None
            
            # Check each candidate toe-off
            for toe_off_idx in toe_off_candidates:
                # Must be at least min_stance_samples after stride start
                if toe_off_idx < min_stance_samples:
                    continue
                    
                # Check if GRF stays below threshold for the next few samples
                if toe_off_idx + 10 < len(stride_grf):  # Ensure we don't go past end of data
                    if np.all(stride_grf[toe_off_idx:toe_off_idx+10] < threshold):
                        valid_toe_off = toe_off_idx
                        break
            
            if valid_toe_off is not None:
                # Convert to percentage of stride
                stance_percent = (valid_toe_off / len(stride_grf)) * 100
                stance_percentages.append(stance_percent)
        
        # Average across all analyzed strides
        mean_stance_end = np.mean(stance_percentages) if stance_percentages else 60
        stance_std = np.std(stance_percentages) if stance_percentages else 0
        
        stance_phase = (0, mean_stance_end)
        swing_phase = (mean_stance_end, 100)
        
        # print(f"Stance duration: {mean_stance_end:.1f}% (±{stance_std:.1f}%)")
        
        return stance_phase, swing_phase, stance_std

    def _process_simulation_data(self, data):
        """Process raw simulation data into format needed for plotting"""
        # Muscle name mapping from JSON to our internal names
        muscle_mapping = {
            'abd': 'HAB',
            'add': 'HAD',
            'iliopsoas': 'HFL',
            'glut_max': 'GLU',
            'hamstrings': 'HAM',
            'rect_fem': 'RF',
            'vasti': 'VAS',
            'bifemsh': 'BFSH',
            'gastroc': 'GAS',
            'soleus': 'SOL',
            'tib_ant': 'TA'
        }

        unpacked_dict = {
            'l_leg': {'joint': {}, 'muscles': {}, 'pelvis_pos': None},
            'r_leg': {'joint': {}, 'muscles': {}, 'pelvis_pos': None},
            'actuator_data': {}  # Add this line to store exo data
        }
        
        # Process joint data - map the JSON joint names to our internal names
        joint_mapping = {
            'hip_flexion_r': ('r_leg', 'hip'),
            'knee_angle_r': ('r_leg', 'knee'),
            'ankle_angle_r': ('r_leg', 'ankle'),
            'hip_flexion_l': ('l_leg', 'hip'),
            'knee_angle_l': ('l_leg', 'knee'),
            'ankle_angle_l': ('l_leg', 'ankle'),
            'pelvis_tilt': ('trunk', None)
        }

        # Process joint data
        for joint_name, joint_data in data['joint_data'].items():
            if joint_name in joint_mapping:
                leg, joint = joint_mapping[joint_name]
                if joint:
                    unpacked_dict[leg]['joint'][joint] = np.array([x[0] for x in joint_data['qpos']])
                elif leg == 'trunk':
                    # Store trunk (pelvis tilt) data
                    unpacked_dict['trunk'] = np.array([x[0] for x in joint_data['qpos']])

        # Process muscle and exo actuator data
        for actuator_name, actuator_data in data['actuator_data'].items():
            if actuator_name in ['Exo_R', 'Exo_L']:
                # Store exo data directly
                unpacked_dict['actuator_data'][actuator_name] = {
                    'force': np.array([x[0] for x in actuator_data['force']]),
                    'ctrl': np.array([x[0] for x in actuator_data['ctrl']])
                }
            else:
                # Process muscle data as before
                leg = 'l_leg' if actuator_name.endswith('_l') else 'r_leg'
                json_muscle = actuator_name.rsplit('_', 1)[0]
                
                if json_muscle in muscle_mapping:
                    muscle = muscle_mapping[json_muscle]
                    
                    if muscle not in unpacked_dict[leg]['muscles']:
                        unpacked_dict[leg]['muscles'][muscle] = {}
                    
                    unpacked_dict[leg]['muscles'][muscle]['f'] = np.array([x[0] for x in actuator_data['force']])
                    unpacked_dict[leg]['muscles'][muscle]['v'] = np.array([x[0] for x in actuator_data['velocity']])
                    unpacked_dict[leg]['muscles'][muscle]['act'] = np.array([x[0] for x in actuator_data['ctrl']])

        # Process load sensor data
        for leg in ['l_leg', 'r_leg']:
            prefix = 'l' if leg == 'l_leg' else 'r'
            sensor_key = f"{prefix}_leg_load_ipsi"
            if sensor_key in data['sensor_data']:
                # Get raw GRF data and ensure it's positive
                grf_data = np.array(data['sensor_data'][sensor_key]['data'])
                unpacked_dict[leg]['load_ipsi'] = np.abs(grf_data) 

        # Store torque data
        for leg in ['l_leg', 'r_leg']:
            prefix = 'hip_flexion_' if leg == 'l_leg' else 'hip_flexion_'
            suffix = 'l' if leg == 'l_leg' else 'r'
            
            for joint in ['hip', 'knee', 'ankle']:
                torque_key = f"{joint}_angle_{suffix}" if joint != 'hip' else f"{prefix}{suffix}"
                if torque_key in data['torque_data']:
                    if 'joint_torque' not in unpacked_dict[leg]:
                        unpacked_dict[leg]['joint_torque'] = {}
                    unpacked_dict[leg]['joint_torque'][joint] = np.array(data['torque_data'][torque_key]['total'])

        return unpacked_dict

    def _create_metadata_page(self, metadata, fig_size=(15.0, 25)):
        """Create a figure with metadata information matching plot page dimensions"""
        fig = plt.figure(figsize=fig_size, layout="constrained")
        
        # Create a single subplot to contain all text
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        fig.suptitle('Simulation Parameters', fontsize=18, fontweight='bold')
        
        # Add metadata items in the center of the page
        y_pos = 0.8  # Start lower to center content
        y_step = 0.05
        for key, value in metadata.items():
            ax.text(0.3, y_pos, f"{key}:", fontweight='bold', fontsize=14)
            ax.text(0.5, y_pos, f"{value}", fontsize=14)
            y_pos -= y_step
        
        return fig
        
    def normToGaitCycle(self, data_mat, stride_info, num_desired_strides=5):
        """
        Normalize data to gait cycle using heel strike to heel strike.
        
        Args:
            data_mat: Data matrix to normalize (can be 1D or 2D)
            stride_info: Dictionary containing stride information
            num_desired_strides: Number of strides to normalize (default 5)
        
        Returns:
            np.array: Normalized data matrix
        """
        if stride_info['num_strides'] < 1:
            raise ValueError("Not enough heel strikes for stride analysis")
        
        # Ensure data_mat is 2D
        data_mat = np.atleast_2d(data_mat)
        if data_mat.shape[1] > data_mat.shape[0]:
            data_mat = data_mat.T
        
        # Take last n strides
        num_strides = min(stride_info['num_strides'], num_desired_strides)
        starts = stride_info['starts'][-num_strides:]
        ends = stride_info['ends'][-num_strides:]
        
        normalized_strides = []
        for start, end in zip(starts, ends):
            # Extract full stride data (heel strike to heel strike)
            stride_data = data_mat[start:end, :]
            
            # Create time vector for this stride
            stride_time = np.linspace(0, 1, len(stride_data))
            
            # Interpolate to 101 points (0-100%)
            interp_stride = np.zeros((101, data_mat.shape[1]))
            for col in range(data_mat.shape[1]):
                interp_stride[:, col] = np.interp(
                    np.linspace(0, 1, 101),  # Use 0-1 for interpolation
                    stride_time,
                    stride_data[:, col]
                )
            normalized_strides.append(interp_stride)
        
        return np.array(normalized_strides)  # Shape: (num_strides, 101, num_features)
    
    def plotAngles_Torques(self, angles, ang_std, forces, force_std, reference, stance_phase=None):
        """Plot joint angles and forces with dynamic axis limits."""
        plt.ioff()
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15.0, 25), layout="constrained")
        
        titles = {
            'angles': ['Trunk', 'Hip', 'Knee', 'Ankle'],
            'forces': ['GRF (%BW)', 'Hip Torque (Nm)', 'Knee Torque (Nm)', 'Ankle Torque (Nm)']
        }
        
        ref_keys = ['trunk', 'hip', 'knee', 'ankle']
        
        for i in range(4):
            # Get reference data for current joint
            ref_data = reference[ref_keys[i]]
            
            # Plot reference data directly
            axes[i, 0].plot(np.linspace(0, 100, len(ref_data)), 
                        ref_data, 
                        'k-', 
                        linewidth=1, 
                        label='Reference')

            # Angle subplot
            axes[i, 0].fill_between(np.arange(len(angles[:, i])),
                                angles[:, i] - ang_std[:, i],
                                angles[:, i] + ang_std[:, i], 
                                color='red', alpha=0.2)
            axes[i, 0].plot(angles[:, i], 'red', linewidth=2, label='Simulated')

            # Force subplot
            axes[i, 1].fill_between(np.arange(len(forces[:, i])),
                                forces[:, i] - force_std[:, i],
                                forces[:, i] + force_std[:, i], 
                                color='red', alpha=0.2)
            axes[i, 1].plot(forces[:, i], 'red', linewidth=2)

            # Add stance phase visualization if provided
            if stance_phase is not None:
                for ax in [axes[i, 0], axes[i, 1]]:
                    ax.axvline(stance_phase[1], color='gray', linestyle='--', alpha=0.5)
                    ax.axvspan(0, stance_phase[1], alpha=0.2, color='gray', label='Stance')
                    ax.axvspan(stance_phase[1], 100, alpha=0.2, color='white', label='Swing')

            # Add labels and formatting
            axes[i, 0].legend(fontsize='large')
            axes[i, 0].set_title(titles['angles'][i], fontsize=14, fontweight='bold')
            axes[i, 1].set_title(titles['forces'][i], fontsize=14, fontweight='bold')
            
            # Set x-axis limits
            axes[i, 0].set_xlim(0, 100)
            axes[i, 1].set_xlim(0, 100)
            
            # Add 5% padding to y-limits
            for ax in [axes[i, 0], axes[i, 1]]:
                ymin, ymax = ax.get_ylim()
                y_range = ymax - ymin
                padding = y_range * 0.05
                ax.set_ylim(ymin - padding, ymax + padding)

        return fig, axes

    def createAngleReport(self, unpacked_dict, ref_angle, leg_arg='l_leg'):
        """Create angle and force report for a single leg."""
        # Get stride information if not already detected
        if not hasattr(self, '_stride_info'):
            self._stride_info = self._detect_strides(unpacked_dict)
        
        # Get analysis strides if not already computed
        if not hasattr(self, '_analysis_strides'):
            self._analysis_strides = self._get_analysis_strides(self._stride_info)
        
        # Check if we have valid strides for analysis
        if self._analysis_strides[leg_arg]['num_strides'] < 1:
            raise ValueError(f"No valid strides available for analysis in {leg_arg}")
        
        # Create matrices for kinematics and dynamics
        extract_kine = np.zeros((len(unpacked_dict[leg_arg]['joint']['hip']), 4))
        extract_force = np.zeros_like(extract_kine)
        
        # Fill kinematics matrix
        rad_to_deg = 180 / np.pi
        extract_kine[:,0] = unpacked_dict['trunk'] * rad_to_deg  # Use trunk (pelvis tilt) instead of pelvis_pos
        extract_kine[:,1] = unpacked_dict[leg_arg]['joint']['hip'] * rad_to_deg
        extract_kine[:,2] = -1 * unpacked_dict[leg_arg]['joint']['knee'] * rad_to_deg  # Knee (negated)
        extract_kine[:,3] = unpacked_dict[leg_arg]['joint']['ankle'] * rad_to_deg
        
        # Fill force matrix
        extract_force[:,0] = unpacked_dict[leg_arg]['load_ipsi'] * 100  # Convert to %BW
        extract_force[:,1] = unpacked_dict[leg_arg]['joint_torque']['hip']
        extract_force[:,2] = unpacked_dict[leg_arg]['joint_torque']['knee']
        extract_force[:,3] = -1 * unpacked_dict[leg_arg]['joint_torque']['ankle']  # Ankle torque negated

        # Normalize to gait cycle
        norm_ang = self.normToGaitCycle(extract_kine, self._analysis_strides[leg_arg])
        norm_force = self.normToGaitCycle(extract_force, self._analysis_strides[leg_arg])
        
        # Calculate means and standard deviations
        ang_mean = np.mean(norm_ang, axis=0)
        ang_std = np.std(norm_ang, axis=0)
        force_mean = np.mean(norm_force, axis=0)
        force_std = np.std(norm_force, axis=0)
        
        # Calculate gait phases
        stance_phase, swing_phase, _ = self._calculate_gait_phases(
            self._analysis_strides[leg_arg],
            unpacked_dict[leg_arg]['load_ipsi']
        )
        
        # Create plot - Always use right leg reference kinematics
        fig, ax = self.plotAngles_Torques(
            ang_mean, ang_std,
            force_mean, force_std,
            ref_angle['r_leg'],  # Always use right leg reference
            stance_phase=stance_phase
        )
        
        # Add labels and title
        leg_name = 'Left' if leg_arg == 'l_leg' else 'Right'
        fig.suptitle(f"{leg_name} Leg Kinematics and Dynamics", fontsize=18, fontweight='bold')
        
        return fig

    def plotMuscleData(self, muscle_dict, muscle_labels, stance_phase=None):
        """Plot muscle force and velocity data."""
        plt.ioff()
        group_labels = list(muscle_dict.keys())
        data_types = ['f', 'v']  # Force and velocity
        data_labels = ['Force (N)', 'Velocity (m/s)']
        
        fig, axes = plt.subplots(nrows=len(group_labels), ncols=2, 
                                figsize=(15.0, 25), layout="constrained")
        
        # Handle single row case
        if len(group_labels) == 1:
            axes = axes.reshape(1, -1)
        
        for row, group in enumerate(group_labels):
            for col, (data_type, label) in enumerate(zip(data_types, data_labels)):
                if group in muscle_dict and data_type in muscle_dict[group]:
                    data = muscle_dict[group][data_type]
                    mean_data = data['mean']
                    std_data = data['std']
                    
                    # Plot each muscle in the group
                    for m in range(mean_data.shape[1]):
                        axes[row, col].fill_between(
                            np.arange(101),
                            mean_data[:, m] - std_data[:, m],
                            mean_data[:, m] + std_data[:, m],
                            color='red', alpha=0.2
                        )
                        axes[row, col].plot(mean_data[:, m], 
                                          color='red',
                                          linewidth=2,
                                          label=muscle_labels[group][m])
                    
                    # Add stance phase if provided
                    if stance_phase is not None:
                        axes[row, col].axvline(stance_phase[1], color='gray', 
                                             linestyle='--', alpha=0.5)
                        axes[row, col].axvspan(0, stance_phase[1], alpha=0.2, 
                                             color='gray', label='Stance')
                        axes[row, col].axvspan(stance_phase[1], 100, alpha=0.2, 
                                             color='white', label='Swing')
                    
                    # Add labels and formatting
                    axes[row, col].set_xlabel('Gait Cycle (%)')
                    axes[row, col].set_ylabel(label)
                    axes[row, col].legend(fontsize='large')
                    
                    # Add padding to y-limits
                    ymin, ymax = axes[row, col].get_ylim()
                    y_range = ymax - ymin
                    padding = y_range * 0.05
                    axes[row, col].set_ylim(ymin - padding, ymax + padding)
        
        return fig, axes
    
    def plotMuscleAct(self, muscle_dict, muscle_labels, reference_EMG=None, stance_phase=None):
        """Plot muscle activations and heatmap."""
        plt.ioff()
        group_labels = list(muscle_dict.keys())
        fig, axes = plt.subplots(nrows=len(group_labels), ncols=2, 
                                figsize=(15.0, 25), layout="constrained")
        
        # Handle single row case
        if len(group_labels) == 1:
            axes = axes.reshape(1, -1)
        
        for row, group in enumerate(group_labels):
            if group in muscle_dict and 'act' in muscle_dict[group]:
                data = muscle_dict[group]['act']
                mean_data = data['mean']
                std_data = data['std']
                
                # Left plot: Activation time series
                if reference_EMG is not None and row < reference_EMG.shape[1]:
                    axes[row, 0].plot(reference_EMG[:, row], 'k-', 
                                    linewidth=1, label='Reference')
                
                # Plot each muscle in the group
                for m in range(mean_data.shape[1]):
                    axes[row, 0].fill_between(
                        np.arange(101),
                        np.clip(mean_data[:, m] - std_data[:, m], 0, 1),
                        np.clip(mean_data[:, m] + std_data[:, m], 0, 1),
                        color='red', alpha=0.2
                    )
                    axes[row, 0].plot(mean_data[:, m], 'red', 
                                    linewidth=2,
                                    label=f'Simulated {muscle_labels[group][m]}')
                
                # Right plot: Heatmap
                im = axes[row, 1].imshow(
                    mean_data.T,
                    aspect='auto',
                    extent=[0, 100, -0.5, len(muscle_labels[group])-0.5],
                    origin='lower',
                    cmap='YlOrRd',
                    vmin=0,
                    vmax=1
                )
                
                # Add labels and formatting
                axes[row, 1].set_yticks(range(len(muscle_labels[group])))
                axes[row, 1].set_yticklabels(muscle_labels[group])
                
                # Add stance phase visualization
                if stance_phase is not None:
                    # Time series plot: add both shading and line
                    axes[row, 0].axvline(stance_phase[1], color='gray', 
                                       linestyle='--', alpha=0.5)
                    axes[row, 0].axvspan(0, stance_phase[1], alpha=0.2, 
                                       color='gray', label='Stance')
                    axes[row, 0].axvspan(stance_phase[1], 100, alpha=0.2, 
                                       color='white', label='Swing')
                    
                    # Heatmap: only add vertical line for toe-off
                    axes[row, 1].axvline(stance_phase[1], color='gray', 
                                       linestyle='--', alpha=0.5)
                
                axes[row, 0].set_ylim(-0.05, 1.05)
                axes[row, 0].legend(fontsize='large')
                axes[row, 0].set_title(group, fontsize=14, fontweight='bold')
        
        # Add colorbar
        fig.colorbar(im, ax=axes[:, 1])
        
        return fig, axes

    def createMuscleReport(self, unpacked_dict, muscle_labels, ref_emg, leg_arg='l_leg'):
        """Create muscle activation, force, and velocity reports."""
        if not hasattr(self, '_analysis_strides'):
            self._stride_info = self._detect_strides(unpacked_dict)
            self._analysis_strides = self._get_analysis_strides(self._stride_info)
        
        if self._analysis_strides[leg_arg]['num_strides'] < 1:
            raise ValueError(f"No valid strides available for analysis in {leg_arg}")
        
        stance_phase, swing_phase, _ = self._calculate_gait_phases(
            self._analysis_strides[leg_arg],
            unpacked_dict[leg_arg]['load_ipsi']
        )
        
        # Process muscle data
        norm_muscle = {}
        muscle_groups = ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']
        
        for group in muscle_groups:
            if group in unpacked_dict[leg_arg]['muscles']:
                norm_muscle[group] = {}
                
                # Process activation data
                if 'act' in unpacked_dict[leg_arg]['muscles'][group]:
                    act_data = unpacked_dict[leg_arg]['muscles'][group]['act']
                    norm_data = self.normToGaitCycle(act_data, self._analysis_strides[leg_arg])
                    norm_muscle[group]['act'] = {
                        'raw': norm_data,  # Shape: (num_strides, 101, num_features)
                        'mean': np.mean(norm_data, axis=0),  # Shape: (101, num_features)
                        'std': np.std(norm_data, axis=0)     # Shape: (101, num_features)
                    }
                
                # Process force and velocity data
                for data_type in ['f', 'v']:
                    raw_data = unpacked_dict[leg_arg]['muscles'][group][data_type]
                    norm_data = self.normToGaitCycle(raw_data, self._analysis_strides[leg_arg])
                    norm_muscle[group][data_type] = {
                        'raw': norm_data,  # Shape: (num_strides, 101, num_features)
                        'mean': np.mean(norm_data, axis=0),  # Shape: (101, num_features)
                        'std': np.std(norm_data, axis=0)     # Shape: (101, num_features)
                    }
        
        # Create activation plots
        act_fig, _ = self.plotMuscleAct(
            norm_muscle,
            muscle_labels[leg_arg],
            reference_EMG=ref_emg,
            stance_phase=stance_phase
        )
        
        # Create force/velocity plots
        fv_fig, _ = self.plotMuscleData(
            norm_muscle,
            muscle_labels[leg_arg],
            stance_phase=stance_phase
        )
        
        leg_name = 'Left' if leg_arg == 'l_leg' else 'Right'
        act_fig.suptitle(f"{leg_name} Leg Muscle Activation", fontsize=18, fontweight='bold')
        fv_fig.suptitle(f"{leg_name} Leg Muscle Data", fontsize=18, fontweight='bold')
        
        return act_fig, fv_fig


    def plotExoReport(self, exo_data, stride_info, exo_norm_data, stance_phases):
        """Plot all exoskeleton data on one page."""
        plt.ioff()
        fig = plt.figure(figsize=(15.0, 25), layout="constrained")
        
        # Create a 4x1 grid for the plots
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])
        
        # Time series plots (full width)
        ax1 = fig.add_subplot(gs[0, :])  # Spans both columns
        ax2 = fig.add_subplot(gs[1, :])  # Spans both columns
        
        # Plot left leg time series
        ax1.plot(-1 * exo_data['Exo_L']['force'], 'red', linewidth=2)
        ax1.set_title('Left Leg Exoskeleton Torque', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Torque (Nm)')
        
        # Plot right leg time series
        ax2.plot(-1 * exo_data['Exo_R']['force'], 'red', linewidth=2)
        ax2.set_title('Right Leg Exoskeleton Torque', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Torque (Nm)')
        ax2.set_xlabel('Time Steps')
        
        # Add stance phase visualization for time series
        for ax, leg in zip([ax1, ax2], ['l_leg', 'r_leg']):
            if stride_info[leg]['num_strides'] > 0:
                # Get GRF data for this leg
                grf_data = np.where(exo_data['Exo_L' if leg == 'l_leg' else 'Exo_R']['force'] > 0.1)[0]
                
                # For each stride
                for start, end in zip(stride_info[leg]['starts'], stride_info[leg]['ends']):
                    stride_length = end - start
                    # Calculate stance end as percentage of stride
                    stance_end = int(stance_phases[leg][1] * stride_length / 100)
                    # Shade only the stance phase portion
                    ax.axvspan(start, start + stance_end, alpha=0.2, color='gray')
        
        # Normalized plots (2x2 grid in bottom half)
        axes = [
            fig.add_subplot(gs[2, 0]),  # Left force
            fig.add_subplot(gs[2, 1]),  # Right force
            fig.add_subplot(gs[3, 0]),  # Left control
            fig.add_subplot(gs[3, 1])   # Right control
        ]
        data_types = ['force', 'force', 'ctrl', 'ctrl']
        legs = ['l_leg', 'r_leg', 'l_leg', 'r_leg']
        titles = ['Left Leg Force', 'Right Leg Force', 'Left Leg Control', 'Right Leg Control']
        y_labels = ['Torque (Nm)', 'Torque (Nm)', 'Control Signal', 'Control Signal']
        
        # Plot normalized data
        for ax, data_type, leg, title, ylabel in zip(axes, data_types, legs, titles, y_labels):
            if leg in exo_norm_data and data_type in exo_norm_data[leg]:
                data = exo_norm_data[leg][data_type]
                
                # Ensure data is 1D
                mean_data = np.ravel(data['mean'])
                std_data = np.ravel(data['std'])
                
                # Plot mean and standard deviation
                ax.fill_between(
                    np.arange(101),
                    -1 * (mean_data - std_data),
                    -1 * (mean_data + std_data),
                    color='red', alpha=0.2
                )
                ax.plot(-1 * mean_data, 'red', linewidth=2)
                
                # Add stance phase visualization
                if stance_phases[leg] is not None:
                    ax.axvline(stance_phases[leg][1], color='gray', linestyle='--', alpha=0.5)
                    ax.axvspan(0, stance_phases[leg][1], alpha=0.2, color='gray', label='Stance')
                    ax.axvspan(stance_phases[leg][1], 100, alpha=0.2, color='white', label='Swing')
                
                # Add labels and formatting
                ax.set_xlabel('Gait Cycle (%)')
                ax.set_ylabel(ylabel)
                ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add padding to y-limits for all plots
        for ax in [ax1, ax2] + axes:
            ymin, ymax = ax.get_ylim()
            y_range = ymax - ymin
            padding = y_range * 0.05
            ax.set_ylim(ymin - padding, ymax + padding)
        
        fig.suptitle('Optimized Torque Profile', fontsize=18, fontweight='bold')
        return fig

    def createExoReport(self, unpacked_dict):
        """Create exoskeleton report figures."""
        if not hasattr(self, '_analysis_strides'):
            self._stride_info = self._detect_strides(unpacked_dict)
            self._analysis_strides = self._get_analysis_strides(self._stride_info)
        
        # Get exo data from unpacked_dict
        exo_data = {
            'Exo_L': unpacked_dict['actuator_data']['Exo_L'],
            'Exo_R': unpacked_dict['actuator_data']['Exo_R']
        }
        
        # Process and normalize exo data
        exo_norm_data = {}
        stance_phases = {}
        
        for leg, exo_key in zip(['l_leg', 'r_leg'], ['Exo_L', 'Exo_R']):
            if self._analysis_strides[leg]['num_strides'] > 0:
                exo_norm_data[leg] = {}
                
                # Calculate stance phase
                stance_phase, _, _ = self._calculate_gait_phases(
                    self._analysis_strides[leg],
                    unpacked_dict[leg]['load_ipsi']
                )
                stance_phases[leg] = stance_phase
                
                # Normalize force and control data
                for data_type in ['force', 'ctrl']:
                    raw_data = exo_data[exo_key][data_type]
                    norm_data = self.normToGaitCycle(raw_data, self._analysis_strides[leg])
                    
                    exo_norm_data[leg][data_type] = {
                        'mean': np.mean(norm_data, axis=0),
                        'std': np.std(norm_data, axis=0)
                    }
        
        # Create all plots on one figure
        fig = self.plotExoReport(exo_data, self._stride_info, exo_norm_data, stance_phases)
        
        return fig

    def plot_cma_training(self, es):
        """Plot the optimization results from CMA-ES using built-in plotting."""

        fig = plt.figure(figsize=(12, 8)) 
        
        es.logger.plot(fig=fig)
        
        # Save the plot with trial name in the same directory as other files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f'{trial_name}_cmaes_optimization_results.png'
        plt.savefig(os.path.join(save_path, plot_filename), 
                    dpi=150,
                    bbox_inches='tight')
        plt.close()

    def plot_cost_distribution(self, es, save_path, trial_name):
        """Plot cost distribution and best cost trend."""
        fig = plt.figure(figsize=(15, 10))
        
        # Get data from CMA-ES logger
        all_f = es.logger.data['f']
        n_iterations = len(all_f)
        pop_size = es.popsize
        
        # For each generation, plot all population members
        for i, gen_costs in enumerate(all_f):
            pop_costs = gen_costs[4:4+pop_size]  # Get population costs for this generation
            plt.scatter([i] * len(pop_costs), pop_costs, 
                    c='lightgray', alpha=0.5, s=20)
        
        # Add line for minimum cost trend
        min_costs = [min(gen_costs[4:4+pop_size]) for gen_costs in all_f]
        plt.plot(range(len(min_costs)), min_costs, 'r-', linewidth=2, label='Best Cost')
        
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Cost", fontsize=12)
        plt.yscale('log')
        plt.grid(True, which="major", ls="-", alpha=0.2)
        plt.grid(True, which="minor", ls=":", alpha=0.1)
        plt.legend(fontsize=10)
        plt.title(f"Cost Distribution per Iteration (Population Size: {pop_size})", fontsize=14)
        
        plt.tight_layout()
        plot_filename = f'{trial_name}_cost_distribution.png'
        plt.savefig(os.path.join(save_path, plot_filename),
                    dpi=300,
                    bbox_inches='tight')
        plt.close()

    def plot_simulation_categories(self, es, save_path, trial_name):
        """Plot pie chart of simulation categories."""
        fig = plt.figure(figsize=(12, 12))
        
        # Get costs from the main population only
        all_f = es.logger.data['f']
        pop_size = es.popsize
        
        # Extract only the population costs from each generation (starting at index 4)
        main_costs = []
        for gen_costs in all_f:
            # ONLY take exactly pop_size values, not pop_size+1
            if len(gen_costs) >= (4 + pop_size):
                pop_costs = gen_costs[4:4+pop_size]  # Exactly pop_size values
                main_costs.extend(pop_costs)
        
        main_costs = np.array(main_costs)
        
        print("\nMain population statistics:")
        print(f"Total evaluations: {len(main_costs)} (expected: {pop_size * len(all_f)})")
        print(f"Cost range - Min: {np.min(main_costs):.4f}, Max: {np.max(main_costs):.4f}")
                    
        error_mask = main_costs >= 120*10000      # Simulation errors (1.2M)
        fall_mask = main_costs >= 95000           # Early termination (~99995)
        fall_mask = fall_mask & ~error_mask       # Exclude error cases from fall_mask
        other_mask = main_costs < 95000           # Should be empty since all non-passing cases should be ≥95000

        print("\nCost statistics:")
        print(f"Min cost: {np.min(main_costs):.4f}")
        print(f"Max cost: {np.max(main_costs):.4f}")
        print(f"Error cases: {np.sum(error_mask)}")
        print(f"Early termination: {np.sum(fall_mask)}")
        print(f"Other: {np.sum(other_mask)} (should be 0)")
        
        categories = ['Simulation Errors', 'Early Termination', 'Other']
        sizes = [np.sum(error_mask), np.sum(fall_mask), np.sum(other_mask)]
        colors = ['#ff9999', '#ffcc99', '#99ff99']
        
        # Only show categories that have non-zero values
        non_zero_idx = [i for i, size in enumerate(sizes) if size > 0]
        categories = [categories[i] for i in non_zero_idx]
        sizes = [sizes[i] for i in non_zero_idx]
        colors = [colors[i] for i in non_zero_idx]
        
        plt.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%')
        plt.title("Simulation Categories Distribution\n" + 
                f"(Main Population Only, {pop_size} per generation)\n" +
                f"Best Cost: {es.result.fbest:.2f}", fontsize=14, pad=20)
        
        plt.tight_layout()
        plot_filename = f'{trial_name}_simulation_categories.png'
        plt.savefig(os.path.join(save_path, plot_filename),
                    dpi=150,
                    bbox_inches='tight')
        plt.close()

    def create_combined_plot(self, es, save_path, trial_name):
        """Create combined plot by overlaying custom plots on CMA-ES standard output."""
        
        # Debug: Print CMA-ES solution info directly
        print("\nCMA-ES Solution Debug:")
        print(f"Best solution cost (es.result.fbest): {es.result.fbest}")
        
        # Create main figure with CMA plots
        fig = plt.figure(figsize=(12, 8))
        es.logger.plot(fig=fig)
        
        # Create white rectangle to cover right side
        rect = plt.Rectangle(
            (0.65, 0.0),     # Bottom left corner
            0.35, 1,       # Width, full column height
            facecolor='white',
            edgecolor='white',
            transform=fig.transFigure,
            zorder=2,
            alpha=1.0
        )
        fig.patches.extend([rect])
        plt.draw()

        # Create axes for our custom plots
        ax_pie = fig.add_axes([0.67, 0.55, 0.30, 0.40], zorder=3)    
        ax_cost = fig.add_axes([0.67, 0.075, 0.30, 0.40], zorder=3)  
        
        ax_pie.set_facecolor('white')
        ax_cost.set_facecolor('white')
        
        # Get data from CMA-ES logger
        all_f = es.logger.data['f']
        pop_size = es.popsize
        
        # Print debug information
        print(f"\nNumber of generations: {len(all_f)}")
        print(f"Population size: {pop_size}")
        
        # Extract population costs - simple approach
        main_costs = []
        for gen_idx, gen_costs in enumerate(all_f):
            if len(gen_costs) >= (4 + pop_size):
                # Take population costs (first pop_size values starting at index 4)
                pop_costs = gen_costs[4:4+pop_size]
                # Use a heuristic: in each generation, keep the top pop_size-1 highest values
                # This assumes the highest values are the actual fitness costs, not metrics
                sorted_costs = sorted(pop_costs, reverse=True)
                actual_costs = sorted_costs[:pop_size-1]
                
                if gen_idx < 3:  # Debug first few generations
                    print(f"Generation {gen_idx} raw costs: {pop_costs}")
                    print(f"Generation {gen_idx} selected costs: {actual_costs}")
                
                main_costs.extend(actual_costs)
        
        main_costs = np.array(main_costs)
        
        # If we have costs, proceed with categorization and plotting
        if len(main_costs) > 0:
            # Create pie chart
            error_mask = main_costs >= 120*10000      # Simulation errors (1.2M)
            fall_mask = (main_costs >= 95000) & (main_costs < 120*10000)  # Early termination (~99995)
            other_mask = main_costs < 95000           # Successful cases
            
            categories = ['Simulation Errors', 'Early Termination', 'Successful']
            sizes = [np.sum(error_mask), np.sum(fall_mask), np.sum(other_mask)]
            colors = ['#ff9999', '#ffcc99', '#99ff99']
            
            print("\nCategory breakdown:")
            print(f"Total costs analyzed: {len(main_costs)}")
            print(f"Cost range: {np.min(main_costs):.4f} to {np.max(main_costs):.4f}")
            print(f"Simulation Errors: {np.sum(error_mask)} ({np.sum(error_mask)/len(main_costs)*100:.1f}%)")
            print(f"Early Termination: {np.sum(fall_mask)} ({np.sum(fall_mask)/len(main_costs)*100:.1f}%)")
            print(f"Successful: {np.sum(other_mask)} ({np.sum(other_mask)/len(main_costs)*100:.1f}%)")
            
            # Only show non-zero categories
            non_zero_idx = [i for i, size in enumerate(sizes) if size > 0]
            categories = [categories[i] for i in non_zero_idx]
            sizes = [sizes[i] for i in non_zero_idx]
            colors = [colors[i] for i in non_zero_idx]
            
            if sizes:  # Only create pie chart if we have data
                ax_pie.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%')
                ax_pie.set_title("Simulation Categories Distribution")
        else:
            print("WARNING: No costs found to analyze")
            ax_pie.text(0.5, 0.5, "No data to display", 
                     ha='center', va='center', fontsize=12)
        
        # Plot cost distribution - brute force approach
        all_iterations = []
        all_valid_costs = []
        
        for i, gen_costs in enumerate(all_f):
            if len(gen_costs) >= (4 + pop_size):
                # Use the same heuristic as above
                pop_costs = gen_costs[4:4+pop_size]
                sorted_costs = sorted(pop_costs, reverse=True)
                actual_costs = sorted_costs[:pop_size-1]
                
                all_iterations.extend([i] * len(actual_costs))
                all_valid_costs.extend(actual_costs)
        
        # Plot scatter points if we have data
        if all_valid_costs:
            ax_cost.scatter(all_iterations, all_valid_costs, 
                         c='lightgray', alpha=0.5, s=20, label='Population Costs')
            
            # Add best cost line
            min_costs = []
            for i, gen_costs in enumerate(all_f):
                if len(gen_costs) >= (4 + pop_size):
                    pop_costs = gen_costs[4:4+pop_size]
                    sorted_costs = sorted(pop_costs, reverse=True)
                    actual_costs = sorted_costs[:pop_size-1]
                    if actual_costs:
                        min_costs.append(min(actual_costs))
            
            if min_costs:
                ax_cost.plot(range(len(min_costs)), min_costs, 'r-', linewidth=2, label='Best Cost')
                
            ax_cost.set_yscale('log')
            ax_cost.grid(True)
            ax_cost.set_title("Cost Distribution")
            
            # Make sure we have labels before calling legend
            handles, labels = ax_cost.get_legend_handles_labels()
            if handles:
                ax_cost.legend()
            
            ax_cost.set_xlabel("Iterations")
            
            # Ensure we have a reasonable x-axis range
            ax_cost.set_xlim(0, len(all_f))
        else:
            print("WARNING: No costs to plot in cost distribution")
            ax_cost.text(0.5, 0.5, "No data to display", 
                      ha='center', va='center', fontsize=12)
        
        # Add metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        param_text = (f"Population Size: {pop_size}\n"
                     f"Best Cost: {es.result.fbest:.2f}\n"
                     f"Timestamp: {timestamp}")
        fig.text(0.995, 0.01, param_text, 
                 fontsize=8, 
                 ha='right', 
                 va='bottom',
                 bbox=dict(facecolor='white', 
                          edgecolor='none', 
                          alpha=0.8))
        
        # Save plots
        plot_filename = f'{trial_name}_combined_results.png'
        plt.savefig(os.path.join(save_path, plot_filename),
                    dpi=300,
                    bbox_inches='tight')
        
        pdf_filename = f'{trial_name}_optimization_results.pdf'
        plt.savefig(os.path.join(save_path, pdf_filename),
                    format='pdf',
                    bbox_inches='tight')
        plt.close()

    def saveToPDF(self, unpacked_dict, muscle_labels, ref_angle, ref_emg, metadata, 
                savepath='reports', filename=None):
        """Generate and save complete PDF report.
        
        Args:
            unpacked_dict: Dictionary containing processed simulation data
            muscle_labels: Labels for each muscle
            ref_angle: Reference angle data
            ref_emg: Reference EMG data
            metadata: Dictionary of simulation metadata
            savepath: Directory to save PDF (default: 'reports')
            filename: Base filename from JSON (without extension)
        """
        # Create reports directory if it doesn't exist
        os.makedirs(savepath, exist_ok=True)
        
        # Generate base filename if not provided
        if filename is None:
            filename = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            filename = filename.replace('_data.json', '')
        
        # Construct final filename with MyoAssist prefix
        pdf_filename = f"MyoAssist_{filename}_report.pdf"
        pdf_path = os.path.join(savepath, pdf_filename)
        
        # Detect strides and compute analysis strides
        self._stride_info = self._detect_strides(unpacked_dict)
        self._analysis_strides = self._get_analysis_strides(self._stride_info)
        
        # Generate all figures
        plt.ioff()
        figures = []
        
        # Add metadata page
        figures.append(self._create_metadata_page(metadata))
        
        # Add angle reports
        figures.append(self.createAngleReport(unpacked_dict, ref_angle, leg_arg='l_leg'))
        figures.append(self.createAngleReport(unpacked_dict, ref_angle, leg_arg='r_leg'))
        
        # Add muscle reports
        for leg in ['l_leg', 'r_leg']:
            act_fig, fv_fig = self.createMuscleReport(
                unpacked_dict, muscle_labels, ref_emg, leg_arg=leg
            )
            figures.extend([act_fig, fv_fig])
        
        # Add exoskeleton report (now single page)
        figures.append(self.createExoReport(unpacked_dict))
        
        # Save all figures to PDF
        with PdfPages(pdf_path) as pdf:
            for fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        # Clear cached stride information
        if hasattr(self, '_stride_info'):
            del self._stride_info
        if hasattr(self, '_analysis_strides'):
            del self._analysis_strides
        
        print(f'Report saved to: {pdf_path}')
        
    def generate_reports(self, prefix, data_dir="visualization_outputs/data", 
                        savepath="reports", muscle_labels=None):
        """
        Generate reports for all matching simulation files.
        
        Args:
            prefix: Prefix for JSON files to process
            data_dir: Directory containing simulation data
            savepath: Directory to save PDF reports
            muscle_labels: Custom muscle labels (uses default if None)
        """
        if muscle_labels is None:
            muscle_labels = self.default_muscle_labels

        processed_data = self.load_simulation_data(prefix, data_dir)
        
        for sim_data in processed_data:
            self.saveToPDF(
                sim_data['unpacked_dict'],
                muscle_labels,
                self.ref_kinematics,
                self.ref_emg,
                sim_data['metadata'],
                savepath=savepath,
                filename=sim_data['filename']
            )