#!/usr/bin/env python3
"""
This script runs a single simulation with the reflex controller.
"""

import os
import sys
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import platform
import glob
from pathlib import Path

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the necessary modules
from ctrl.reflex.reflex_interface import myoLeg_reflex
from optim.optim_utils.config_parser import load_params_and_create_testenv, print_config_summary


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """Print progress bar - copied from eval.py"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total:
        print()


def open_video_in_new_window(video_path):
    """Open video file in a new window using the default system video player."""
    try:
        if platform.system() == "Windows":
            os.startfile(video_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", video_path])
        else:  # Linux
            subprocess.run(["xdg-open", video_path])
        print(f"Video opened in new window: {os.path.basename(video_path)}")
    except Exception as e:
        print(f"Could not open video automatically: {e}")
        print(f"Video saved to: {video_path}")


def find_config_file(results_dir):
    """Find configuration file (.bat or .sh) in the results directory"""
    # Look for both .bat and .sh files
    config_files = []
    for ext in ["*.bat", "*.sh"]:
        config_files.extend(glob.glob(os.path.join(results_dir, ext)))
    
    if not config_files:
        raise FileNotFoundError(f"No configuration file (.bat or .sh) found in directory: {results_dir}")
    
    # Return the first found configuration file
    return config_files[0]


def main():
    """Main function to run the simulation."""
    
    # --- Load from Optimization Results ---
    LOAD_FROM_FILE = True
    notebook_dir = os.getcwd()
    PARAMS_FILE_PATH = os.path.join(notebook_dir, "results", "optim_results", "exo_npoint_tutorial", "myorfl_Kine_2D_1_25_2025Jul25_1827_None_BestLast.txt")
    
    if LOAD_FROM_FILE:
        if not os.path.exists(PARAMS_FILE_PATH):
            raise FileNotFoundError(f"Parameter file not found: {PARAMS_FILE_PATH}")

    # --- Manual Configuration Settings ---
    SIMULATION_TIME = 5      # seconds
    SLOPE_DEG = 0             # env slope degrees
    MODEL = "tutorial"           # Options: dephy, hmedi, humotech, osl, baseline
    EXO_BOOL = False           # Enable or disable exoskeleton
    USE_4PARAM_SPLINE = False # Use 4-parameter spline for exoskeleton
    N_POINTS = 4              # Number of points for n-point spline
    MAX_TORQUE = 100          # Maximum exoskeleton torque

    # --- Output Directory Setup ---
    output_folder = "results/evaluation_outputs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_folder = os.path.join(output_folder, f"run_ctrl_{timestamp}")
    os.makedirs(run_output_folder)
    print(f"Outputs will be saved to: {run_output_folder}")

    # --- Environment Initialization ---
    if LOAD_FROM_FILE:
        if not os.path.exists(PARAMS_FILE_PATH):
            raise FileNotFoundError(f"Parameter file not found: {PARAMS_FILE_PATH}")
        
        results_dir = os.path.dirname(PARAMS_FILE_PATH)
        filename = os.path.basename(PARAMS_FILE_PATH)
        
        # Find configuration file (.bat or .sh)
        config_file_path = find_config_file(results_dir)
        
        env, config, _ = load_params_and_create_testenv(
            results_dir=results_dir,
            filename=filename,
            bat_file_path=config_file_path,
            sim_time=SIMULATION_TIME
        )
        print_config_summary(config, title="Loaded Configuration")
        
    else:
        # --- Uses the Manual Settings from previous cell ---
        config = {
            'mode': '2D', 'init_pose': 'walk_left', 'delayed': False,
            'slope_deg': SLOPE_DEG,
            'model': MODEL,
            'exo_bool': EXO_BOOL,
            'use_4param_spline': USE_4PARAM_SPLINE,
            'n_points': N_POINTS,
            'max_torque': MAX_TORQUE
        }
        
        if config['exo_bool']:
            spline_params = 4 if config['use_4param_spline'] else (config['n_points'] * 2)
        else:
            spline_params = 0
        control_params = np.ones(77 + spline_params)
        
        env = myoLeg_reflex(
            sim_time=SIMULATION_TIME,
            control_params=control_params,
            **config
        )
        print_config_summary(config, title="Manual Configuration")

    env.reset()
    print("\nEnvironment initialized.")

    # --- Simulation and Video Generation ---
    env.reset()
    
    # Calculate timesteps
    timesteps = int(SIMULATION_TIME / env.dt)
    frames = []
    
    print(f"Running {timesteps} timesteps...")
    
    # Set up renderer with higher resolution - EXACTLY matching eval.py
    video_width, video_height = 1920, 1080
    env.env.sim.renderer.render_offscreen(camera_id=4, width=video_width, height=video_height)
    env.env.sim.renderer._scene_option.flags[0] = 0  # Remove convex hull
    env.env.sim.renderer._scene_option.flags[4] = 0
    
    # Camera setup for video with increased resolution - EXACTLY matching eval.py
    free_cam = mujoco.MjvCamera()
    camera_speed = 1.25
    slope_angle_rad = np.radians(env.slope_deg)
    start_position = env.env.unwrapped.sim.data.body("pelvis").xpos.copy()
    
    camera_pos = start_position.copy()
    camera_pos[2] = 0.8
    
    for i in range(timesteps):
        # Show progress bar every 10%
        if i % max(1, timesteps // 10) == 0 or i % 50 == 0:
            print_progress_bar(i, timesteps, prefix='Progress:', suffix=f'({i}/{timesteps})', length=30)
        
        # Update camera position for following - EXACTLY matching eval.py
        if not env.delayed:
            distance_traveled = camera_speed * env.dt * i
            camera_pos[0] = start_position[0] + distance_traveled
            
            slope_correction = 0.2
            height_increase = (camera_pos[0] - start_position[0]) * np.tan(slope_angle_rad) * slope_correction
            camera_pos[2] = 0.8 + height_increase
            
            pelvis_pos = env.env.unwrapped.sim.data.body("pelvis").xpos.copy()
            lookat_pos = camera_pos.copy()
            lookat_pos[1] = pelvis_pos[1]
            
            free_cam.distance = 2.5
            free_cam.azimuth = 90
            free_cam.elevation = 0
            free_cam.lookat = lookat_pos
            
            frame = env.env.unwrapped.sim.renderer.render_offscreen(
                    camera_id=free_cam, width=video_width, height=video_height)
        else:
            if i % 10 == 0:
                env.env.sim.data.camera(4).xpos[2] = 2.181
                
            frame = env.env.sim.renderer.render_offscreen(camera_id=4, width=video_width, height=video_height)
        
        # Handle frame resizing if needed (same as eval.py)
        if frame is not None and len(frame.shape) == 3:
            actual_height, actual_width = frame.shape[:2]
            
            # Report dimensions for first frame
            if i == 0:
                if actual_width != video_width or actual_height != video_height:
                    video_width, video_height = actual_width, actual_height
        
        frames.append(frame)
        
        # Run simulation step
        _, _, is_done = env.run_reflex_step_Cost()
        
        if is_done:
            print(f"Simulation terminated early at timestep {i}")
            break
    
    print_progress_bar(timesteps, timesteps, prefix='Progress:', suffix=f'({timesteps}/{timesteps})', length=30)
    
    # Save regular video using skvideo
    video_filename = "simulation_regular.mp4"
    video_path = os.path.join(run_output_folder, video_filename)
    
    try:
        import skvideo.io
        skvideo.io.vwrite(video_path, 
                        np.asarray(frames),
                        inputdict={"-r": "100"}, 
                        outputdict={"-r": "100", "-pix_fmt": "yuv420p"})
        print(f"Video saved: {os.path.basename(video_path)} ({video_width}x{video_height})")
    except Exception as e:
        print(f"Error: Video generation failed: {e}")
    
    # Open video in new window
    open_video_in_new_window(video_path)
    
    print(f"\nAll outputs saved to: {run_output_folder}")
    print("Simulation completed successfully!")


if __name__ == "__main__":
    main()