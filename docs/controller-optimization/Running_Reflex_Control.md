---
title: Running Reflex Control
parent: Controller Optimization
nav_order: 1
layout: home
---

# Running Reflex Control

This document explains how to run the reflex controller using the MyoAssist framework. There are two main scripts available:

1. **`run_ctrl_minimal.py`**: A simple script for running a reflex control simulation with random parameters
2. **`run_ctrl.py`**: A more in-depth script for reflex control simulation with video generation

## Running `run_ctrl_minimal.py`

### Usage

```bash
cd ctrl_optim
python run_ctrl_minimal.py
```

### What it does

- Creates random control parameters (77 parameters for 2D reflex controller)
- Runs a 5-second simulation with default settings
- Reports walking duration
- No video generation or file outputs

Walking duration will vary each time `run_ctrl_minimal.py` is rerun since the parameter vector is randomized.

### Configuration

The script uses these default settings:
- **Simulation time**: 5 seconds
- **Model**: `tutorial` (2D model)
- **Initial pose**: `walk_left`
- **Slope**: 0 degrees (flat ground)
- **Exoskeleton**: Disabled
- **Control parameters**: Random normal distribution

### Output

The script prints the walking duration to the console:
```
Walking duration: 0.35 seconds
```

## Running `run_ctrl.py`

### Usage

```bash
cd ctrl_optim
python run_ctrl.py
```

### Method 1: Loading from Optimization Results

To visualize a controller that you have already optimized:

```python
# In run_ctrl.py, set:
LOAD_FROM_FILE = True
PARAMS_FILE_PATH = "results/optim_results/results_folder/parameters.txt"
```

The script will automatically:
- Load the optimized parameters from the `.txt` file
- Find the corresponding configuration file in the same directory
- Reconstruct the environment used during optimization

### Method 2: Manual Configuration

To create a new simulation from scratch:

```python
# In run_ctrl.py, set:
LOAD_FROM_FILE = False

# Manual settings:
SIMULATION_TIME = 5      # seconds
SLOPE_DEG = 0           # terrain slope in degrees
MODEL = "tutorial"       # Options: tutorial, dephy, hmedi, humotech, osl, baseline
EXO_BOOL = False        # Enable/disable exoskeleton
USE_4PARAM_SPLINE = False # Use 4-parameter spline for exoskeleton
N_POINTS = 4            # Number of points for n-point spline
MAX_TORQUE = 100        # Maximum exoskeleton torque
```

### Environment Initialization

The part of the script handles environment creation based on your configuration:

```python
if LOAD_FROM_FILE:
    # Load from optimization results
    env, config, _ = load_params_and_create_testenv(
        results_dir=results_dir,
        filename=filename,
        bat_file_path=bat_file_path,
        sim_time=SIMULATION_TIME
    )
    print_config_summary(config, title="Loaded Configuration")
    
else:
    # Use manual settings
    config = {
        'mode': '2D', 'init_pose': 'walk_left', 'delayed': False,
        'slope_deg': SLOPE_DEG, 'model': MODEL, 'exo_bool': EXO_BOOL,
        'use_4param_spline': USE_4PARAM_SPLINE, 'n_points': N_POINTS,
        'max_torque': MAX_TORQUE
    }
    
    # Calculate control parameters
    if config['exo_bool']:
        spline_params = 4 if config['use_4param_spline'] else (config['n_points'] * 2)
    else:
        spline_params = 0
    control_params = np.ones(77 + spline_params)
    
    env = myoLeg_reflex(sim_time=SIMULATION_TIME, control_params=control_params, **config)
    print_config_summary(config, title="Manual Configuration")
```

### Simulation and Video Generation

The script runs the simulation with frame recording for video creation:

1. **Camera Setup**: Configures a free camera that follows the model's movement
2. **High-Resolution Rendering**: 1920x1080 resolution at 100 FPS
3. **Progress Tracking**: Shows progress bar during simulation
4. **Frame Collection**: Captures each frame for video compilation

### Example Output

```
Outputs will be saved to: results/evaluation_outputs/run_ctrl_date_time
Running 500 timesteps...
Progress: |████████████████████████████████████████████████████| 100.0% (500/500)
Video saved: simulation_regular.mp4 (1920x1080)
Video opened in new window: simulation_regular.mp4
All outputs saved to: results/evaluation_outputs/run_ctrl_date_time
Simulation completed successfully!
```

## Troubleshooting

### Common Issues

1. **Video not generating**: Ensure `skvideo.io` is installed:
   ```bash
   pip install sk-video
   ```

2. **Model not found**: Check that the specified model exists in the `models/` directory

3. **Parameter file not found**: Verify the path to your optimization results file

4. **Camera issues**: The script automatically handles camera positioning, but you can modify the camera behavior in the script if needed

### Performance Notes

- For faster rendering, reduce the video resolution in the script or adjust `SIMULATION_TIME`
- The script automatically terminates early if the model falls or fails 