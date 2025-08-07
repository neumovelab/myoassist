---
title: Evaluating Results
parent: Controller Optimization
nav_order: 6
layout: home
---

# Evaluating Results

This guide explains how to use the evaluation pipeline located in `ctrl_optim/results/evaluation/` to analyze and visualize the results of your optimization runs.

## Overview

The evaluation module provides two primary ways to analyze results:
1. **A Graphical User Interface (GUI)** for interactive, single-run analysis or batch processing multiple results folders (Windows/Linux only).
2. **A Command-Line Interface** that uses a JSON file for automated, scriptable processing or batch processing (all platforms).

The primary output for both methods includes simulation videos, kinematics plots, and detailed analysis reports.

## Quick Start

### Using `run_eval.py`

The evaluation framework uses a unified approach with `run_eval.py` as the main entry point:

```bash
# Navigate to the ctrl_optim directory
cd ctrl_optim

# Run the evaluation pipeline
python run_eval.py
```

This will launch the GUI interface (on Windows/Linux) or provide command-line options.

## Method 1: Using the GUI (Windows/Linux Only)

**Note: The GUI interface does not work on macOS due to tkinter compatibility issues. Mac users should use Method 2 (Command-Line) instead.**

### Launching the GUI

```bash
cd ctrl_optim
python run_eval.py
```

This will launch the "Controller Optimization Evaluation Pipeline" window.

### GUI Options

The GUI is divided into sections that allow you to configure how the results are evaluated.

#### 1. Select Results Folder(s)

This is the primary input for the tool.
- **Add Folder(s)**: Opens a dialog to select one or more results folders (e.g., `results/optim_results/baseline_date_time`). You can select multiple folders for batch processing.
- **Clear**: Removes all selected folders from the list.

When a folder is selected, the tool automatically finds the associated configuration file (`.bat` or `.sh`) and the `_Best.txt` or `_BestLast.txt` parameter files.

#### 2. Environment Configuration

This section displays the loaded environment configuration. You may also override the environment settings that were used during the original optimization if desired. **Note**: Overriding environment settings is *not* recommended, as the optimized results are environment dependent, but this feature can be used to swap models or test other environments with your optimized parameters.

- **Model**: The musculoskeletal model to use (e.g., `tutorial`, `baseline`, `dephy`)
- **Mode**: 2D or 3D simulation mode
- **Slope**: Terrain slope in degrees
- **Max Torque**: Maximum exoskeleton torque
- **Init Pose**: Initial walking pose
- **Boolean Options**: Delayed controller, exoskeleton on/off, fixed exo profile, 4-parameter spline

#### 3. Parameter Types to Evaluate

Choose which parameter files to evaluate:
- **Best**: Evaluate the best parameters found during optimization (`_Best.txt`)
- **BestLast**: Evaluate the best parameters from the final population (`_BestLast.txt`)

#### 4. Evaluation Mode

This defines the level of detail in the output:
- **Short**: 5-second simulation with video and kinematics plot
- **Long**: 10-second simulation with video and kinematics plot

#### 5. Output Directory

Specify where the evaluation results will be saved. Results are automatically organized in timestamped folders.

## Method 2: Command-Line with JSON (All Platforms)

For automated workflows or macOS users, you can run the evaluation script from the command line using a JSON configuration file.

### Running from the Command Line

```bash
cd ctrl_optim
python run_eval.py --config your_config.json
```

### JSON Configuration File

Create a `.json` file to specify the evaluation parameters. See `ctrl_optim/results/evaluation/eval_config/example_config.json` for a template.

**Example `config.json`:**
```json
{
    "results_dir": "results/optim_results/baseline_0701_1200",
    "evaluation_mode": "short",
    "output_dir": "results/evaluation_outputs",
    "include_best": true,
    "include_bestlast": true
}
```

**Example batch processing with multiple directories:**
```json
{
    "results_dirs": [
        "results/optim_results/baseline_0701_1200",
        "results/optim_results/exo_4param_0702_1400",
        "results/optim_results/exo_npoint_0703_1600"
    ],
    "evaluation_mode": "short",
    "output_dir": "results/evaluation_outputs",
    "include_best": true,
    "include_bestlast": false
}
```

### Configuration Parameters

- **`results_dir`** or **`results_dirs`**: Path to one or more results directories
- **`evaluation_mode`**: Analysis mode (`"short"` or `"long"`)
- **`output_dir`**: Directory where evaluation outputs will be saved
- **`include_best`**: Whether to evaluate `_Best.txt` files (default: `true`)
- **`include_bestlast`**: Whether to evaluate `_BestLast.txt` files (default: `true`)

## Output Structure

Each evaluation run creates a timestamped output directory:

```
ctrl_optim/results/evaluation_outputs/MMDD_HHMM/
├── parameter_name_001.mp4              # Simulation video
├── parameter_name_001_kinematics.png   # Kinematics plot
├── parameter_name_001_stats.txt        # Kinematic statistics
├── parameter_name_001_exo.mp4          # Exoskeleton video (if applicable)
├── parameter_name_001_exo_cost.png     # Exoskeleton controller + cost plot (if applicable)
└── config_name_timestamp.bat           # Configuration file copy
```

## Platform Compatibility

### Windows/Linux
- GUI interface available
- Command-line interface available
- All features supported

### macOS
- GUI interface NOT supported (tkinter compatibility issues)
- Command-line interface available
- All evaluation features supported via JSON configuration

## Troubleshooting

### Common Issues

1. **"Module not found" errors**: Make sure you're running from the correct directory:
   ```bash
   cd ctrl_optim
   python run_eval.py
   ```

2. **GUI not working**: Use the command-line interface instead:
   ```bash
   python run_eval.py --config example_config.json
   ```

3. **No parameter files found**: Verify the results directory contains `_Best.txt` or `_BestLast.txt` files

4. **Video generation fails**: Ensure required dependencies are installed:
   ```bash
   pip install imageio sk-video
   ```

### Creating Custom Configurations

You can create custom evaluation configurations by:

1. Copying the example config file:
   ```bash
   cp ctrl_optim/results/evaluation/eval_config/example_config.json my_config.json
   ```

2. Modifying the parameters as needed

3. Running with your custom config:
   ```bash
   python run_eval.py --config my_config.json
   ```

## Quick Visualization

For simple video generation without detailed analysis, you can also use the simulation scripts:

- **`run_ctrl_minimal.py`**: Quick testing with random parameters
- **`run_ctrl.py`**: Detailed simulation with video generation

See the **[Running Reflex Control](Running_Reflex_Control)** guide for more details on these scripts. 