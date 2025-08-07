---
title: Processing Results
parent: Control & Optimization
nav_order: 6
layout: home
---

# Processing Results

This guide explains how to use the processing pipeline located in `myoassist_reflex/processing/` to analyze and visualize the results of your optimization runs.

## Overview

The processing module provides two primary ways to analyze results:
1.  **A Graphical User Interface (GUI)** for interactive, single-run analysis or for batch processing multiple results folders.
2.  **A Command-Line Interface** that uses a JSON file for automated, scriptable processing or batch processing.

The primary output for both methods is a detailed report, including videos and summary plots, for each processed simulation.

## Method 1: Using the GUI

The most straightforward way to process results is with the interactive GUI.

### Launching the GUI

Navigate to the project's root directory and run the `processing` module as a script:

```bash
python -m myoassist_reflex.processing
```

This will launch the "MyoReflex Processing Pipeline" window.

### GUI Options

The GUI is divided into sections that allow you to configure how the results are processed.

#### 1. Select Results Folder(s)

This is the primary input for the tool.
- **Add Folder(s)**: Opens a dialog to select one or more results folders (e.g., `results/baseline_0701_1200`). You can select multiple folders for batch processing.
- **Clear**: Removes all selected folders from the list.

When a folder is selected, the tool automatically tries to find the associated `.bat` configuration file and the `_Best.txt` or `_BestLast.txt` parameter file.

#### 2. Set Environment

This section allows you to override the environment settings that were used during the original optimization.

- **Use settings from config file**: (Default) If checked, the simulation will run using the exact parameters found in the `.bat` file associated with the results folder.
- **Manual Settings**: If unchecked, you can manually set the environment parameters (`Model`, `Slope`, `Exo`, etc.). This is useful for testing how a controller performs in a different environment than the one it was optimized for.

#### 3. Set Processing Mode

This defines the level of detail in the output report.
- **Quick**: Generates a simulation video and a basic kinematics plot.
- **Full**: Generates a comprehensive report using `MyoReport`, which includes detailed plots for kinematics, GRF, muscle activations, and more. This is more time-consuming.
- **Debug**: A minimal mode for testing the processing pipeline itself.

#### 4. Run Processing
- **Start Processing**: Begins the analysis. For each selected results folder, it will run a new simulation with the specified parameters and generate the report in the `processing_outputs` directory.
- **Cancel**: Closes the GUI.

## Method 2: Command-Line with JSON

For automated workflows, you can run the processing script from the command line by providing a JSON configuration file.

### Running from the Command Line

```bash
python -m myoassist_reflex.processing --config path/to/your_config.json
```

### JSON Configuration File

Create a `.json` file to specify the processing parameters. See `myoassist_reflex/processing/example_config.json` for a template.

**Example `config.json`:**
```json
{
    "results_dir": "results/baseline_0701_1200",
    "processing_mode": "full",
    "output_dir": "custom_outputs"
}
```

- **`results_dir`**: Path to a single results directory. For batch processing multiple directories, you can create a simple shell script to loop over your folders and call the processor for each one.
- **`processing_mode`**: The analysis mode (`"quick"`, `"full"`, or `"debug"`).
- **`output_dir`**: (Optional) The directory where the output reports will be saved. Defaults to `processing_outputs`.

## A Note on Visualizing Single Runs

While this processing pipeline is powerful for generating detailed reports, sometimes you just want to quickly watch a video of a specific controller. For this purpose, the **`sample_run.ipynb`** notebook is the ideal tool.

Please refer to the **[Running a Simulation](Running_Simulations)** guide for detailed instructions on how to use the notebook to load any parameter file and generate a video. 
