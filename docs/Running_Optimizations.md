# Running Optimizations

This guide explains how to run optimizations for the neuromuscular reflex controller using the `train.py` script, covering everything from basic configurations to advanced settings.

## Quick Start

The easiest way to begin is by using the `run_training.bat` script, which executes predefined configurations from the `training_configs/` directory.

```bash
run_training.bat <config_name>
```

For example, to run the baseline configuration:
```bash
run_training.bat baseline
```

## Training Configurations

The `training_configs/` directory contains several `.bat` files, each pre-configured for a specific optimization scenario.

| Configuration         | Description                                                                                             |
|-----------------------|---------------------------------------------------------------------------------------------------------|
| `baseline.bat`        | A standard optimization for the 11-muscle model without an exoskeleton. A good starting point.          |
| `debug.bat`           | A small, quick run with few iterations, designed for testing and debugging the optimization pipeline.   |
| `exo_4param.bat`      | Optimizes the controller with an exoskeleton using the legacy 4-parameter spline for its torque profile. |
| `exo_4param_kine.bat` | Similar to `exo_4param`, but uses a kinematics-focused cost function (`-kine`).                           |
| `exo_npoint.bat`      | Optimizes with an exoskeleton using the modern n-point spline controller.                                 |
| `exo_npoint_cont.bat` | An example of a continued optimization, starting from the results of a previous run.                      |

You can modify these files or create new ones to define custom optimization runs.

## Command-Line Arguments

The `train.py` script accepts a wide range of arguments to customize the optimization. Here are the most important ones, grouped by category. For a complete list, refer to `myoassist_reflex/config/arg_parser.py`.

### Model Configuration
- `--model`: The physical model for the legs and feet. Options: `baseline`, `dephy`, `hmedi`, `humotech`, `osl`.
- `--musc_model`: The muscle model to use. `leg_11` (22 muscles total) is standard. `leg_80` (160 muscles total) is experimental.
- `--delayed`: Use delayed muscle dynamics. (Default: `False`)

### Exoskeleton Configuration
- `--exo_bool`: Enable (`True`) or disable (`False`) the exoskeleton.
- `--use_4param_spline`: If `exo_bool` is `True`, use the legacy 4-parameter spline controller. If `False`, uses the n-point spline.
- `--n_points`: Number of control points for the n-point spline (e.g., `4` for a 4-point spline).
- `--max_torque`: Maximum torque the exoskeleton can apply (in Nm).
- `--fixed_exo`: Keep exoskeleton parameters fixed (not optimized).

### Optimization Target
- `-eff`, `-vel`, `-kine`, `-combined`, etc.: These flags set the primary objective of the cost function. They are mutually exclusive. Choose one that best fits your goal (e.g., minimizing effort, matching a target velocity, or tracking reference kinematics).

### Optimizer Settings
- `--popsize`: The population size for the CMA-ES optimizer (number of solutions per generation).
- `--maxiter`: The maximum number of generations the optimizer will run.
- `--threads`: Number of parallel threads for evaluating the population.
- `--sigma`: The initial standard deviation (step size) for the CMA-ES optimizer.

### Continuing an Optimization

You can start a new optimization from the results of a previous one or resume an interrupted run.

#### `--param_path`: Start with Existing Parameters
Use this to start a new optimization (e.g., with a different cost function or model) using the best parameters from a previous run as the starting point.
- **Argument**: `--param_path <path_to_results_folder>`
- **Behavior**: The script looks for a `_Best.txt` or `_BestLast.txt` file inside the specified folder and loads it as the initial guess (`x0`) for the new optimization. The optimizer's internal state (covariance matrix, step size) is reset.

**Example from `exo_npoint_cont.bat`**:
```batch
--param_path results/exo_npoint_0630_1557
```

#### `--pickle_path`: Resume a Saved State
Use this to continue an optimization that was stopped prematurely.
- **Argument**: `--pickle_path <path_to_pickle_file>`
- **Behavior**: The script loads a `.pkl` file which contains the entire state of the CMA-ES optimizer at the moment it was saved. This allows the optimization to resume from exactly where it left off, preserving the covariance matrix, step size, and evolution paths. Pickle files are automatically saved in the results directory at the end of an optimization or when it's interrupted.

**Example**:
```batch
--pickle_path results/my_run_20240101_1200/myo_reflex_20240101_1200.pkl
```

### Advanced CMA-ES Termination Criteria

You can fine-tune the optimizer's stopping conditions by passing `CMAOptions` directly via the command line. These are useful for preventing premature termination or for ending a run once a satisfactory solution is found.

- `--cma_options "tolfun:1e-9"`: Sets the tolerance for the change in fitness value. The optimization stops if the change in the best function value over recent generations is less than this tolerance.
- `--cma_options "tolx:1e-9"`: Sets the tolerance for the change in the parameter vector (`x`). The optimization stops if the change in the solution vector is less than this tolerance.
- `--cma_options "tolstagnation:100"`: Sets the number of generations to consider for stagnation. The optimization stops if there is no significant improvement in the median fitness over this number of generations.

You can combine multiple options:
```batch
--cma_options "tolfun:1e-10,tolx:1e-10,tolstagnation:150"
``` 