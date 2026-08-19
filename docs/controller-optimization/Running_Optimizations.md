# Running Optimizations

This guide explains how to run optimizations for the reflex controller with the `run_optim.py` script. It covers basic and advanced settings.

## Quick Start

The optimization framework uses a unified approach with `run_optim.py` as the main entry point.

Turn the model cache on before a long run. CMA-ES composes one model for each candidate, so a
run at `--popsize 32 --maxiter 1000` composes approximately 32,000 models. Without the cache
each one costs 13 to 15 times more:

```bash
export MYOASSIST_CACHE_DIR=~/.cache/myoassist
```

See [Caching](https://myoassist.neumove.org/modeling/devices/exporting-and-loading/#caching).

### Using `run_optim.py`

The `run_optim.py` script provides a cross-platform way to run optimizations:

```bash
# Navigate to the ctrl_optim directory
cd ctrl_optim

# Run an optimization using a configuration name
python run_optim.py <config_name>
```

For example, to run the tutorial configuration:
```bash
python run_optim.py tutorial
```

### Available Configurations

The `optim/training_configs/` directory contains configuration files for different optimization scenarios. Every configuration has a Windows `.bat` file. Six also have a Unix `.sh` file: `tutorial`, `debug`, `amp_kfoot`, `kfoot_stiffness`, `reflex_bilat`, and `anatomics_rom`. The other five (`baseline`, `exo_4param`, `exo_4param_kine`, `exo_npoint`, and `exo_npoint_cont`) are `.bat`-only, so they do not run on Unix. `run_optim.py` selects the file type for your OS. Some example configurations are below.

| Configuration         | Description                                                                                             |
|-----------------------|---------------------------------------------------------------------------------------------------------|
| `baseline`            | A standard optimization for the 22-muscle model without an exoskeleton. A good starting point.          |
| `debug`               | A small, quick run with few iterations, designed for testing and debugging the optimization pipeline.   |
| `tutorial`            | A tutorial configuration for learning the framework.                                                     |
| `exo_4param`          | Optimizes the controller with an exoskeleton using the 4-parameter spline for its torque profile. |
| `exo_4param_kine`     | The same as `exo_4param`; both use the `-kine` cost. It differs only in its `--save_path`.               |
| `exo_npoint`          | Optimizes with an exoskeleton using the modern n-point spline controller.                                 |
| `exo_npoint_cont`     | An example of a continued optimization, starting from the results of a previous run.                      |
| `reflex_bilat`        | Bilateral reflex on the 3D 26-muscle model, with independent per-leg blocks.                             |
| `amp_kfoot`           | Amputee reflex on the passive K-Foot prosthesis (see [Amputee and Prosthetic Control](Amputee_Prosthetic_Control.md)).  |
| `kfoot_stiffness`     | Amputee reflex plus prosthetic ankle stiffness optimization.                                             |
| `anatomics_rom`       | Ankle range-of-motion study with the Anatomics exoskeleton.                                              |

### Listing Available Configurations

To see all available configurations:

```bash
python run_optim.py
```

This will display a list of all available configuration files in the `optim/training_configs/` directory.

## Configuration File Structure

The configuration files in `optim/training_configs/` hold the command-line arguments for the `train.py` script. `run_optim.py` runs the file from `ctrl_optim/optim/`. The `.bat` and `.sh` files hold the same arguments, with different line-continuation syntax.

The shipped `tutorial` configuration is a continued run. It loads prepared parameters with `--param_path ../results/optim_results/tutorial_prep`.

**Example `tutorial.bat`:**
```batch
python train.py ^
    --msk myolegs22 ^
    --device Tutorial_L1 ^
    --sim_time 20 ^
    --pose_key walk_left ^
    --num_strides 5 ^
    --delayed 0 ^
    --optim_mode single ^
    --reflex_mode uni ^
    --tgt_vel 1.25 ^
    --trunk_err_type ref_diff ^
    --tgt_sym_th 0.1 ^
    --tgt_grf_th 1.5 ^
    -eff ^
    --ExoOn 1 ^
    --use_4param_spline ^
    --max_torque 100.0 ^
    --popsize 8 ^
    --maxiter 50 ^
    --threads 8 ^
    --sigma_gain 10 ^
    --param_path ../results/optim_results/tutorial_prep ^
    --save_path tutorial
```

**Equivalent `tutorial.sh`:**
```bash
exec "$PYTHON_CMD" -m ctrl_optim.optim.train \
    --msk myolegs22 \
    --device Tutorial_L1 \
    --sim_time 20 \
    --pose_key walk_left \
    --num_strides 5 \
    --delayed 0 \
    --optim_mode single \
    --reflex_mode uni \
    --tgt_vel 1.25 \
    --trunk_err_type ref_diff \
    --tgt_sym_th 0.1 \
    --tgt_grf_th 1.5 \
    -eff \
    --ExoOn 1 \
    --use_4param_spline \
    --max_torque 100.0 \
    --popsize 8 \
    --maxiter 50 \
    --threads 8 \
    --sigma_gain 10 \
    --param_path ../results/optim_results/tutorial_prep \
    --save_path tutorial
```

### Creating Custom Configurations

You can create new configurations by:
1. Copying an existing `.bat` or `.sh` file from `optim/training_configs/`
2. Modifying the arguments as needed
3. Save with a new name in the `optim/training_configs/` directory

## Arguments

The `train.py` script accepts a wide range of arguments to customize the optimization. Here are the most important ones, grouped by category. For a complete list, refer to `ctrl_optim/optim/config/arg_parser.py`.

### Model Configuration

Raw registry keys define the environment. See [Defining an Environment](../getting-started/defining-an-environment.md) for the full reference. Run `python -m assist_sim list` for the valid keys.

- `--msk`: the human MSK model. Use `myolegs22` for 2D. Use `myolegs26` or the 80-muscle `myolegs` for 3D. The muscle count and the 2D or 3D control mode come from this key.
- `--device`: the assistive device, for example `Tutorial_L1`, `Humotech_L1`, or `DephyExoBoot_L1`.
- `--terrain`: an optional terrain. Give a `myoassist_terrains` JSON path or an inline config such as `'{"terrain":"slope","deg":8}'`. Omit it for flat ground. A `slope` terrain sets the course grade, so there is no separate `--tgt_slope`.
- `--env-spec`: a path to a JSON env-spec (`{msk, device, terrain}`). Use it in place of the three flags above.
- `--delayed`: set to `1` to use delayed muscle dynamics. The default is off.

### Run and Simulation Settings

Every shipped configuration sets these.

- `--optim_mode`: the run mode. The framework implements `single` (one optimization) and `evaluate` (score existing parameters). Other values in the help text are not implemented.
- `--save_path`: the name or path for the results folder.
- `--sim_time`: the maximum simulation time per evaluation, in seconds.
- `--num_strides`: the minimum number of strides used to compute the cost.
- `--pose_key`: the initial keypose of the model, for example `walk_left`.

### Reflex Mode

`--reflex_mode` sets how the reflex controller maps parameters to the two legs.

- `uni` or unset (the default): symmetric. One reflex block drives both legs. `uni` is the common symmetric setting. Every shipped 2D configuration and the tutorial use `--reflex_mode uni` with `myolegs22`.
- `bilat`: bilateral. Each leg gets its own reflex block, so the two legs are independent. This doubles the reflex parameter count.
- `amp`: amputee. This is `bilat` plus prosthetic tolerance, for a model with a prosthetic device. See [Amputee and Prosthetic Control](Amputee_Prosthetic_Control.md).
- `ind`: another accepted value. The framework maps it as symmetric, the same as `uni`.

### Amputee and Prosthetic Devices

To optimize on an amputee model, pair a prosthetic device with `--reflex_mode amp`. To also tune a passive prosthetic ankle, add `--optimize_stiffness`. See [Amputee and Prosthetic Control](Amputee_Prosthetic_Control.md) for both.

### Ankle Range of Motion

`--ankle_range MIN MAX` limits the ankle travel, in radians. `MIN` is the plantarflexion limit (negative). `MAX` is the dorsiflexion limit (positive). The framework clamps both ankles to this range on every step. Use it as a swept study variable, for example with the Anatomics exoskeleton.

### Exoskeleton Configuration
- `--ExoOn`: set to `1` to enable the exoskeleton, or `0` to disable it.
- `--use_4param_spline`: with the exoskeleton on, use the 4-parameter spline controller. Without this flag, the framework uses the n-point spline.
- `--n_points`: the number of control points for the n-point spline, for example `4`.
- `--max_torque`: the maximum torque the exoskeleton can apply, in Nm. It also sets the initial torque values for both controllers. The default is `10.0`.
- `--fixed_exo`: keep the exoskeleton parameters fixed, so the optimizer does not tune them. This affects the 4-parameter controller only. With the n-point spline it does nothing.

### Optimization Target
- `-eff`, `-vel`, `-kine`, etc.: These flags set the primary objective of the cost function. They are mutually exclusive. Choose the one that best fits your goal, for example minimizing effort, matching a target velocity, or tracking reference kinematics. For more information see (**[Understanding Cost](Understanding_Cost.md)**).
- `--tgt_vel`: the target walking velocity, in m/s.
- `--tgt_sym_th`: the symmetry threshold used in the cost.
- `--tgt_grf_th`: the normalized ground-reaction-force threshold used in the cost.

### Optimizer Settings
- `--popsize`: The population size for the CMA-ES optimizer (number of solutions per generation).
- `--maxiter`: The maximum number of generations the optimizer will run.
- `--threads`: Number of parallel threads for evaluating the population.
- `--sigma_gain`: Gain value for the initial standard deviation (step size) for the CMA-ES optimizer (if gain = 1, sigma = 0.01).

### Continuing an Optimization

You can start a new optimization from the results of a previous one or resume an interrupted run.

#### `--param_path`: Start with Existing Parameters
Use this to start a new optimization (e.g., with a different cost function or model) using the best parameters from a previous run as the starting point.
- **Argument**: `--param_path <path_to_results_folder>`
- **Behavior**: The script looks for a `*_BestLast.txt` file inside the specified folder and loads it as the initial guess for the new optimization. A folder with only a `_Best.txt` file does not work. The optimizer's internal state (covariance matrix, step size) is reset.

**Example**:
```bash
--param_path results/exo_npoint_date_time
```

#### `--pickle_path`: Resume a Saved State
Use this to continue an optimization that was stopped prematurely.
- **Argument**: `--pickle_path <path_to_pickle_file>`
- **Behavior**: The script loads a `.pkl` file which contains the entire state of the CMA-ES optimizer at the moment it was saved. This allows the optimization to resume from exactly where it left off, preserving the covariance matrix, step size, and evolution paths. Pickle files are automatically saved in the results directory at the end of an optimization or when it's interrupted.

**Example**:
```bash
--pickle_path results/my_run_date_time/myo_reflex_date_time.pkl
```

## Results and Configuration Saving

### Results Location
All results are automatically saved in the `ctrl_optim/results/optim_results/` directory, with each run creating a timestamped subdirectory.

### Configuration Saving
The system automatically saves the final configuration used for each run:

- **Configuration files**: Saves as `config_name_timestamp.bat` or `config_name_timestamp.sh` depending on the platform
- **Results directory**: Creates a timestamped subdirectory containing all optimization outputs

### Output Files
Each optimization run produces several output files:
- `*_Best.txt`: The best parameters found during optimization
- `*_BestLast.txt`: The best parameters from the final population
- `*_Cost.txt`: Detailed cost breakdown for the best solution
- `*_Pickle.pkl`: CMA-ES state for resuming optimization
- `outcmaes/`: Directory containing CMA-ES internal files

## Troubleshooting

### Common Issues

1. **"Module not found" errors**: Make sure you're running from the correct directory:
   ```bash
   cd ctrl_optim
   python run_optim.py <config_name>
   ```

2. **Configuration not found**: Verify the configuration name exists in `optim/training_configs/` directory:
   ```bash
   python run_optim.py
   # This will list all available configurations
   ```

3. **File path errors**: The system automatically resolves paths, but ensure your directory structure is correct.

4. **Permission denied on .sh files**: The `run_optim.py` script handles this automatically, but if you need to run .sh files directly:
   ```bash
   chmod +x optim/training_configs/*.sh
   ```

`run_optim.py` uses one command on every OS. This works for the configurations that ship both a `.bat` and a `.sh` file. The five `.bat`-only configurations run on Windows only. To use one on Unix, first copy its arguments into a `.sh` file.
