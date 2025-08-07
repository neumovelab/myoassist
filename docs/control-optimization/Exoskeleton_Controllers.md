---
title: Exoskeleton Controllers
parent: Control & Optimization
nav_order: 5
layout: home
---

# Exoskeleton Controllers

This document details the architecture, implementation, and optimization of the provided exoskeleton controllers within the MyoAssist Reflex framework.

## Overview

Each exoskeleton's torque profile is governed by one of two spline-based controllers that are active during the stance phase of the gait cycle. The parameters of these controllers are optimized alongside the neuromuscular reflex parameters by the CMA-ES algorithm (**[Running_Optimizations](Running_Optimizations)**).

## 1. Actuator Definition

Each exoskeleton is defined as an actuator within the MuJoCo `.xml` model. There are multiple actuator types available within MuJoCo (**[Modeling](./Modeling.md)**). This actuator is what allows the framework to apply torque to the model.

Here is an example:
```xml
<general biasprm="0 0 0" gainprm="100 0 0" dynprm="1 0 0" biastype="none" gaintype="fixed" dyntype="none" joint="ankle_angle_r" name="Exo_R" gear="1.0" ctrlrange="-1 0" ctrllimited="true"/>
<general biasprm="0 0 0" gainprm="100 0 0" dynprm="1 0 0" biastype="none" gaintype="fixed" dyntype="none" joint="ankle_angle_l" name="Exo_L" gear="1.0" ctrlrange="-1 0" ctrllimited="true"/>
```
Two key attributes are:
-   **`joint`**: This specifies which joint the actuator acts upon (e.g., `ankle_angle_r`).
-   **`name`**: This provides a unique identifier for the actuator (e.g., `Exo_R`).

The `reflex_interface.py` script uses the actuator's name to identify it and apply the calculated torque to the correct entry in the simulation's control vector (`env.sim.data.ctrl`).

## 2. Controller Architecture

Both provided controllers share a common architecture:
- **Finite-State Machine (FSM)**: A simple FSM determines if the leg is in a "STANCE" or "SWING" state based on a vertical ground reaction force (vGRF) threshold.
- **Stance Phase Tracking**: When the leg is in "STANCE", the controller tracks the elapsed time.
- **Torque Spline**: The torque profile is defined by a PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) spline. The input to the spline is the current percentage of the stance phase (0-100%), and the output is the torque to be applied.
- **Stance Duration Averaging**: To normalize the stance phase percentage, the controller maintains a running average of the duration of the last three stances.

NOTE: Both controllers operate on the basis of normalized *stance* percentage (heel strike to toe off) rather than full normalized gait cycle, while the figures shown represent the full normalized gait cycle.

### Controller A: 4-Parameter Spline (`fourparam_spline_ctrl.py`)

This is a version of a widely used controller defined by four parameters that describe the shape of a single torque pulse.

- **Parameters**:
    1.  `peak_torque`: The magnitude of the torque pulse (normalized 0-1, then scaled by `max_torque`).
    2.  `rise_time`: The time it takes to ramp up to peak torque (as a % of stance).
    3.  `peak_time`: The point in the stance phase where the peak torque occurs (as a % of stance).
    4.  `fall_time`: The time it takes to ramp down to zero torque after the peak (as a % of stance).

- **Fixed Controller (`--fixed_exo`)**: This command-line option allows the 4-parameter controller to be used with a fixed, predefined set of initial parameters instead of being optimized. This is useful for evaluating a known, static assistance profile.

<p align="center">
  <img src="./assets/4param.png" alt="4-Parameter Controller Diagram" width="350"/>
</p>

### Controller B: N-Point Spline (`npoint_spline_ctrl.py`)

This is a more flexible controller that defines the torque profile using a variable number of control points.

- **Parameters**: The controller is defined by `2 * n` parameters, where `n` is the number of points (`--n_points`).
    - `n torque` parameters: The torque value at each control point (normalized 0-1).
    - `n timing` parameters: The temporal position of each control point (normalized 0-1).

<p align="center">
  <img src="./assets/npoint.png" alt="NPoint Controller Diagram" width="350"/>
</p>

## 3. Integration and Optimization

### Parameter Bounds and Initialization
- **Bounds (`bounds.py`)**: The optimization bounds for all exoskeleton parameters (both torque and timing) are normalized to a `[0, 1]` range. This provides a consistent and well-behaved search space for the CMA-ES optimizer.
- **Initial Parameters (`train.py`)**: When a *new* optimization is started (i.e., not from a `--param_path`), the initial exoskeleton parameters are set to predefined defaults:
    - **4-Parameter Controller**: The initial shape is based on human-in-the-loop experiments from previous studies (Poggesnsee & Collins 2021), providing a good starting point for optimization.
        - `peak_torque`: 0.5
        - `rise_time`: 0.467
        - `peak_time`: 0.90
        - `fall_time`: 0.075
    - **N-Point Controller**: The initialization uses two key strategies:
        - **Torque Values** (`utils/npoint_torque.py`): Initial values follow a geometric decay pattern where peak torque (0.5 x peak_torque) is placed at or just after the middle point. Surrounding points decrease by powers of two based on distance from peak (i.e., with 4 points: `[0.125, 0.25, 0.5, 0.25]`). 
        - **Timing Values**: Uses a segmented normalization approach: Divides stance phase into `n` equal segments (e.g., for `n=4: [0-25%], [25-50%], [50-75%], [75-100%]`) where each timing parameter is normalized to `[0, 1]` within its segment. This segmentation, or "binning", prevents parameter clustering and CMA-ES destabilization.

### Simulation Interfacing
- **Interface (`reflex_interface.py`)**: The `myoLeg_reflex` class is the core integrator. It instantiates the chosen exoskeleton controller (`FourParamSplineController` or `NPointSplineController`) based on the command-line arguments.
- **Torque Application**: During each step of the simulation, the interface calls the controller's `.update()` method to get the current torque value and applies it to the correct ankle joint actuator.
- **Spline Validity Check**: The interface includes a crucial safety check, `check_spline_validity()`. This function is called before evaluating the cost. It ensures that the timing parameters for the spline are monotonically increasing (i.e., `time_1 < time_2 < ... < time_n`). If the order is invalid, the simulation is assigned a high penalty cost, preventing the optimizer from exploring unstable regions.

## 4. Continued Optimization and Bootstrapping

The framework provides two options for continuing or refining previous optimizations, with some additional logic for the n-point controller. This is handled via the `--param_path` argument in `train.py` (**[Running_Optimizations](./Running_Optimizations.md)**).

### Standard Continued Optimization
If you provide a `--param_path` to an optimization result that used the *same* number of exoskeleton parameters as your new optimization, the framework simply loads the parameters and continues optimizing from that point. This will *always* be true for the `4param controller` and *only* true for the  `npoint controller` if the same npoints value is passed. The same logic applies if you load human-only parameters for a new optimization with an exoskeleton; the framework will initialize the specified number of exo parameters with their default values and append them.

### N-Point Bootstrapping Logic
A more complex option within the framework is the **bootstrapping** capability for continued `npoint` optimizations. This is triggered *only* when you use `--param_path` to load a result with a *different* number of n-points than your new optimization specifies (`--n_points` value). This allows you to, for example, find a good general torque shape with a 3-point controller and then refine it with a higher value n-point controller without starting from scratch.

- **Process**:
    1. The script loads the human parameters and the old `n-point` exo parameters.
    2. It reconstructs the old PCHIP spline from the loaded parameters.
    3. It finds the absolute peak of this old spline.
    4. It generates a new set of new `n timing` parameters, spaced accordingly.
    5. It evaluates the old spline at these new timing points to get the corresponding initial torque values.
    6. **Crucially**, it replaces the new time/torque point closest to the old peak with the *exact* time and torque of that peak. This ensures (arguably) the most important feature of the curve is preserved.
    7. The new, "bootstrapped" set of `2 * n` exo parameters are combined with the human parameters, and the optimization begins.

<p align="center">
  <img src="./assets/bootstrap.png" alt="Bootstrap Diagram" width="600"/>
</p>

This method of bootstrapping provides a clean way to increase or decrease the complexity of the exoskeleton controller while transferring knowledge from previous optimization runs.
