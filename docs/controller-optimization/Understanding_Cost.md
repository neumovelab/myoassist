# Understanding Cost Functions

The objective function of an optimization is its "cost". The optimizer tries to minimize it. This framework uses a multi-stage cost. The stages guide the CMA-ES optimizer from random parameters to a controller that walks in a stable and realistic way.

## A Brief Introduction to CMA-ES

The **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** is a stochastic optimization algorithm well-suited for complex, non-linear problems where the gradient is unavailable. At its core, CMA-ES works by iteratively sampling a "population" of candidate solutions (in our case, sets of controller parameters) from a multivariate normal distribution.

<div style="text-align: center; display: flex; justify-content: center; gap: 20px;">
  <div style="flex: 1; max-width: 800px;">
    <img src="https://myoassist.neumove.org/assets/cmaes.png" alt="CMAES Overview" style="width: 100%; height: 400px; object-fit: contain;">
    <br>
    <i>CMA-ES cost landscape example</i>
  </div>
</div>

For each generation, it performs three key steps:
1.  **Sampling**: It generates a new population of solutions from a Gaussian distribution defined by its mean (the current best guess), its step-size (sigma), and its covariance matrix (the shape and orientation of the distribution).
2.  **Evaluation**: It calculates the "fitness" or "cost" of each solution by running a simulation and seeing how well it performs.
3.  **Update**: It updates the distribution's parameters (mean, step-size, and covariance) based on the ranking of the solutions. The mean is shifted towards the better-performing solutions, and the covariance matrix is adapted to better align with the directions of successful steps.

This process allows CMA-ES to efficiently explore the search space and converge on optimal solutions. The structure of our cost function is critical because it creates a "landscape" that CMA-ES can navigate effectively.

## The Staged Cost Evaluation

A controller with random parameters is highly unlikely to produce a stable walking gait. Most initial values and hand-tuning will result in the model falling sooner or later. Thus, we provide a gradient of feedback that tells the optimizer *how badly* a solution failed.

To achieve this, our framework uses a three-stage cost evaluation system. The cost returned is designed to be orders of magnitude different at each stage, creating a clear path for the optimizer to follow.

### Simulation Error (Cost ≈ 1.2E6)

This is the penalty for simulations that fail to complete. An early termination can be triggered by the following errors:
- Invalid physics states (e.g., `NaN` values).
- No detected ground reaction forces (e.g., model not contacting the ground properly, or incorrect sensor placement)
- Invalid initial pose (e.g., model intersecting ground)

e.g. `walk_cost.py`
```python
        Myo_env.reset(params)
        pose_valid = Myo_env.check_pose_validity()
        
        if not pose_valid:
            return 120 * 10000
```


Note: debug prints for this cost value can be found in `reflex_interface.py`.

If one of these errors is triggered at the start of optimization, it is unlikely to change. CMA-ES will (more often than not) terminate due to stagnation. However, it is possible for these cost values to occur within a population throughout optimization. CMA-ES will not terminate as long as 1.2E6 is not the best overall cost of an iteration.

### Stage 1: Early Termination (Cost ≈ 9.8E4)

A cost of ≈ 9.8E4 represents a successful environment initialization and the standard initial CMA-output cost value if all is well when running an optimization without initial parameters. However, the model is still falling. This cost structure rewards controllers that can remain upright for longer, even if they don't complete the full simulation.

```python
def calculate_early_cost(cost_const: float, data_store: List[Dict], left_stance_foot: List[int], right_stance_foot: List[int], failure_mode: int = 99) -> float:
    """Calculate cost for early termination cases."""
    total_cost = (
        failure_mode * cost_const - 
        (data_store[len(data_store)-1]['obj_func_out']['pelvis_dist'][0]) - 
        (0.5 * (len(left_stance_foot) + len(right_stance_foot)))
    )
    return total_cost
```

### Stage 2: Constraint Violation (Cost ≈ 1.0E4)

A simulation may run to completion but still not produce a desirable walking gait. This stage assigns a medium-tier penalty if the controller fails to meet additional constraints.

The primary constraints are:
- **Minimum Strides**: The model must complete a minimum number of successful strides (e.g., 5).
- **Symmetry**: The timing of left and right footfalls must be symmetric within a threshold.
- **Velocity**: The average velocity must be close to the target velocity.
- **Pelvis Orientation**: The pelvis must remain reasonably upright (for 3D models).

Values for these thresholds and targets are set through the config files and in `train.py` (**[Running_Optimizations](Running_Optimizations.md)**).

If these constraints are not met, the cost is calculated as:

**Example constraint penalties:**
- Velocity penalty: <code>100 × velocity_cost × (velocity_cost > 0.01)</code> (the velocity error is relative to the target speed, so 0.01 is 1%)
- Symmetry penalty: <code>100 × sym_cost × (sym_cost > tgt_sym)</code> (the symmetry error is in leg lengths)

### Stage 3: Final Performance Cost (Cost < 1.0E3)

If a controller produces a valid walk that passes all constraints, it is evaluated with a final performance cost. This final cost is a weighted sum of several terms, determined by the chosen optimization type (e.g., `-eff`, `-kine`, `-vel`). Different final cost functions will produce different results.

| Optimization type | Final-stage terms |
|---|---|
| Effort (`-eff`) | effort |
| Eff_Knee | effort + joint_limit |
| Vel_grf; vel_musc; vel_musc_grf | effort + grf + trunk; effort + emg; effort + trunk + grf + emg |
| Classic | effort + trunk + joint_limit |
| Kine (`-kine`); Kine_grf; Kine_grf_musc | effort + kinematic + trunk; + grf; + grf + emg |

Each term is multiplied by its weight from `--cost_weights` (1 by default). Each term is normalized so that its value has a physical meaning (see below), so the final cost is typically below 1 and well below the constraint stage.

> **Saved results and earlier cost definitions.** The provided parameter sets in `ctrl_optim/results/` (the tutorial example and the preoptimized controllers) were optimized before the cost terms were normalized as described below. The parameters replay unchanged, because the cost does not affect the simulation, but their saved `*_Cost.txt` files and optimization-progress histories use the earlier cost scale. Run the evaluation mode (`--optim_mode evaluate`) to get their cost breakdown in the current units.

## Cost Components Explained

All terms use the evaluation window of the last `--num_strides` strides. The cumulative terms (effort, GRF, joint limits) are time integrals over the window divided by the distance traveled in the window, so they measure the accumulated penalty per meter and do not depend on the control period. The profile terms (kinematics, EMG) compare gait-cycle-normalized profiles, so they do not penalize differences in gait duration. The constraint terms (velocity, symmetry) are dimensionless deviations from the desired gait.

### Effort Cost

The time integral of the squared muscle activations, summed over the muscles, divided by the distance traveled `D` and the number of muscles `n_m`:

<code>C_E = Σ_m ∫ a_m(t)² dt / (D · n_m)</code> (s/m)

This equals the mean squared activation divided by the mean speed: a value of 1 means that every muscle is fully active at 1 m/s. The muscles are those of the groups the reflex controller drives, on both legs.

### Kinematics Cost

The per-stride joint angles (hip, knee, and ankle of both legs) are resampled to 100 points of the gait cycle and compared with the reference. The cost is the absolute error in radians, averaged over the samples, the joints, and the strides. The trunk term is computed separately (`--trunk_err_type`) and averaged the same way.

### Velocity Cost

The norm of the difference between the mean horizontal velocity in the window and the target velocity, divided by the target speed. The forward and lateral components are considered (on slopes, forward is computed in the x–z plane).

### Ground Reaction Force (GRF) Cost

The time integral of the excess of the total vertical GRF (left plus right, in body weights) over the threshold `--tgt_grf_th`, divided by the distance traveled (s/m).

### Joint Limit Cost (Pain Cost)

Penalizes the controller for relying on joint-limit torques, which represent the passive ligaments and structures of the joints rather than active muscle control. The cost is the time integral of the magnitude of the joint-limit torques of the hip, knee, and ankle of both legs, divided by the distance traveled, the number of joints, and the body weight times the leg length (`m g L`), in s/m. The breakdown also reports each joint pair.

### EMG Profile Cost

The squared difference between each muscle's activation and the reference EMG of its group, after each stride is resampled to 100 points. It is averaged over the samples, the strides, and the muscles of each group, then over the groups.

### Symmetry Cost

Computed at detected left/right stance events using Euclidean differences of knee and ankle positions relative to the pelvis, averaged over the requested number of strides and divided by the leg length (hip to knee plus knee to ankle).

An amputee gait is asymmetric by nature, because the prosthetic leg does not move like the intact leg. The symmetry cost and the Stage 2 symmetry constraint still apply, so read their values with that in mind. Raise `--tgt_sym_th` if the constraint is too strict for an amputee run. See [Amputee and Prosthetic Control](Amputee_Prosthetic_Control.md).
