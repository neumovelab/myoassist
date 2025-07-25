# Understanding Cost Functions

The heart of any optimization is its objective function—the "cost" that the optimizer tries to minimize. In this framework, the cost function is a sophisticated, multi-stage process designed to guide the CMA-ES optimizer from a random set of parameters to a controller that produces stable, efficient, and biologically plausible locomotion.

## A Brief Introduction to CMA-ES

The **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** is a powerful stochastic optimization algorithm well-suited for complex, non-linear problems where the gradient is unavailable. At its core, CMA-ES works by iteratively sampling a "population" of candidate solutions (in our case, sets of controller parameters) from a multivariate normal distribution.

For each generation, it performs three key steps:
1.  **Sampling**: It generates a new population of solutions from a Gaussian distribution defined by its mean (the current best guess), its step-size (sigma), and its covariance matrix (the shape and orientation of the distribution).
2.  **Evaluation**: It calculates the "fitness" or "cost" of each solution by running a simulation and seeing how well it performs.
3.  **Update**: It updates the distribution's parameters (mean, step-size, and covariance) based on the ranking of the solutions. The mean is shifted towards the better-performing solutions, and the covariance matrix is adapted to better align with the directions of successful steps.

This process allows CMA-ES to efficiently explore the search space and converge on optimal solutions. The structure of our cost function is critical because it creates a "landscape" that CMA-ES can navigate effectively.

## The Staged Cost Evaluation

A controller with random parameters is highly unlikely to produce a stable walking gait. Most initial attempts will result in the model immediately falling. A simple "pass/fail" cost is not enough; we need to provide a gradient of feedback that tells the optimizer *how badly* a solution failed.

To achieve this, our framework uses a three-stage cost evaluation system. The cost returned is designed to be orders of magnitude different at each stage, creating a clear path for the optimizer to follow.

### Stage 1: Catastrophic Failure (Cost ≈ 1.2E6)

This is the penalty for simulations that fail to complete. An early termination can be triggered by:
- The model's center of mass falling below a certain height.
- Invalid physics states (e.g., `NaN` values).

If the simulation stops before the designated `sim_time`, it is assigned a high penalty.

**Equation:**
\[ C_{\text{early}} = C_{\text{const}} - (t_{\text{sim}} \times 1000) \]
Where:
- \( C_{\text{const}} \) is a large constant, typically set to **1,200,000**.
- \( t_{\text{sim}} \) is the number of timesteps the simulation successfully ran.

This formula rewards controllers that can remain upright for longer, even if they don't complete the full simulation, providing a smooth gradient for the optimizer to climb.

### Stage 2: Constraint Violation (Cost ≈ 9.0E4)

A simulation may run to completion but still not produce a valid walking gait. This stage assigns a medium-tier penalty if the controller fails to meet fundamental constraints required for a meaningful evaluation.

The primary constraints are:
- **Minimum Strides**: The model must complete a minimum number of successful strides (e.g., 5).
- **Symmetry**: The timing of left and right footfalls must be reasonably symmetric.
- **Ground Reaction Force (GRF)**: The peak GRF must exceed a minimum threshold, ensuring the model is pushing off the ground and not just shuffling.

If these constraints are not met, the cost is calculated as:
\[ C_{\text{constraint}} = C_{\text{const}} - (\text{Component Costs}) \]
Where:
- \( C_{\text{const}} \) is a constant, typically **90,000**.
- The component costs are subtracted to provide a gentle gradient, rewarding minor improvements even if the main constraints are not yet passed.

### Stage 3: Final Performance Cost (Cost ≈ 1.0E2)

If a controller produces a valid walk that passes all constraints, it is evaluated with a detailed performance cost. This final cost is a weighted sum of several metrics, determined by the chosen optimization type (e.g., `-eff`, `-kine`, `-vel`).

This cost is on a much lower scale (typically in the hundreds), signaling to the optimizer that it has found a "good" region of the search space and should now focus on fine-tuning the performance.

## Cost Components Explained

The final performance cost is assembled from various components, each quantifying a specific aspect of the gait.

### Effort Cost (Cost of Transport)

Measures the metabolic energy efficiency of the gait. It is calculated as the sum of squared muscle activations over the evaluation period, normalized by the model's mass and the distance traveled.

**Equation:**
\[ C_{\text{effort}} = \frac{\sum_{t=t_{\text{start}}}^{t_{\text{end}}} \sum_{i \in \text{muscles}} a_i(t)^2}{m \cdot d} \]
Where:
- \( a_i(t) \) is the activation of muscle \(i\) at time \(t\).
- \( m \) is the total mass of the model.
- \( d \) is the distance traveled.

### Kinematics Cost

Measures how closely the model's joint angles match a set of reference kinematics from a healthy human walk. The controller's gait cycle for each joint is interpolated to 100 points and compared to the reference data.

**Equation:**
\[ C_{\text{kine}} = \sum_{\text{strides}} \sum_{\text{joints}} \sum_{p=1}^{100} \left( \sqrt{(\theta_{\text{sim}}(p) - \theta_{\text{ref}}(p))^2} \right) \]
Where:
- \( \theta_{\text{sim}}(p) \) is the simulated joint angle at point \(p\) of the gait cycle.
- \( \theta_{\text{ref}}(p) \) is the reference joint angle at point \(p\).
- The **Trunk Cost** is calculated separately using the same formula but is often weighted differently.

### Velocity Cost

Measures the difference between the model's average velocity and a target velocity.

**Equation:**
\[ C_{\text{vel}} = (v_{\text{avg}} - v_{\text{tgt}})^2 \]
Where:
- \( v_{\text{avg}} \) is the average velocity of the model over the evaluation strides.
- \( v_{\text{tgt}} \) is the target velocity.

### Ground Reaction Force (GRF) Cost

Penalizes gaits where the peak vertical ground reaction force is below a certain threshold, ensuring a dynamic walk.

**Equation:**
\[ C_{\text{grf}} = \max(0, GRF_{\text{tgt}} - \max(GRF_{\text{sim}}))^2 \]
Where:
- \( GRF_{\text{tgt}} \) is the target peak GRF (e.g., 1.2 Body Weights).
- \( GRF_{\text{sim}} \) is the simulated vertical GRF.

### Joint Limit Cost (Pain Cost)

Penalizes the controller for relying on joint-limit torques, which represent the passive ligaments and structures of the joints rather than active muscle control. High values indicate the controller is "hanging" on its joints.

**Equation:**
\[ C_{\text{pain}} = \frac{\sum_{t=t_{\text{start}}}^{t_{\text{end}}} \sum_{j \in \text{joints}} |\tau_j(t)|}{t_{\text{end}} - t_{\text{start}}} \]
Where:
- \( \tau_j(t) \) is the passive limit torque at joint \(j\) at time \(t\).

### EMG Profile Cost

Similar to the kinematics cost, this measures the difference between the model's muscle activation patterns and reference EMG data from human subjects.

**Equation:**
\[ C_{\text{emg}} = \sum_{\text{strides}} \sum_{\text{muscles}} \sum_{p=1}^{100} (\text{act}_{\text{sim}}(p) - \text{emg}_{\text{ref}}(p))^2 \]
Where:
- \( \text{act}_{\text{sim}}(p) \) is the simulated muscle activation at point \(p\) of the gait cycle.
- \( \text{emg}_{\text{ref}}(p) \) is the reference EMG value at point \(p\). 