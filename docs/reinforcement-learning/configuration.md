---
title: Understanding Configuration
parent: Reinforcement Learning
nav_order: 2
layout: home
---

# Configuration

Configuration files define training parameters for reinforcement learning experiments. The system uses JSON files that are converted to dataclasses for easy access.

## Quick Start

### Running with Configuration

```bash
python rl_train/run_train.py --config_file_path [path/to/config.json]
```

### Overriding Configuration Parameters

You can override any configuration parameter using command-line arguments:

example:
```bash
python rl_train/run_train.py --config_file_path config.json --config.total_timesteps 1000 --config.env_params.num_envs 16
```
This example overrides two configuration parameters via the command line:
- Sets the total training timesteps to 1000
- Sets the number of parallel training environments to 16

## Configuration Structure

### Default Configuration Files

Configuration files are located in `myoassist_rl/rl_train/train_configs/`:

- `imitation_tutorial_22_separated_net_partial_obs.json` - Imitation learning using the "TUTORIAL" model, which provides only ankle angle and velocity to the exo
- `imitation_tutorial_22_separated_net_full_obs.json` - Full Exo observation imitation

### Configuration Hierarchy

The configuration system uses a hierarchical dataclass structure:

```
TrainSessionConfigBase
└── ImitationTrainSessionConfig
    └── ExoImitationTrainSessionConfig
```

## Configuration Components

### Core Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `total_timesteps` | Total training timesteps | 3e7 |

### Evaluate Parameters

These parameters are provided as a list of dictionaries, where each dictionary represents a different evaluation configuration. Multiple configurations will be executed in sequence.

| Parameter | Description | Example |
|-----------|-------------|---------|
| `num_timesteps` | Number of timesteps for evaluation | 200 |
| `min_target_velocity` | Minimum target velocity | 1.25 |
| `max_target_velocity` | Maximum target velocity | 1.25 |
| `target_velocity_period` | Target velocity period | 2 |
| `velocity_mode` | Velocity mode (UNIFORM, SINUSOIDAL, STEP) | "UNIFORM" |
| `cam_type` | Camera type | "follow" |
| `cam_distance` | Camera distance | 2.5 |
| `visualize_activation` | Visualize muscle activation | true |

**Example Configuration:**
<details>
<summary>Click to expand example configuration</summary>

```json
[
  {
    "num_timesteps": 200,
    "min_target_velocity": 1.25,
    "max_target_velocity": 1.25,
    "velocity_mode": "UNIFORM",
    ...
  },
  {
    "num_timesteps": 300,
    "min_target_velocity": 1.0,
    "max_target_velocity": 2.0,
    "velocity_mode": "SINUSOIDAL"
    ...
    
  }
]
```
</details>

### Logger Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `logging_frequency` | Logging frequency | 8 |

### Environment Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `env_id` | Environment identifier | "myoAssistLegImitationExo-v0" |
| `num_envs` | Number of parallel environments | 32 |
| `seed` | Random seed | 1234 |
| `safe_height` | Safe height for fall detection | 0.7 |
| `out_of_trajectory_threshold` | Threshold for trajectory deviation | 0.2 |
| `flag_random_ref_index` | Randomize reference motion index | true |
| `control_framerate` | Control frequency | 30 |
| `physics_sim_framerate` | Physics simulation framerate | 1200 |
| `min_target_velocity` | Minimum target velocity | 1.25 |
| `max_target_velocity` | Maximum target velocity | 1.25 |
| `min_target_velocity_period` | Minimum target velocity period | 2 |
| `max_target_velocity_period` | Maximum target velocity period | 10 |
| `enable_lumbar_joint` | Enable lumbar joint | false |
| `lumbar_joint_fixed_angle` | Lumbar joint fixed angle | -0.13 |
| `lumbar_joint_damping_value` | Lumbar joint damping value | 0.05 |
| `observation_joint_pos_keys` | Joint position observation keys | ["ankle_angle_l", "hip_flexion_l"] |
| `observation_joint_vel_keys` | Joint velocity observation keys | ["ankle_angle_l", "hip_flexion_l"] |
| `observation_joint_sensor_keys` | Joint sensor observation keys | ["r_foot", "l_foot"] |
| `terrain_type` | Terrain type (flat, random, harmonic_sinusoidal, slope, dev) | "flat" |
| `terrain_params` | Terrain parameters (space-separated values) | "0.1 20" |
| `custom_max_episode_steps` | Maximum episode steps | 1000 |
| `model_path` | MuJoCo model file path | "models/22muscle_2D/myoLeg22_2D_TUTORIAL.xml" |
| `reference_data_path` | Path to reference motion data | "rl_train/reference_data/short_reference_gait.npz" |
| `reference_data_keys` | Joint keys for reference data | ["ankle_angle_l", "hip_flexion_l"] |
| `prev_trained_policy_path` | Path to previous trained policy | null |

### Environment Parameters - Reward Keys and Weights

| Parameter | Description | Example |
|-----------|-------------|---------|
| `qpos_imitation_rewards` | Joint position imitation rewards | {"pelvis_ty": 0.1, "hip_flexion_l": 0.2} |
| `qvel_imitation_rewards` | Joint velocity imitation rewards | {"pelvis_ty": 0.1, "hip_flexion_l": 0.2} |
| `end_effector_imitation_reward` | End effector imitation reward | 0.0 |
| `forward_reward` | Forward movement reward | 1.0 |
| `muscle_activation_penalty` | Muscle activation penalty | 0.1 |
| `muscle_activation_diff_penalty` | Muscle activation difference penalty | 0.1 |
| `footstep_delta_time` | Footstep delta time | 0.0 |
| `average_velocity_per_step` | Average velocity per step | 0.0 |
| `muscle_activation_penalty_per_step` | Muscle activation penalty per step | 0.0 |
| `joint_constraint_force_penalty` | Joint constraint force penalty | 1.0 |
| `foot_force_penalty` | Foot force penalty | 0.5 |

### PPO Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `learning_rate` | Learning rate | 0.0001 |
| `n_steps` | Number of tuples collected per environment (n_steps * num_envs must be ≥ batch_size) | 512 |
| `batch_size` | Batch size | 8192 |
| `n_epochs` | Number of epochs per update | 30 |
| `gamma` | Discount factor | 0.99 |
| `gae_lambda` | GAE lambda parameter | 0.95 |
| `clip_range` | PPO clip range | 0.2 |
| `clip_range_vf` | Value function clip range | 100 |
| `ent_coef` | Entropy coefficient | 0.001 |
| `vf_coef` | Value function coefficient | 0.5 |
| `max_grad_norm` | Maximum gradient norm | 0.5 |
| `use_sde` | Use state dependent exploration | false |
| `sde_sample_freq` | SDE sample frequency | -1 |
| `target_kl` | Target KL divergence | 0.01 |
| `device` | Device for training | "cpu" |

### Policy Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `custom_policy_params` | Custom policy parameters | See below |

### Policy Parameters - Custom Policy Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `net_arch` | Network architecture for human actor, exo actor, and common critic | {"human_actor": [64, 64], "exo_actor": [8, 8], "common_critic": [64, 64]} |
| `net_indexing_info` | Network indexing information for observation and action ranges | See [Network Index Handler](network-index-handler) |
| `log_std_init` | Initial log standard deviation | 0.0 |

### Auto Reward Adjust Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `learning_rate` | Learning rate for reward adjustment | 0.0 |

## Related Documentation

- [Terrain Types](terrain-types) - Detailed explanation of terrain types and parameters
- [Network Index Handler](network-index-handler) - Network indexing information and structure

## Example Configuration

[imitation_tutorial_22_separated_net_partial_obs.json](https://github.com/neumovelab/myoassist/blob/myoassist0.1prepare/rl_train/train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json)
