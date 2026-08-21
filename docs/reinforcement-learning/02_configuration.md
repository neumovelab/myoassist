# RL Configuration

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

### Environment Variables

Some settings come from environment variables, not the JSON config:

- `MYOASSIST_CACHE_DIR` - Directory for the model cache. Turn it on before full training (see [Caching](https://myoassist.neumove.org/modeling/devices/exporting-and-loading/#caching)).
- `MYOASSIST_NUM_THREADS` - Caps the threads each worker uses. The default is 8.

## Configuration Structure

### Default Configuration Files

Configuration files are located in `rl_train/train/train_configs/`:

- `imitation_tutorial_22_separated_net_partial_obs.json` - Imitation learning with the `myolegs22` MSK and the Tutorial device. Partial observation gives the exo only ankle angle and velocity.
- `imitation_tutorial_22_separated_net_full_obs.json` - Same setup with full exo observation.
- `imitation_tutorial_22_separated_net_exo_off*.json` - Same setup with the exo held at a constant command (exo off).
- `test.json` - Minimal config for the quick training test (see [Reinforcement Learning](index.md#2-quick-training-test)).
- `device_sweep/` - Eight configs, one per bilateral exo, for the per-device sweep.
- `prosthesis/` - Four configs, one per prosthetic device, on amputee models. See the [directory README](https://github.com/neumovelab/myoassist/blob/main/rl_train/train/train_configs/prosthesis/README.md).

### Configuration Hierarchy

The configuration system uses a hierarchical dataclass structure:

```
TrainSessionConfigBase
└── ImitationTrainSessionConfig
    └── ExoImitationTrainSessionConfig
```

## Configuration Components


## Components Table of Contents

- [General Parameters](#general-parameters)
- [Logger Parameters](#logger-parameters)
- [Environment Parameters](#environment-parameters)
- [PPO Parameters](#ppo-parameters)
- [Policy Parameters](#policy-parameters)
- [Evaluate Parameters](#evaluate-parameters)

### General Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `total_timesteps` | Total training timesteps | 3e7 |



### Logger Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `logging_frequency` | Rollouts between a checkpoint plus a `train_log.json` write | 8 |
| `evaluate_frequency` | Rollouts between analyzer runs, which write `analyze_results_*` and the replay video | 64 |

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
| `msk_key` | Human MSK model key, composed with `device_key` (see [Defining an Environment](../getting-started/defining-an-environment.md)) | "myolegs22" |
| `device_key` | Assistive device key | "Humotech_L1" |
| `terrain` | Terrain spec (path or inline config); defined by [Terrains](https://myoassist.neumove.org/modeling/terrains/) | null |
| `custom_max_episode_steps` | Maximum episode steps | 1000 |
| `model_path` | Optional literal MJCF path (escape hatch); leave null to compose from `msk_key`/`device_key` | null |
| `reference_data_path` | Path to reference motion data (accepts `.npz` or `.json`) | "rl_train/reference_data/short_reference_gait.npz" |
| `reference_data_keys` | Joint keys for reference data | ["ankle_angle_l", "hip_flexion_l"] |
| `reset_keyframe_joint_keys` | Joints restored to the composed model's standing keyframe at every reset. `reset` seeds the next episode from the current `qpos`, so a DOF the reference does not write carries over from the previous episode -- including the value it held while the model was falling. Amputee configs name their prosthetic joints here; empty restores nothing, which is what the intact configs do. | [] |
| `prev_trained_policy_path` | Path to previous trained policy | null |
| `hidden_geom_groups` | Geom groups hidden from rendering. Which group holds clutter is an authoring convention of the model, so it is set here. Rendering only. | [] |
| `joint_limit_sensor_keys` | Joint-limit sensor names feeding `joint_constraint_force_penalty`; empty uses `MyoAssistLegBase.JOINT_LIMIT_SENSOR_NAMES` | [] |

### Environment Parameters - Reward Keys and Weights

| Parameter | Description | Example |
|-----------|-------------|---------|
| `qpos_imitation_rewards` | Joint position imitation rewards | {"pelvis_ty": 0.1, "hip_flexion_l": 0.2} |
| `qvel_imitation_rewards` | Joint velocity imitation rewards | {"pelvis_ty": 0.1, "hip_flexion_l": 0.2} |
| `end_effector_imitation_reward` | End effector imitation reward | 0.0 |
| `forward_reward` | Forward movement reward | 1.0 |
| `muscle_activation_penalty` | Muscle activation penalty, squared | 0.1 |
| `exo_activation_penalty` | Device effort penalty, in the same units as `muscle_activation_penalty`. The muscle mean is over 22 actuators and the device mean over 2, so the per-actuator price is `muscle_activation_penalty`/22 against `exo_activation_penalty`/2. | 0.0 |
| `muscle_activation_diff_penalty` | Smoothness reward, not a penalty despite the name. It adds `dt*mean(exp(-4*(prev-curr)^2))`, which is largest when activations change little between steps. | 0.1 |
| `footstep_delta_time` | Footstep delta time | 0.0 |
| `average_velocity_per_step` | Average velocity per step | 0.0 |
| `muscle_activation_penalty_per_step` | Muscle activation penalty per step | 0.0 |
| `joint_constraint_force_penalty` | Joint constraint force penalty | 1.0 |
| `foot_force_penalty` | Foot force penalty | 0.5 |

### PPO Parameters
> For more detailed explanations of each PPO parameter, please refer to the Stable-Baselines3 documentation:  
> https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters


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
| `mirror_coef` | Weight on the left/right mirror-symmetry penalty (`rl_train/train/mirror_ppo.py`). 0 disables it and PPO behaves as plain PPO. | 0.0 |

`MirrorPPO` is selected only when training starts from scratch. A run that loads
`prev_trained_policy_path` uses `stable_baselines3.PPO`.

### Policy Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `custom_policy_params` | Custom policy parameters | See below |

### Policy Parameters - Custom Policy Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `net_arch` | Network architecture per network. Use `exo_actor` for one exo network, or `exo_actor_r` plus `exo_actor_l` for one weight-shared per-side network (see [Network Index Handler](04_network-index-handler.md)). | {"human_actor": [64, 64], "exo_actor": [8, 8], "common_critic": [64, 64]} |
| `net_indexing_info` | Network indexing information for observation and action ranges | See [Network Index Handler](04_network-index-handler.md) |
| `log_std_init` | Initial log standard deviation | 0.0 |
| `reset_shared_net_after_load` | No-op. The custom policies have no shared feature-extractor trunk, so `reset_network` ignores this flag. | false |
| `reset_policy_net_after_load` | Reinitialize the policy network after loading `prev_trained_policy_path` | false |
| `reset_value_net_after_load` | Reinitialize the value network after loading `prev_trained_policy_path` | false |

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
| `realtime_plotting_info` | Optional list of signals to plot alongside the replay video, one subplot per entry | [] |

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

<!-- ### Auto Reward Adjust Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `learning_rate` | Learning rate for reward adjustment | 0.0 | -->

## Related Documentation

- [Terrains](https://myoassist.neumove.org/modeling/terrains/) - Terrain generation and configuration
- [Network Index Handler](04_network-index-handler.md) - Network indexing information and structure

## Example Configuration

[imitation_tutorial_22_separated_net_partial_obs.json](https://github.com/neumovelab/myoassist/blob/main/rl_train/train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json)
