---
title: Getting Started
parent: Reinforcement Learning
nav_order: 1
layout: home
---

# Getting Started

This guide shows you the fastest way to test the RL system and run training in the MyoAssist RL system.

## RL Training Folder Structure

Here is a quick overview of the main entry point scripts in the [`rl_train`](../../rl_train/) folder:

| File | Purpose |
|------|---------|
| [`run_simulation.py`](../../rl_train/run_simulation.py) | The simplest way to create and test a MyoAssist RL environment. No training, just environment creation and random actions. |
| [`run_train.py`](../../rl_train/run_train.py) | Main entry point for running RL training sessions. Loads configuration, sets up environments, and starts training. |
| [`run_policy_eval.py`](../../rl_train/run_policy_eval.py) | Entry point for evaluating and analyzing trained policies. Useful for testing policy performance and generating analysis results. |

### Folder Structure









## Quick Test Commands

### 1. Environment Creation Example

See how to create a simulation environment and run for 150 frames(5sec):

```bash
python rl_train/run_simulation.py
```

![result of run_simulation.py](../assets/rl_random_action_tutorial_env.png)

<!--
Display the image at 50% width for better layout.
-->
<p align="center">
  <img src="../assets/rl_random_action_tutorial_env.png" alt="result of run_simulation.py" width="50%">
</p>



**What this does:**
- Shows an example of creating a Gym wrapped MuJoCo simulation environment
- No actual training - just environment creation example

### 2. Quick Training Test

Run a minimal training session to verify everything works:

```bash
python rl_train/run_train.py --config_file_path rl_train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json --config.total_timesteps 12 --config.env_params.num_envs 1 --config.ppo_params.n_steps 4 --config.ppo_params.batch_size 4 --config.logger_params.logging_frequency 1 --config.logger_params.evaluate_frequency 1 --flag_rendering
```

**What this does:**
- Runs actual reinforcement learning training
- Training for only 12 timesteps (very fast)
- Uses 1 environment (minimal resource usage)
- Enables rendering to see the simulation
- Logs results after every rollout (4 steps) for immediate feedback

### 3. Check Results

After training, check the results folder:

```bash
# Results location
rl_train/results/train_session_[date-time]/
```
![Training session result example](/docs/assets/train_session_result.png)


**What you'll find:**
- `analyze_results_[timesteps]_[evaluate_number]`: Training analysis results
- `session_config.json`: Configuration used for this training
- `train_log.json`: Training log data

## Full Training (When Ready)

Once you've verified everything works, run full training:

```bash
python rl_train/run_train.py --config_file_path rl_train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json
```

This file is the default example configuration we provide.  
For more details, see the [Understanding Configuration](/docs/reinforcement-learning/configuration.md) section.


## Policy Evaluation

Test a trained model:

```bash
python rl_train/run_policy_eval.py [path/to/trainsession/folder]
```

**Example (evaluating with a pretrained model we provide):**
```bash
python rl_train/run_policy_eval.py docs/assets/tutorial_rl_models/train_session_20250728-161129_tutorial_partial_obs
```


After training, an `analyze_results` folder will be created inside your `train_session` directory.  
This folder contains various plots and videos that visualize your agent's performance.

- **Where to find:**  
  ```
  rl_train/results/train_session_[date-time]/analyze_results/
  ```
- **What's inside:**  
  - Multiple plots (e.g., reward curves, kinematics, etc.)
  - Videos

The parameters used for evaluation and analysis (such as which plots/videos are generated) are controlled by the `evaluate_param_list` in your `session_config.json` file.

For more details on how to customize these parameters, see the [Understanding Configuration](/docs/reinforcement-learning/configuration.md) section.

## Realtime Policy Running
You can run a trained policy in realtime simulation:
![result of run_simulation.py](/docs/assets/realtime_eval_flat_tutorial.gif)


```bash
python rl_train/run_train.py --config_file_path [path/to/config.json] --config.env_params.prev_trained_policy_path [path/to/model_file] --flag_realtime_evaluate
```

**Parameters:**
- `[path/to/config.json]`: Path to the JSON file in the train_session folder
- `[path/to/model_file]`: Path to the model file (.zip) without extension. It is located in the train_models folder
![trained model](/docs/assets/train_models.png)

**Example:**
```bash
python rl_train/run_train.py --config_file_path docs/assets/tutorial_rl_models/train_session_20250728-161129_tutorial_partial_obs/session_config.json --config.env_params.prev_trained_policy_path docs/assets/tutorial_rl_models/train_session_20250728-161129_tutorial_partial_obs/trained_models/model_19939328 --flag_realtime_evaluate
```

## Next Steps

- [Configuration Guide](/docs/reinforcement-learning/configuration.md) - Understand all parameters
- [Terrain Types](/docs/reinforcement-learning/terrain-types.md) - Learn about different terrain options
- [Network Index Handler](/docs/reinforcement-learning/network-index-handler.md) - Understand network architecture
