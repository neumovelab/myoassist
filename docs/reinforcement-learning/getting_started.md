---
title: Getting Started
parent: Reinforcement Learning
nav_order: 1
layout: home
---

# Getting Started

This guide shows you the fastest way to test the RL system and run training in the MyoAssist RL system.

## Quick Test Commands

### 1. Environment Creation Example

See how to create a simulation environment:

```bash
python rl_train/run_simulation.py
```

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

**What you'll find:**
- `analyze_results_[timesteps]_[evaluate_number]`: Training analysis results
- `session_config.json`: Configuration used for this training
- `train_log.json`: Training log data

## Full Training (When Ready)

Once you've verified everything works, run full training:

```bash
python rl_train/run_train.py --config_file_path rl_train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json
```

## Policy Evaluation

Test a trained model:

```bash
python rl_train/run_policy_eval.py path/to/trainsession/folder
```

**Example:**
```bash
python rl_train/run_policy_eval.py docs/assets/tutorial_rl_models/train_session_20250728-161129_tutorial_partial_obs
```

## Next Steps

- [Configuration Guide](/docs/reinforcement-learning/configuration.md) - Understand all parameters
- [Terrain Types](/docs/reinforcement-learning/terrain-types.md) - Learn about different terrain options
- [Network Index Handler](/docs/reinforcement-learning/network-index-handler.md) - Understand network architecture
