---
title: Code Structure
parent: Reinforcement Learning
nav_order: 5
layout: home
---

# RL Training Code Structure

This page provides an overview of the directory layout and the main responsibilities of each module inside the `rl_train` package. Use it as a quick reference when you need to modify, debug, or extend the training pipeline.

---

## Entry Points

| Script | Purpose |
|--------|---------|
| `run_simulation.py` | Quickly spin up an environment and roll **random actions** for smoke-testing the simulation. |
| `run_train.py` | Main **training launcher**. Reads a JSON config, constructs environments, and starts Stable-Baselines3 PPO training. |
| `run_policy_eval.py` | Replay a **trained policy** in evaluation mode and generate analysis artefacts. |

All three scripts accept a rich set of CLI flags so that most hyper-parameters can be overridden without editing the JSON config files.

---

## Directory Layout

```text
rl_train/
├── envs/                # Gym / MuJoCo environment definitions
│   ├── myoassist_leg_base.py
│   ├── myoassist_leg_imitation.py
│   └── environment_handler.py
│
├── train/               # Training pipeline (configs, commands, policies)
│   ├── train_configs/   # JSON files that fully specify a training session
│   ├── train_commands/  # Convenience shell commands for long experiments
│   └── policies/        # Custom policy networks
│
├── utils/               # Generic utilities used across training / analysis
│   └── learning_callback.py  # Custom SB3 callback for logging & checkpoints
│
├── analyzer/            # Post-training analysis & visualisation
│   ├── gait_analyze.py
│   ├── gait_evaluate.py
│   └── train_analyzer.py
│
├── reference_data/      # Human Mo-cap data used for imitation or evaluation
│   └── short_reference_gait.npz
│
└── results/             # Auto-generated output (checkpoints, logs, videos)
```

### `envs/`
*Home of all MuJoCo-based Gym environments*

| File | Key Class | Notes |
|------|-----------|-------|
| `myoassist_leg_base.py` | `MyoAssistLegBase` | Base class that wires intrinsic simulation logic, observation construction and reward terms. |
| `myoassist_leg_imitation.py` | `MyoAssistLegImitationEnv` | Environment for **muscle-driven imitation learning** (human-only). |
| `myoassist_leg_imitation_exo.py` | `MyoAssistLegImitationExoEnv` | Variant that adds **exoskeleton actuation**. |
| `environment_handler.py` | `EnvironmentHandler` | Factory that instantiates and vectorises envs based on JSON config. |

### `train/`
*Launch, configure, and extend PPO training*

* **`train_configs/`** – Dozens of ready-made JSON presets. The file name usually describes the experiment (`imitation_tutorial_22_separated_net_partial_obs.json`).
* **`train_commands/`** – Helper shell scripts or `*.sh` bundles so long experiments can be reproduced easily on a cluster.
* **`policies/`** – Custom network architectures. If absent, SB3’s default MLP is used.

### `utils/`
*Shared helpers – no training logic inside*

| File | What it does |
|------|--------------|
| `learning_callback.py` | Saves checkpoints, videos and metrics every *N* steps. Also handles curriculum switches. |
| `train_log_handler.py` | Small wrapper around **loguru** to standardise log output across scripts. |
| `numpy_utils.py` | Misc. helper functions for fast array ops. |
| `data_types.py` | Pydantic-style typed dicts used for config validation. |

### `analyzer/`
*Post-hoc evaluation & visualisation*

The analysis pipeline is modular – run `train_analyzer.py` to generate plots in `results/train_session_*/analyze_results/`.

### `reference_data/`
Contains reference gait trajectories (e.g., **NPZ** files) used for imitation or for computing biomechanical metrics.

---

## Typical Data Flow

1. **`run_train.py`** loads a JSON config → constructs an `EnvironmentHandler`.
2. The handler creates multiple **`MyoAssistLegImitationEnv`** instances and wraps them using SB3’s `SubprocVecEnv`.
3. A PPO policy (custom or default) is initialised and starts learning.
4. Every *k* steps `LearningCallback` saves:
   - `trained_models/model_<steps>.zip`
   - `train_log.json`
   - preview videos (if `flag_rendering` is on)
5. After training, run **`run_policy_eval.py`** to replay checkpoints and kick off **`analyzer/train_analyzer.py`**.

---

## Extending the Pipeline

1. **Add a new terrain** – update `HfieldManager` and reference it in your JSON config.
2. **Custom reward** – subclass `MyoAssistLegBase` and override `_calculate_reward()`.
3. **Different algorithm** – replace the PPO import in `run_train.py` with any SB3 algorithm; the callback remains compatible.
4. **New plots** – add a function in `analyzer/gait_analyze.py` and call it from `train_analyzer.py`.

---
