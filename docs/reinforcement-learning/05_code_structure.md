# RL Training Code Structure

This page provides an overview of the directory layout and the main responsibilities of each module inside the `rl_train` package. Use it as a quick reference when you need to modify, debug, or extend the training pipeline.

---

## Entry Points

| Script | Purpose |
|--------|---------|
| `run_sim_minimal.py` | Quickly spin up an environment and roll **random actions** for smoke-testing the simulation. |
| `run_train.py` | Main **training launcher**. Reads a JSON config, constructs environments, and starts Stable-Baselines3 PPO training. |
| `run_policy_eval.py` | Replay a **trained policy** in evaluation mode and generate analysis artifacts. |

`run_train.py` and `run_policy_eval.py` accept CLI flags, so most hyper-parameters can be overridden without editing the JSON config files. `run_sim_minimal.py` takes no arguments.

---

## Directory Layout

```text
rl_train/
├── envs/                # Gym / MuJoCo environment definitions
│   ├── myoassist_leg_base.py
│   ├── myoassist_leg_imitation.py
│   ├── myoassist_leg_imitation_exo.py
│   └── environment_handler.py
│
├── train/               # Training pipeline (configs, commands, policies)
│   ├── train_configs/   # JSON files that fully specify a training session
│   ├── train_commands/  # Windows .bat files recording full training invocations
│   └── policies/        # Custom policy networks
│
├── utils/               # Generic utilities used across training / analysis
│   └── learning_callback.py  # Custom SB3 callback for logging & checkpoints
│
├── analyzer/            # Post-training analysis & visualization
│   ├── gait_analyze.py
│   ├── gait_evaluate.py
│   └── train_analyzer.py
│
├── reference_data/      # Human Mo-cap data used for imitation or evaluation
│   ├── segmented.npz
│   └── short_reference_gait.npz
│
└── results/             # Auto-generated output (checkpoints, logs, videos)
```

### `envs/`
*Home of all MuJoCo-based Gym environments*

| File | Key Class | Notes |
|------|-----------|-------|
| `myoassist_leg_base.py` | `MyoAssistLegBase` | Base class that wires intrinsic simulation logic, observation construction and reward terms. |
| `myoassist_leg_imitation.py` | `MyoAssistLegImitation` | Environment for **muscle-driven imitation learning** (human-only). |
| `myoassist_leg_imitation_exo.py` | `MyoAssistLegImitationExo` | Variant that adds **exoskeleton actuation**. |
| `environment_handler.py` | `EnvironmentHandler` | Factory that instantiates and vectorizes envs based on JSON config. |

### `train/`
*Launch, configure, and extend PPO training*

* **`train_configs/`**: Dozens of ready-made JSON presets. The file name usually describes the experiment (`imitation_tutorial_22_separated_net_partial_obs.json`).
* **`train_commands/`**: Windows `.bat` files recording full training invocations, so long experiments can be reproduced verbatim.
* **`policies/`**: Custom network architectures. A custom policy is always used.

### `utils/`
*Shared helpers. No training logic inside.*

| File | What it does |
|------|--------------|
| `learning_callback.py` | Saves a checkpoint and writes `train_log.json` every `logger_params.logging_frequency` rollouts; every `logger_params.evaluate_frequency` rollouts it runs the analyzer in a worker process. |
| `train_log_handler.py` | Small wrapper around **loguru** to standardize log output across scripts. |
| `numpy_utils.py` | Misc. helper functions for fast array ops. |
| `data_types.py` | `DictionableDataclass`: dataclass ↔ dict conversion, and generation of the `--config.*` CLI overrides. |

### `analyzer/`
*Post-hoc evaluation & visualization*

The analysis pipeline is modular. `TrainAnalyzer` has no CLI entry point: the training callback invokes it in a worker process, writing to `rl_train/results/train_session_*/analyze_results_<timesteps>_<NN>/`.

### `reference_data/`
Contains reference gait trajectories (e.g., **NPZ** files) used for imitation or for computing biomechanical metrics. `segmented.npz` is loaded by a relative path, so `run_policy_eval.py` must be run from the repository root.

---

## Typical Data Flow

1. **`run_train.py`** loads a JSON config → constructs an `EnvironmentHandler`.
2. The handler creates the env instances (**`MyoAssistLegImitationExo`** for the default exo env). It wraps them with SB3’s `SubprocVecEnv` only when `num_envs > 1` and rendering is off. Otherwise it uses a single plain gym env.
3. A custom PPO policy is initialized and starts learning.
4. Every *k* steps `LearningCallback` saves:
   - `trained_models/model_<steps>.zip`
   - `train_log.json`
   - preview videos (rendered by the analyzer, gated by `logger_params.evaluate_frequency`)
5. After training, run **`run_policy_eval.py`** to replay checkpoints; it drives `TrainLogAnalyzer`, `GaitAnalyzer` and `ImitationGaitEvaluator` directly (not `TrainAnalyzer`).

---

## Extending the Pipeline

1. **Add a new terrain**: set `env_params.terrain` to a terrain config ([Terrains](https://myoassist.neumove.org/modeling/terrains/)). Use a JSON path or an inline config. See [Defining an Environment](../getting-started/defining-an-environment.md).
2. **Custom reward**: subclass `MyoAssistLegBase` and override `get_reward_dict()`, `_calculate_base_reward()` or `_calculate_reward_per_step()`.
3. **Different algorithm**: the algorithm is selected in `EnvironmentHandler.get_stable_baselines3_model()`, which already switches to `MirrorPPO` when `ppo_params.mirror_coef > 0`. Add further SB3 algorithms there, not in `run_train.py`, which imports none.
4. **New plots**: add a function in `analyzer/gait_analyze.py` and call it from `train_analyzer.py`.

---
