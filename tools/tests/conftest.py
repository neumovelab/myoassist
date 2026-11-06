"""Shared fixtures for the RL smoke tests.

These tests build real composed models and take real PPO steps, so they are slower than a unit
test but still seconds each. They exist to catch the breakages that a config or model change causes
silently: a shifted observation index, an actuator count that changes the action space, a terrain
that no longer composes.
"""

from __future__ import annotations

import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEVICE_SWEEP_DIR = REPO_ROOT / "rl_train/train/train_configs/device_sweep"


def shipped_configs() -> list[pathlib.Path]:
    """The device-sweep configs, which are the ones a user is told to run."""
    configs = sorted(DEVICE_SWEEP_DIR.glob("imitation_22_*.json"))
    assert configs, f"no shipped configs found in {DEVICE_SWEEP_DIR}"
    return configs


def build_env(config_path: pathlib.Path, *, terrain=None, num_envs: int = 1):
    """Compose the model and build the vec env described by a training config."""
    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
    from rl_train.utils.data_types import DictionableDataclass

    config = DictionableDataclass.create(ExoImitationTrainSessionConfig, json.loads(config_path.read_text()))
    config.env_params.num_envs = num_envs
    if terrain is not None:
        config.env_params.terrain = str(terrain)
    # One short rollout is enough to reach the update; the defaults would collect 4096 steps.
    config.ppo_params.n_steps = 64
    config.ppo_params.batch_size = 64
    return config, EnvironmentHandler.create_environment(config, is_rendering_on=False)


def take_a_few_ppo_steps(config, env, total_timesteps: int = 128) -> str:
    """Run past the first PPO update and return the algorithm class name."""
    from rl_train.envs.environment_handler import EnvironmentHandler

    model = EnvironmentHandler.get_stable_baselines3_model(config, env)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return type(model).__name__


@pytest.fixture(scope="session", autouse=True)
def _cap_threads():
    """Keep the per-process thread pools small so the parametrised cases do not oversubscribe."""
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "2")
