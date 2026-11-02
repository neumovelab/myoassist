"""Reward annealing moves the weights, and stops the reference terminating episodes with it.

The amputee configs use imitation as a scaffold: the reference cannot describe the affected side,
so it can only teach a posture to bootstrap from, and it is annealed away to leave forward progress
against effort. Two things have to hold for that to be a fair experiment -- the weights the reward
is actually formed from have to move, and once imitation is worth nothing the out-of-trajectory
check has to stop ending episodes for drifting from a reference nobody is paid to follow.
"""

from __future__ import annotations

import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROSTHESIS_DIR = REPO_ROOT / "rl_train/train/train_configs/prosthesis"
INTACT = (
    REPO_ROOT / "rl_train/train/train_configs/device_sweep/imitation_22_Tutorial_L1_h128_e32_sidenet_mirror0p1_actpen10.json"
)


def _build(path: pathlib.Path):
    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.utils.data_types import DictionableDataclass

    raw = json.loads(path.read_text())
    config = DictionableDataclass.create(EnvironmentHandler.get_config_type_from_session_id(raw["env_params"]["env_id"]), raw)
    config.env_params.num_envs = 1
    return config, EnvironmentHandler.create_environment(config, is_rendering_on=False)


def _one_prosthesis_config() -> pathlib.Path:
    configs = sorted(PROSTHESIS_DIR.glob("imitation_22_KFoot_L1_*.json"))
    assert configs, f"no KFoot config in {PROSTHESIS_DIR}"
    return configs[0]


def test_scaling_moves_the_weights_the_reward_is_built_from():
    """`set_reward_weight_scales` has to move `rwd_keys_wt`, not just the per-joint dict.

    `get_reward_dict` forms `dense` from `rwd_keys_wt`; the pre-existing `set_reward_weights`
    updated only `_reward_keys_and_weights`, which changes the shape of the imitation term but
    not how much of it reaches the total.
    """
    config, env = _build(_one_prosthesis_config())
    try:
        base = dict(env.rwd_keys_wt)
        env.set_reward_weight_scales({"qpos_imitation_rewards": 0.0, "forward_reward": 2.0})
        assert env.rwd_keys_wt["qpos_imitation_rewards"] == 0.0
        assert env.rwd_keys_wt["forward_reward"] == pytest.approx(base["forward_reward"] * 2.0)
        # Unnamed keys keep their configured value.
        assert env.rwd_keys_wt["muscle_activation_penalty"] == base["muscle_activation_penalty"]
        # Scales apply to the configured weights, so repeating a call does not compound.
        env.set_reward_weight_scales({"forward_reward": 2.0})
        assert env.rwd_keys_wt["forward_reward"] == pytest.approx(base["forward_reward"] * 2.0)
    finally:
        env.close()


def test_annealed_imitation_stops_terminating_episodes():
    """With the imitation weight at zero the out-of-trajectory check must be inert.

    Otherwise the run keeps enforcing a reference it no longer rewards, which is the failure the
    annealing is meant to escape.
    """
    import numpy as np

    config, env = _build(_one_prosthesis_config())
    try:
        env.set_reward_weight_scales({"qpos_imitation_rewards": 0.0, "qvel_imitation_rewards": 0.0})
        env.reset()
        # A tracking error far past the threshold: with imitation off it must not end the episode.
        env._out_of_trajectory_threshold = 1e-9
        terminated = False
        for _ in range(5):
            _, _, terminated, _, _ = env.step(np.zeros(env.sim.model.nu))
            if float(env.sim.data.joint("pelvis_ty").qpos[0]) < config.env_params.safe_height:
                pytest.skip("model fell within the probe window; the fall check is the terminator here")
            if terminated:
                break
        assert not terminated, (
            "out-of-trajectory ended the episode although the imitation weight is zero, so the "
            "reference is still enforced after being annealed away"
        )
    finally:
        env.close()


def test_intact_configs_declare_no_reward_curriculum():
    """The intact configs must be untouched: annealing is an amputee-only change."""
    env_params = json.loads(INTACT.read_text())["env_params"]
    assert not env_params.get("reward_curriculum"), (
        "an intact config declares a reward curriculum; the annealing is only justified where the "
        "reference cannot describe the model"
    )
