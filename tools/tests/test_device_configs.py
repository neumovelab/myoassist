"""Every shipped device config builds an env, keeps the observation layout, and trains a step.

The observation layout assertion is the point of this file. Configs address observations by absolute
index, so anything that changes the length of an earlier block silently repoints every sub-policy:
a device whose actuators carry their own dynamics adds entries to `data.act`, which used to lengthen
the "act" block from 22 to 24 and hand the exo net two device filter states in place of two foot
contacts. That produces a policy that trains and is wrong, which no crash would reveal.
"""

from __future__ import annotations

import pytest

from tools.tests.conftest import build_env, shipped_configs, take_a_few_ppo_steps

# qpos(8) | qvel(9) | act(22 muscles) | sensor(4) | target_velocity(1)
EXPECTED_OBS = 44
# 22 muscles + 2 device actuators
EXPECTED_ACT = 24


@pytest.mark.parametrize("config_path", shipped_configs(), ids=lambda p: p.stem.replace("imitation_22_", ""))
def test_shipped_config_trains(config_path):
    config, env = build_env(config_path)
    try:
        assert env.observation_space.shape[0] == EXPECTED_OBS, (
            f"{config_path.name}: observation is {env.observation_space.shape[0]}, expected {EXPECTED_OBS}. "
            "Every config addresses observations by absolute index, so a changed length repoints them."
        )
        assert env.action_space.shape[0] == EXPECTED_ACT, (
            f"{config_path.name}: action is {env.action_space.shape[0]}, expected {EXPECTED_ACT}"
        )
        algo = take_a_few_ppo_steps(config, env)
        # These configs all set mirror_coef, so PPO must be the mirror-penalty subclass; plain PPO
        # here would mean the coefficient was silently dropped and the penalty never applied.
        assert config.ppo_params.mirror_coef > 0
        assert algo == "MirrorPPO", f"{config_path.name}: mirror_coef is set but the algorithm is {algo}"
    finally:
        env.close()
