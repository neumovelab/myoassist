"""VelocityMode.MAP: the target velocity is a terrain velocity map's speed at the pelvis.

The lookup interpolates bilinearly between the map's samples and holds the edge value beyond the
outermost ones. These tests pin both, and that the env reads the map where the pelvis actually is,
from the first observation of an episode on.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools.tests.conftest import REPO_ROOT, shipped_configs

TUTORIAL = next(p for p in shipped_configs() if "Tutorial_L1" in p.name)
TERRAIN = json.loads((REPO_ROOT / "docs/examples/env_tiled_course.json").read_text())["terrain"]
MAP_PARAMS = {"start": [-6.0, 0.0, 0.0], "goal": [6.0, 0.0, 0.0], "samples_per_tile": 4, "base_speed": 1.25}


def _map():
    from rl_train.envs.target_velocity_map import TargetVelocityMap

    return TargetVelocityMap(TERRAIN, MAP_PARAMS)


def test_lookup_returns_the_samples_and_interpolates_between_them():
    from myoassist_terrains.config import config_from_dict
    from myoassist_terrains.velocity_map import generate_velocity_map

    vmap = _map()
    for s in generate_velocity_map(config_from_dict(TERRAIN), **MAP_PARAMS):
        assert vmap.speed_at(s.position[0], s.position[1]) == pytest.approx(s.speed, abs=1e-12)

    # Halfway between two neighbouring samples on each axis: the mean of the four corners.
    xs, ys, sp = vmap.xs, vmap.ys, vmap.speed
    x, y = (xs[2] + xs[3]) / 2, (ys[1] + ys[2]) / 2
    assert vmap.speed_at(x, y) == pytest.approx(sp[1:3, 2:4].mean(), abs=1e-12)

    # Beyond the outermost samples the edge value holds.
    assert vmap.speed_at(xs[0] - 5.0, ys[0] - 5.0) == pytest.approx(sp[0, 0], abs=1e-12)
    assert vmap.speed_at(xs[-1] + 5.0, ys[1]) == pytest.approx(sp[1, -1], abs=1e-12)


def _env(velocity_map, *, is_evaluate_mode=False):
    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.utils.data_types import DictionableDataclass

    raw = json.loads(TUTORIAL.read_text())
    raw["env_params"]["terrain"] = TERRAIN
    raw["env_params"]["velocity_map"] = velocity_map
    config = DictionableDataclass.create(EnvironmentHandler.get_config_type_from_session_id(raw["env_params"]["env_id"]), raw)
    config.env_params.num_envs = 1
    return EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=is_evaluate_mode)


def test_env_tracks_the_map_at_the_pelvis():
    from rl_train.envs.myoassist_leg_base import MyoAssistLegBase

    vmap = _map()
    env = _env(MAP_PARAMS)
    u = env.unwrapped
    try:
        rng = np.random.default_rng(0)
        for _ in range(3):
            obs, _ = env.reset()
            assert u._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.MAP
            # The start target is read at the keyframe pose; it has to be where the episode starts.
            start = vmap.speed_at(*u.sim.data.body("pelvis").xpos[:2])
            assert u._target_velocity == pytest.approx(start, abs=1e-9)
            assert obs[-1] == pytest.approx(start, abs=1e-6)
            for _ in range(20):
                xy = u.sim.data.body("pelvis").xpos[:2].copy()
                obs, _, terminated, truncated, _ = env.step(rng.uniform(-1, 1, env.action_space.shape))
                assert u._target_velocity == pytest.approx(vmap.speed_at(*xy), abs=1e-12)
                assert obs[-1] == pytest.approx(u._target_velocity, abs=1e-6)
                if terminated or truncated:
                    break
    finally:
        env.close()


def test_map_mode_is_never_drawn_without_a_map():
    from rl_train.envs.myoassist_leg_base import MyoAssistLegBase

    env = _env(None)
    try:
        modes = set()
        for _ in range(40):
            env.reset()
            modes.add(env.unwrapped._velocity_mode_for_this_episode)
        assert MyoAssistLegBase.VelocityMode.MAP not in modes
        with pytest.raises(AssertionError, match="needs env_params.velocity_map"):
            env.unwrapped.set_target_velocity_mode_manually(MyoAssistLegBase.VelocityMode.MAP, 0.0, 1.0, 1.0, 1.0)
    finally:
        env.close()
