"""The activation penalties are quadratic, and the device term is normalised per actuator.

Both properties were wrong at some point and neither failure crashes. The muscle penalty was linear
for months, which prices a unit of activation the same in early stance as at push-off. The device
term has to divide each actuator's ctrl by its own ctrlrange, because those are not comparable
across devices -- the ankle exos take [-1, 0], Hippo [-1, 1], STRIDE [0, 400] N -- so an unnormalised
term would price STRIDE's effort a few hundred times higher than Tutorial's and make one weight
meaningless across the sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.tests.conftest import build_env, shipped_configs

TUTORIAL = next(p for p in shipped_configs() if "Tutorial_L1" in p.name)
STRIDE = next(p for p in shipped_configs() if "STRIDE_L2" in p.name)


def _unwrap(env):
    return (env.envs[0] if hasattr(env, "envs") else env).unwrapped


def test_muscle_activation_penalty_is_quadratic():
    """Doubling every activation must quadruple the penalty, which only holds for a squared form."""
    _, env = build_env(TUTORIAL)
    try:
        u = _unwrap(env)
        env.reset()
        penalties = {}
        for level in (0.1, 0.2):
            u.sim.data.act[u._muscle_act_indices] = level
            obs_dict = u.get_obs_dict(u.sim)
            penalties[level] = u.get_reward_dict(obs_dict)["muscle_activation_penalty"]
        ratio = float(penalties[0.2] / penalties[0.1])
        assert ratio == pytest.approx(4.0, rel=1e-6), (
            f"muscle_activation_penalty scales by {ratio:.3f} when activation doubles; "
            "4.0 means squared, 2.0 means the linear form regressed"
        )
    finally:
        env.close()


def test_device_effort_is_quadratic_and_ctrlrange_normalised():
    """Full-scale device command costs 1.0 whatever the actuator's units, and half costs a quarter."""
    for config_path in (TUTORIAL, STRIDE):
        _, env = build_env(config_path)
        try:
            u = _unwrap(env)
            env.reset()
            assert len(u._device_actuator_ids) > 0, f"{config_path.name}: no device actuators found"

            # Drive every device actuator to the end of its own range with the larger magnitude.
            ctrlrange = u.sim.model.actuator_ctrlrange[u._device_actuator_ids]
            full = np.array([r[np.argmax(np.abs(r))] for r in ctrlrange])
            u.sim.data.ctrl[u._device_actuator_ids] = full
            assert u._get_device_effort() == pytest.approx(1.0, rel=1e-9), (
                f"{config_path.name}: full-scale device command should cost 1.0 after per-actuator ctrlrange normalisation"
            )

            u.sim.data.ctrl[u._device_actuator_ids] = full / 2
            assert u._get_device_effort() == pytest.approx(0.25, rel=1e-9), (
                f"{config_path.name}: half-scale should cost 0.25 if the term is squared"
            )
        finally:
            env.close()


def test_device_activation_is_not_observed():
    """`data.act` may carry device state; the observation must still expose only the 22 muscles."""
    for config_path in shipped_configs():
        _, env = build_env(config_path)
        try:
            u = _unwrap(env)
            assert len(u._muscle_act_indices) == 22, (
                f"{config_path.name}: {len(u._muscle_act_indices)} muscle activation entries, expected 22"
            )
            assert len(u.get_obs_dict(u.sim)["act"]) == 22
        finally:
            env.close()
