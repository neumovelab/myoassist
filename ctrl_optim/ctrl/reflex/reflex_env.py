# Author(s): Calder Robbins <robbins.cal@northeastern.edu>
"""Minimal, reach-free MyoAssist env for the Controller-Optimization framework.

The CO reflex controller needs a plain steppable MuJoCo sim, not a task env.
Historically it borrowed myosuite's ``myoLegStandRandom-v0`` (a ``ReachEnvV0``),
but that env's ``_setup`` looks up a reach-target site (``pelvis``) that the
MyoAssist *composed* leg models (``myolegs22``/``myolegs26`` + assistive device,
built by :func:`myoassist_utils.compose.compose_env_model`) do not carry, so it
raises ``ValueError: No site with name "pelvis"``.

``ReflexEnvV0`` is that env with all reach specifics stripped
(``target_reach_range``, tip/target sites, reach obs + reward).  It subclasses
myosuite's ``BaseV0`` -- which is itself a ``myosuite.envs.env_base.MujocoEnv``
and which the old env also derived from -- so the muscle stepping semantics CO
relied on (``normalize_act=False`` -> raw ctrl passthrough) are preserved
exactly.  CO consumes only ``.sim`` (model + data reads), ``.step(act)``,
``.reset()``, ``.forward()`` and ``.dt``; the obs/reward here are trivial
placeholders (CO computes its own reflex sensors and cost).
"""
from __future__ import annotations

import collections

import numpy as np

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils import gym


class ReflexEnvV0(BaseV0):
    """Reach-free leg env: a steppable sim that resets to the stand keyframe."""

    # Minimal obs; CO ignores these but env_base needs a non-empty obs vector.
    DEFAULT_OBS_KEYS = ["qpos", "qvel", "act"]
    # Single zero-weight key -> reward is always 0 (CO computes its own cost).
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {"act_reg": 0.0}

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two-step (pickle-safe) construction, mirroring myosuite envs:
        # EzPickle captures ctor args, __init__ builds the sim, _setup finishes.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            env_credits=self.MYO_CREDIT,
        )
        self._setup(**kwargs)

    def _setup(
        self,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        joint_random_range: tuple = (0.0, 0.0),
        **kwargs,
    ):
        # Kept for signature parity with the old env; CO passes (0, 0).
        self.joint_random_range = joint_random_range
        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            **kwargs,
        )
        # Reset baseline == the model's "stand" keyframe (key 0), exactly as the
        # old ReachEnvV0 did (``init_qpos = key_qpos[0]``).  CO overrides the
        # pose afterwards via keyframes + explicit joint sets.
        if self.sim.model.nkey > 0:
            self.init_qpos[:] = self.sim.model.key_qpos[0]
            self.init_qvel[:] = self.sim.model.key_qvel[0]

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["time"] = np.array([sim.data.time])
        obs_dict["qpos"] = sim.data.qpos[:].copy()
        obs_dict["qvel"] = sim.data.qvel[:].copy() * self.dt
        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):
        # Trivial, always-alive reward; CO never reads it.
        rwd_dict = collections.OrderedDict(
            (
                ("act_reg", np.array([0.0])),
                # Must-keys for env_base bookkeeping.
                ("sparse", np.array([0.0])),
                ("solved", np.array([False])),
                ("done", np.array([False])),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict
