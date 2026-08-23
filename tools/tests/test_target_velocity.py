"""The episode's target velocity is the one the config asks for.

`_change_mode_and_target_velocity_randomly` runs on every training reset and hands the episode's
speed profile to `set_target_velocity_mode_manually`. It passed them positionally and had two in
the wrong slots, so the randomly drawn phase (`[0, 2*pi]`) landed in `max_target_velocity`. The
setter also writes `_min_target_velocity` from its arguments, so the corruption carried into the
next reset and both bounds drifted through [0, 6.28]. Every config in the repo declares a band and
none of them got it: measured on `imitation_22_Tutorial_L1`, which asks for a flat 1.25 m/s, the
target averaged 2.94 m/s with 47% of episodes above 3 m/s.

Nothing raised, and a policy that cannot hit a 5 m/s demand simply learns to give up on the
forward term, so this is exactly the kind of defect a test has to hold down.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _prosthesis_config(device: str) -> pathlib.Path:
    """A shipped prosthesis config for `device`, found by glob.

    The generator encodes its flags in the filename, so the name moves whenever the shipped
    settings change; matching on the device alone keeps these tests pinned to the behaviour
    rather than to a particular experiment's suffix.
    """
    matches = sorted((REPO_ROOT / "rl_train/train/train_configs/prosthesis").glob(f"imitation_22_{device}_*.json"))
    assert matches, f"no shipped config for {device}"
    return matches[0]


CONFIGS = [
    REPO_ROOT / "rl_train/train/train_configs/device_sweep/imitation_22_Tutorial_L1_h128_e32_sidenet_mirror0p1_actpen10.json",
    _prosthesis_config("KFoot_L1"),
]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem[:40])
def test_reset_draws_the_configured_target_velocity(path):
    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.utils.data_types import DictionableDataclass

    raw = json.loads(path.read_text())
    env_params = raw["env_params"]
    lo, hi = env_params["min_target_velocity"], env_params["max_target_velocity"]

    config = DictionableDataclass.create(EnvironmentHandler.get_config_type_from_session_id(env_params["env_id"]), raw)
    config.env_params.num_envs = 1
    # Training mode on purpose: is_evaluate_mode skips the randomiser, which is why the bug never
    # showed up in an evaluation rollout.
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False)
    try:
        drawn = []
        for _ in range(60):
            env.reset()
            drawn.append(float(env._target_velocity))
        drawn = np.array(drawn)
        # A sinusoidal or step episode is sampled somewhere inside the band, so bound rather than
        # equate -- with a small tolerance for the band being a single point.
        assert drawn.min() >= lo - 1e-6 and drawn.max() <= hi + 1e-6, (
            f"{path.name}: config asks for [{lo}, {hi}] m/s but resets drew "
            f"[{drawn.min():.2f}, {drawn.max():.2f}] (mean {drawn.mean():.2f})"
        )
    finally:
        env.close()


def test_velocity_curriculum_survives_the_reset_randomiser():
    """A curriculum set through `set_target_velocity_range` is not undone by the next reset.

    The randomiser re-reads and re-writes both bounds every episode, so a curriculum that does not
    round-trip through it would be silently discarded.
    """
    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.utils.data_types import DictionableDataclass

    raw = json.loads(_prosthesis_config("KFoot_L1").read_text())
    config = DictionableDataclass.create(EnvironmentHandler.get_config_type_from_session_id(raw["env_params"]["env_id"]), raw)
    config.env_params.num_envs = 1
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False)
    try:
        env.set_target_velocity_range(0.4, 0.4)
        drawn = []
        for _ in range(30):
            env.reset()
            drawn.append(float(env._target_velocity))
        drawn = np.array(drawn)
        assert np.allclose(drawn, 0.4, atol=1e-6), (
            f"curriculum set the band to 0.4 m/s but resets drew [{drawn.min():.2f}, {drawn.max():.2f}]"
        )
    finally:
        env.close()
