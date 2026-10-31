"""Each training reset draws a target velocity inside the configured range, and leaves that range alone.

``_change_mode_and_target_velocity_randomly`` once passed its arguments out of order: the configured
max landed in the min slot and the random starting phase in the max slot. Nothing crashed. Because the
method writes those values back onto the env, every reset then read the corrupted range as its input,
so a config asking for exactly 1.25 m/s trained against targets that wandered up to ~6 m/s, with min
above max, while the phase stayed pinned to the old velocity.
"""

from __future__ import annotations

import numpy as np

from tools.tests.conftest import build_env, shipped_configs

TUTORIAL = next(p for p in shipped_configs() if "Tutorial_L1" in p.name)


def test_reset_keeps_target_velocity_range():
    """Many resets across every velocity mode keep min/max fixed and the target and phase in range."""
    _, env = build_env(TUTORIAL)
    try:
        u = (env.envs[0] if hasattr(env, "envs") else env).unwrapped
        # A non-degenerate range, so a swapped min/max or a phase in the max slot shows up.
        u._min_target_velocity, u._max_target_velocity = 0.75, 1.75
        modes = set()
        for _ in range(30):
            env.reset()
            modes.add(u._velocity_mode_for_this_episode)
            assert (u._min_target_velocity, u._max_target_velocity) == (0.75, 1.75), (
                f"reset rewrote the target velocity range to [{u._min_target_velocity:.3f}, "
                f"{u._max_target_velocity:.3f}]; the arguments to set_target_velocity_mode_manually are out of order"
            )
            assert 0.75 <= u._target_velocity <= 1.75, f"target velocity {u._target_velocity:.3f} is outside [0.75, 1.75]"
            assert 0.0 <= u._starting_phase <= 2 * np.pi, f"starting phase {u._starting_phase:.3f} is outside [0, 2*pi]"
        assert modes == set(type(u).VelocityMode), f"30 resets only exercised {sorted(m.name for m in modes)}"
    finally:
        env.close()
