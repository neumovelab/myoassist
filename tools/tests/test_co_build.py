"""CO build-and-step smoke: compose + reflex_interface + planar_root, end to end.

For each representative (model, device, mode, reflex_mode) case, build the reflex
env with the canonical ``np.ones`` x0 (what ``train.py`` seeds CMA-ES with), reset,
and confirm the initial pose is valid -- the ``check_pose_validity`` gate that
``walk_cost`` turns into the ``1.2e6`` sim-error sentinel -- then step the reflex
controller and confirm ``qpos`` stays finite. There is no learning-quality
assertion: ``np.ones`` is the unoptimised start, so the model only has to compose,
seat with foot contact, and step -- not walk.

Coverage: the 26/3D case exercises the CO-only ``planar_root`` re-orientation; amp
exercises the bilateral + prosthetic K-Foot layout; ``+stiffness`` and
``ankle_range`` exercise those feature tails; ``myoLeg_reflex`` itself asserts
``len(params) == layout.total`` on construction, so each case also checks the
param contract for its mode.

The 80-muscle ``myolegs`` 3D case also guards the runtime seat
(``adjust_model_height``): the reflex env is always planar-root, so its vertical DOF
is ``pelvis_ty``, not ``qpos[2]`` (the freejoint z on the RL build). Seating the wrong
axis in 3D used to leave the 80-muscle feet ~7 cm off the ground with all foot GRF 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from ctrl_optim.ctrl.reflex.reflex_interface import myoLeg_reflex


def _build(mode, msk, device, leg, n, reflex_mode=None, optimize_stiffness=False, ankle_range=None):
    return myoLeg_reflex(
        sim_time=5,
        control_params=np.ones(n),
        mode=mode,
        init_pose="walk_left",
        delayed=False,
        slope_deg=0,
        msk_key=msk,
        device_key=device,
        leg_model=leg,
        reflex_mode=reflex_mode,
        optimize_stiffness=optimize_stiffness,
        ankle_range=ankle_range,
        exo_bool=False,
    )


CASES = [
    pytest.param(dict(mode="2D", msk="myolegs22", device="Tutorial_L1", leg="22", n=77), id="22-2D"),
    pytest.param(dict(mode="3D", msk="myolegs26", device="Tutorial_L1", leg="26", n=97), id="26-3D-planar_root"),
    pytest.param(
        dict(mode="2D", msk="myolegs22", device="KFoot_L1", leg="22", n=128, reflex_mode="amp"),
        id="amp-KFoot",
    ),
    pytest.param(
        dict(mode="2D", msk="myolegs22", device="KFoot_L1", leg="22", n=130, reflex_mode="amp", optimize_stiffness=True),
        id="amp-KFoot-stiffness",
    ),
    pytest.param(
        dict(mode="2D", msk="myolegs22", device="Anatomics_L1", leg="22", n=77, ankle_range=[-0.1745, 0.2618]),
        id="ROM-Anatomics",
    ),
    pytest.param(dict(mode="3D", msk="myolegs", device="Tutorial_L1", leg="80", n=97), id="80-3D"),
]


@pytest.mark.parametrize("case", CASES)
def test_reflex_env_builds_and_steps(case):
    env = _build(**case)  # myoLeg_reflex validates len(params) == layout.total on construction
    env.reset()
    assert env.check_pose_validity(), "initial pose invalid (would be the 1.2e6 sim-error)"
    steps = 0
    for i in range(int(5 / env.dt)):
        _, _, is_done = env.run_reflex_step_Cost()
        steps = i + 1
        if is_done:
            break
    assert steps >= 1
    assert np.all(np.isfinite(np.asarray(env.env.sim.data.qpos))), "non-finite qpos after stepping"
