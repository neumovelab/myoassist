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

The ``+impedance`` cases run the powered prostheses (OpenSourceLeg_A, OpenSourceLeg_KA,
NEUankle) under the four-phase impedance controller, in 2D on myolegs22 and in 3D on
myolegs26. Their impedance tail starts at the initDefaults gains, as ``train.py`` seeds it.
The test also checks that the controller, not the 0.01 reflex default, sets the prosthetic
actuator controls, and that the actuators produce torque.

The 80-muscle ``myolegs`` 3D case also guards the runtime seat
(``adjust_model_height``): the reflex env is always planar-root, so its vertical DOF
is ``pelvis_ty``, not ``qpos[2]`` (the freejoint z on the RL build). Seating the wrong
axis in 3D used to leave the 80-muscle feet ~7 cm off the ground with all foot GRF 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ctrl_optim.ctrl.reflex.reflex_interface import myoLeg_reflex
from ctrl_optim.optim.cost_functions.evaluate_cost import EFFORT_GROUPS, evaluateCost

REF_DIR = Path(__file__).resolve().parents[2] / "ctrl_optim" / "optim" / "ref_data"


def _build(mode, msk, device, leg, n, reflex_mode=None, optimize_stiffness=False, ankle_range=None, optimize_impedance=False):
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
        optimize_impedance=optimize_impedance,
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
    pytest.param(
        dict(
            mode="2D", msk="myolegs22", device="OpenSourceLeg_A_L1", leg="22", n=140, reflex_mode="amp", optimize_impedance=True
        ),
        id="amp-OSL_A-impedance",
    ),
    pytest.param(
        dict(
            mode="2D",
            msk="myolegs22",
            device="OpenSourceLeg_KA_L1",
            leg="22",
            n=152,
            reflex_mode="amp",
            optimize_impedance=True,
        ),
        id="amp-OSL_KA-impedance",
    ),
    pytest.param(
        dict(mode="2D", msk="myolegs22", device="NEUankle_L1", leg="22", n=140, reflex_mode="amp", optimize_impedance=True),
        id="amp-NEUankle-impedance",
    ),
    pytest.param(
        dict(
            mode="3D",
            msk="myolegs26",
            device="OpenSourceLeg_KA_L1",
            leg="26",
            n=184,
            reflex_mode="amp",
            optimize_impedance=True,
        ),
        id="26-3D-amp-OSL_KA-impedance",
    ),
]


@pytest.mark.parametrize("case", CASES)
def test_reflex_env_builds_and_steps(case):
    env = _build(**case)  # myoLeg_reflex validates len(params) == layout.total on construction
    params = np.ones(case["n"])
    if env.ProsCtrl is not None:
        assert env.layout.impedance == env.ProsCtrl.n_params == 12 * len(env.ProsCtrl.joints)
        params[env.layout.slice_impedance()] = env.ProsCtrl.default_params()
    env.reset(params)
    assert env.check_pose_validity(), "initial pose invalid (would be the 1.2e6 sim-error)"
    steps = 0
    store = []
    pros_force = []
    for i in range(int(5 / env.dt)):
        cost_dict, sim_time, is_done = env.run_reflex_step_Cost()
        store.append({"obj_func_out": cost_dict.copy(), "sim_time": sim_time})
        if env.ProsCtrl is not None:
            data = env.env.sim.data
            for joint in env.ProsCtrl.joints:  # the last substep's control is the controller's torque law
                tau = env.ProsCtrl.last_torque[joint.name]
                assert data.ctrl[joint.actuator_id] == pytest.approx(joint.sign * tau / joint.gain)
            pros_force.append([abs(data.actuator_force[j.actuator_id]) for j in env.ProsCtrl.joints])
        steps = i + 1
        if is_done:
            break
    assert steps >= 1
    assert np.all(np.isfinite(np.asarray(env.env.sim.data.qpos))), "non-finite qpos after stepping"
    if env.ProsCtrl is not None:
        peak = np.max(pros_force, axis=0)
        assert np.all(peak > 1.0), f"prosthetic actuators produced no torque: peak |force| {peak} Nm"

    # The cost evaluation unpacks every frame of the rollout (in 3D also the hip adduction
    # and rotation and the pelvis list and rotation) before it checks the strides. A time
    # limit equal to the rollout length makes it unpack even a rollout that fell.
    one_step = np.genfromtxt(REF_DIR / "ref_kinematics_radians.csv", delimiter=",")
    one_emg = np.genfromtxt(REF_DIR / "ref_EMG.csv", delimiter=",")
    cost = evaluateCost(
        store, env.dt, env.mode, len(store), 0, env.muscles_dict, "Kine", one_step, one_emg, "ref_diff", 1.25, 5, 0.1, 1.5
    )
    assert np.isfinite(cost)

    # Every effort group is a group of the controller (a name mismatch silently drops it).
    # On a prosthetic side, groups can be empty or absent, so check the intact left leg.
    missing = set(EFFORT_GROUPS[env.mode]) - set(env.muscles_dict["l_leg"])
    assert not missing, f"effort groups not in muscles_dict: {missing}"


def test_rom_clamp_applies_to_the_device_side_only():
    """The ROM clamp holds only the ankle of a side that carries the ROM device.

    The Anatomics ROM frame is on the right shank; its soles and uppers on both feet
    are shoes. Clamping both ankles constrained the intact left ankle as well.
    """
    env = _build(mode="2D", msk="myolegs22", device="Anatomics_L1", leg="22", n=77, ankle_range=[-0.1745, 0.2618])
    assert env._rom_ankle_joints() == ("ankle_angle_r",)


def test_rom_clamp_needs_a_device_on_the_shank():
    """ankle_range with a device that has no shank part is an error, not a silent no-op."""
    env = _build(mode="2D", msk="myolegs22", device="HMEDI_L1", leg="22", n=77, ankle_range=[-0.1745, 0.2618])
    with pytest.raises(ValueError, match="no body on a tibia"):
        env._rom_ankle_joints()
