"""The four-phase powered-prosthesis impedance controller (ctrl/prosthetic/impedance_ctrl.py).

The state machine and the torque law are pure Python, so the first tests drive them with
synthetic sensor data. The joint tests compose the three shipped powered devices and check
what the controller reads off each model: which joints are powered, the actuator capacity,
and the sign that maps a model joint to the controller convention (knee flexion and ankle
dorsiflexion positive). The sign is checked against a kinematic perturbation, not against a
comment about the model. The build-and-step tests live in test_co_build.py.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from ctrl_optim.ctrl.prosthetic.impedance_ctrl import (
    DEFAULT_GAINS,
    DEFAULT_THRESHOLDS,
    PHASES,
    ImpedanceStateMachine,
    ProsthesisImpedanceController,
    find_powered_joints,
    init_phase_for_pose,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "ctrl_optim" / "optim" / "training_configs"

# A stance and a swing value for each sensor, in the controller convention.
LOADED, UNLOADED = 0.6, 0.0  # body weights
DORSIFLEXED, NEUTRAL = np.deg2rad(10.0), 0.0
FLEXING = np.deg2rad(200.0)  # knee velocity in early swing, rad/s


def test_state_machine_goes_through_the_four_phases_in_order():
    fsm = ImpedanceStateMachine(["knee", "ankle"], init_phase="e_stance")
    seen = [fsm.phase]

    # Early stance needs load AND dorsiflexion: load alone (heel strike) does not leave it.
    assert fsm.update({"load": LOADED, "ankle_angle": NEUTRAL, "knee_angle": 0.1, "knee_vel": 0.0}) == "e_stance"
    # The ankle alone (a dorsiflexed foot in the air) does not leave it either.
    assert fsm.update({"load": UNLOADED, "ankle_angle": DORSIFLEXED, "knee_angle": 0.1, "knee_vel": 0.0}) == "e_stance"
    seen.append(fsm.update({"load": LOADED, "ankle_angle": DORSIFLEXED, "knee_angle": 0.1, "knee_vel": 0.0}))

    # Late stance ends when the foot unloads (toe-off).
    assert fsm.update({"load": LOADED, "ankle_angle": DORSIFLEXED, "knee_angle": 0.3, "knee_vel": FLEXING}) == "l_stance"
    seen.append(fsm.update({"load": UNLOADED, "ankle_angle": DORSIFLEXED, "knee_angle": 0.6, "knee_vel": FLEXING}))

    # Early swing ends at peak knee flexion (the knee velocity drops below 3 deg/s).
    assert fsm.update({"load": UNLOADED, "ankle_angle": NEUTRAL, "knee_angle": 0.7, "knee_vel": FLEXING}) == "e_swing"
    seen.append(fsm.update({"load": UNLOADED, "ankle_angle": NEUTRAL, "knee_angle": 0.75, "knee_vel": 0.0}))

    # Late swing ends when the knee extends below 30 deg (or at heel strike).
    assert fsm.update({"load": UNLOADED, "ankle_angle": NEUTRAL, "knee_angle": 0.7, "knee_vel": -FLEXING}) == "l_swing"
    seen.append(fsm.update({"load": UNLOADED, "ankle_angle": NEUTRAL, "knee_angle": 0.3, "knee_vel": -FLEXING}))

    assert seen == ["e_stance", "l_stance", "e_swing", "l_swing", "e_stance"]


def test_state_machine_moves_at_most_one_phase_per_update():
    # These sensors satisfy the exits of early swing and of late swing at once.
    fsm = ImpedanceStateMachine(["ankle"], init_phase="e_swing")
    assert fsm.update({"load": UNLOADED, "ankle_angle": NEUTRAL, "knee_angle": 0.1, "knee_vel": 0.0}) == "l_swing"
    assert fsm.update({"load": UNLOADED, "ankle_angle": NEUTRAL, "knee_angle": 0.1, "knee_vel": 0.0}) == "e_stance"


def test_state_machine_skips_an_absent_sensor():
    # A device without an ankle angle: early stance then exits on load alone.
    fsm = ImpedanceStateMachine(["knee"], init_phase="e_stance")
    assert fsm.update({"load": LOADED, "ankle_angle": None, "knee_angle": 0.1, "knee_vel": 0.0}) == "l_stance"


def test_torque_law_and_clipping():
    fsm = ImpedanceStateMachine(["ankle"], init_phase="l_stance")
    fsm.set_gain("ankle", "l_stance", stiffness=100.0, damping=2.0, target_angle=0.1)
    # tau = K (theta_eq - theta) - B dtheta
    assert fsm.torque("ankle", angle=0.0, velocity=1.0) == pytest.approx(100.0 * 0.1 - 2.0 * 1.0)
    assert fsm.torque("ankle", angle=0.3, velocity=-0.5) == pytest.approx(100.0 * (0.1 - 0.3) + 2.0 * 0.5)
    # Clipped symmetrically to the actuator capacity.
    assert fsm.torque("ankle", angle=-1.0, velocity=0.0, capacity=50.0) == pytest.approx(50.0)
    assert fsm.torque("ankle", angle=1.0, velocity=0.0, capacity=50.0) == pytest.approx(-50.0)
    # The gains of the other phases do not act.
    fsm.reset("e_swing")
    k, b, theta = DEFAULT_GAINS["ankle"]["e_swing"]
    assert fsm.torque("ankle", angle=0.0, velocity=0.0) == pytest.approx(k * theta)


def test_defaults_are_the_myosuite_init_defaults():
    """Gains and thresholds equal MyoOSLController.initDefaults (load thresholds as body weights)."""
    myoosl = pytest.importorskip("myosuite.envs.myo.assets.leg.myoosl_control")
    body_mass = 75.0
    ref = myoosl.MyoOSLController(body_mass).OSL_PARAM_LIST[0]
    weight = body_mass * 9.81
    for phase in PHASES:
        for joint, gains in DEFAULT_GAINS.items():
            k, b, theta = gains[phase]
            assert k == pytest.approx(ref[phase]["gain"][f"{joint}_stiffness"])
            assert b == pytest.approx(ref[phase]["gain"][f"{joint}_damping"])
            assert theta == pytest.approx(ref[phase]["gain"][f"{joint}_target_angle"])
        _, conditions = DEFAULT_THRESHOLDS[phase]
        ref_conditions = ref[phase]["threshold"]
        assert {c[0] for c in conditions} == set(ref_conditions)
        for key, value, direction in conditions:
            ref_value, ref_direction = ref_conditions[key]
            assert direction == ref_direction
            assert value == pytest.approx(ref_value / weight if key == "load" else ref_value)


def test_initial_phase_follows_the_reflex_initial_phase():
    # walk_left starts the right leg in swing and the left leg in stance (reflex_ctrl.reset_spinal_phases).
    assert init_phase_for_pose("walk_left", "r") == "e_swing"
    assert init_phase_for_pose("walk_left", "l") == "e_stance"
    assert init_phase_for_pose("walk_right", "r") == "e_stance"
    assert init_phase_for_pose("stand", "r") == "e_stance"


def _model(msk, device):
    from myoassist_utils.compose import compose_env_model

    return mujoco.MjModel.from_xml_string(compose_env_model(msk, device, terrain=None, planar_root=True))


# device -> (powered joint suffixes in parameter order, capacity per joint in Nm)
DEVICES = {
    "OpenSourceLeg_A_L1": (["osl_ankle_angle_r"], [58.4 * 2.88]),
    "OpenSourceLeg_KA_L1": (["osl_knee_angle_r", "osl_ankle_angle_r"], [49.4 * 2.88, 58.4 * 2.88]),
    "NEUankle_L1": (["neuankle_ankle_angle_r"], [50.0]),
}


def _perturbation_sign(model, joint_name, joint_type):
    """+1 if a positive qpos step dorsiflexes the ankle (toe rises against the heel) or flexes
    the knee (the foot moves backward), measured on the walk_left keyframe."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("walk_left").id)
    mujoco.mj_kinematics(model, data)
    toes, heel = (int(model.sensor(s).objid[0]) for s in ("r_toes", "r_foot"))
    before = data.site_xpos[[toes, heel]].copy()
    data.qpos[model.jnt_qposadr[model.joint(joint_name).id]] += 0.05
    mujoco.mj_kinematics(model, data)
    moved = data.site_xpos[[toes, heel]] - before
    if joint_type == "ankle":
        return float(np.sign(moved[0, 2] - moved[1, 2]))
    return float(np.sign(-moved[1, 0]))


@pytest.mark.parametrize("device", sorted(DEVICES))
def test_powered_joints_are_found_with_measured_signs(device):
    model = _model("myolegs22", device)
    suffixes, capacities = DEVICES[device]
    joints = find_powered_joints(model)
    assert [j.name.split("_L1_")[-1] for j in joints] == suffixes
    for joint, capacity in zip(joints, capacities):
        assert joint.side == "r"
        assert joint.capacity == pytest.approx(capacity)
        assert model.actuator(joint.actuator_id).trnid[0] == model.joint(joint.name).id
        assert joint.sign == _perturbation_sign(model, joint.name, joint.joint_type)
        # Every OSL and NEUankle joint already uses the controller convention.
        assert joint.sign == 1.0
        lo, hi = joint.angle_range
        for phase in PHASES:  # the OSL default targets lie inside the joint range
            assert lo <= DEFAULT_GAINS[joint.joint_type][phase][2] <= hi

    ctrl = ProsthesisImpedanceController(model)
    assert ctrl.n_params == 12 * len(suffixes)
    if "KA" not in device:
        # Transtibial: the knee sensor is the person's own knee, with negative flexion on myolegs22.
        ((q_adr, _, sign),) = ctrl.sources["r"]["knee"]
        assert q_adr == model.jnt_qposadr[model.joint("knee_angle_r").id]
        assert sign == _perturbation_sign(model, "knee_angle_r", "knee") == -1.0


def test_exoskeleton_is_not_a_prosthesis():
    """An exo actuator on an intact joint is not a powered prosthetic joint."""
    model = _model("myolegs22", "Tutorial_L1")
    assert find_powered_joints(model) == []
    with pytest.raises(ValueError):
        ProsthesisImpedanceController(model)


def test_parameter_tail_maps_onto_the_ranges():
    ctrl = ProsthesisImpedanceController(_model("myolegs22", "OpenSourceLeg_KA_L1"))
    # The normalized defaults map back onto the initDefaults gains.
    gains = ctrl.denormalize(ctrl.default_params())
    for joint in ctrl.joints:
        for phase in PHASES:
            assert gains[joint.name][phase] == pytest.approx(DEFAULT_GAINS[joint.joint_type][phase], abs=1e-12)
    # 0 and 1 are the ends of each range; the tail is joint, then phase, then (K, B, theta_eq).
    zeros, ones = ctrl.denormalize(np.zeros(24)), ctrl.denormalize(np.ones(24))
    knee, ankle = ctrl.joints
    assert zeros[knee.name]["e_stance"] == (0.0, 0.0, knee.angle_range[0])
    assert ones[ankle.name]["l_swing"] == pytest.approx((400.0, 1.0, ankle.angle_range[1]))
    tail = np.zeros(24)
    tail[12 + 3 * PHASES.index("l_stance")] = 0.5  # ankle, late stance, stiffness
    assert ctrl.denormalize(tail)[ankle.name]["l_stance"][0] == pytest.approx(200.0)
    # The description of the joints and ranges can be stored as JSON next to the results.
    import json

    assert json.loads(json.dumps(ctrl.describe()))["joints"][0]["name"] == knee.name


@pytest.mark.parametrize("name", ["osl_a_impedance", "osl_ka_impedance", "neuankle_impedance"])
@pytest.mark.parametrize("ext", ["bat", "sh"])
def test_shipped_config_replays_with_the_controller(name, ext):
    """Evaluation rebuilds a run from its config, so the config must carry the option."""
    from ctrl_optim.optim.optim_utils.config_parser import parse_bat_config

    cfg = parse_bat_config(str(CONFIG_DIR / f"{name}.{ext}"))
    assert cfg["optimize_impedance"] is True
    assert cfg["reflex_mode"] == "amp"
    assert cfg["msk_key"] == "myolegs22"
    assert not cfg["optimize_stiffness"] and not cfg["exo_bool"]


def test_substeps_integrate_like_env_step():
    """With the torque held, the substep loop of _step_env gives the same state as env.step.

    So the only change for a prosthesis run is the torque update before each substep.
    """
    from ctrl_optim.ctrl.reflex.reflex_interface import myoLeg_reflex

    def build():
        env = myoLeg_reflex(
            sim_time=1,
            control_params=np.ones(140),
            mode="2D",
            init_pose="walk_left",
            delayed=False,
            msk_key="myolegs22",
            device_key="OpenSourceLeg_A_L1",
            leg_model="22",
            reflex_mode="amp",
            exo_bool=False,
            optimize_impedance=True,
        )
        params = np.ones(140)
        params[env.layout.slice_impedance()] = env.ProsCtrl.default_params()
        env.reset(params)
        return env

    held, plain = build(), build()
    ctrl = held.ProsCtrl
    write = ctrl.write_controls

    def apply(data, loads, act):  # the control-step update and write, unchanged
        ctrl.update_phases(data, loads)
        return write(data, act)

    ctrl.apply = apply
    ctrl.write_controls = lambda data, act: act  # no torque update between substeps
    plain._step_env = plain.env.step
    for _ in range(10):
        held.run_reflex_step_Cost()
        plain.run_reflex_step_Cost()
    for field in ("qpos", "qvel", "act", "sensordata"):
        assert np.array_equal(getattr(held.env.sim.data, field), getattr(plain.env.sim.data, field)), field
