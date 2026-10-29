"""Four-phase finite-state impedance controller for a powered prosthesis.

The controller follows MyoSuite's ``MyoOSLController`` (the Open-Source Leg
project defaults).  It is not specific to one device.  It finds the powered
prosthetic joints (knee, ankle, or both) in the composed model and drives each
joint actuator with the same four-phase state machine.

Phases, in order: early stance (``e_stance``), late stance (``l_stance``), early
swing (``e_swing``) and late swing (``l_swing``).  In each phase every joint has a
stiffness K, a damping B and a target angle theta_eq.  The joint torque is::

    tau = K * (theta_eq - theta) - B * dtheta,   clipped to +-capacity
    ctrl = sign * tau / gain

Conventions (controller frame, the MyoSuite and Open-Source Leg convention):

- Knee: a positive angle is flexion.  0 is full extension.
- Ankle: a positive angle is dorsiflexion.  0 is neutral.
- A positive torque acts in the positive-angle direction.

The model joints do not all use this convention.  The biological knee of the 22-
and 26-muscle models has negative flexion.  The 80-muscle knee and the
Open-Source Leg knee have positive flexion.  The code does not assume a sign per
device.  It measures the
sign from the composed model: hip flexion, knee extension and ankle dorsiflexion
turn about the same lateral axis.  So the controller angle is ``sign * qpos``,
with ``sign`` from the dot product of the joint axis and the hip-flexion axis of
the same side.  The tests check this rule against a kinematic perturbation.

Transitions read four sensors: the vertical ground force under the prosthetic
foot (``load``, in body weights), the knee angle and velocity, and the ankle
angle.  A powered joint is its own sensor.  When the knee is not powered (a
transtibial device), the knee sensor is the person's own knee on that side.
The thresholds are the ``initDefaults`` values.  ``initDefaults`` gives the load
thresholds as a fraction of body weight in newtons; here the load sensor is
already in body weights, so the thresholds are the fractions.  The machine moves
at most one phase per update.  One condition is enough to leave a phase, as in
``MyoOSLController``, except early stance (see ``DEFAULT_THRESHOLDS``).

The phases update once per control step.  The torque law runs before each
physics substep (see ``ProsthesisImpedanceController``), as a motor driver runs
its impedance loop faster than the state machine.

The optimizer tunes only the gains: K, B and theta_eq per phase per joint, 12
values per joint.  Each value is normalized to ``[0, 1]`` over a fixed range, as
the other CO device tails are.  The thresholds stay fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

from ctrl_optim.ctrl.prosthetic.ankle_stiffness import denormalize

PHASES = ("e_stance", "l_stance", "e_swing", "l_swing")
JOINT_TYPES = ("knee", "ankle")  # joint order in the parameter tail
GAIN_NAMES = ("stiffness", "damping", "target_angle")
N_PARAMS_PER_JOINT = len(PHASES) * len(GAIN_NAMES)  # 12

# Default gains from MyoOSLController.initDefaults, per phase:
# (stiffness Nm/rad, damping Nm s/rad, target angle rad), controller convention.
DEFAULT_GAINS = {
    "knee": {
        "e_stance": (99.372, 3.180, np.deg2rad(5.0)),
        "l_stance": (99.372, 1.272, np.deg2rad(8.0)),
        "e_swing": (39.749, 0.063, np.deg2rad(60.0)),
        "l_swing": (15.899, 3.816, np.deg2rad(5.0)),
    },
    "ankle": {
        "e_stance": (19.874, 0.0, np.deg2rad(-2.0)),
        "l_stance": (79.498, 0.063, np.deg2rad(-20.0)),
        "e_swing": (7.949, 0.0, np.deg2rad(25.0)),
        "l_swing": (7.949, 0.0, np.deg2rad(15.0)),
    },
}

# Transition thresholds from MyoOSLController.initDefaults, per phase:
# (rule, ((sensor, threshold, "above" | "below"), ...)).  Load is in body weights,
# angles in rad, velocity in rad/s.  With rule "any", one satisfied condition moves
# the machine to the next phase; with "all", every condition must be satisfied.
#
# MyoOSLController uses "any" for every phase.  Early stance uses "all" here, as the
# Open-Source Leg project's own state machine does (load AND ankle angle).  With
# "any", early stance starts in late swing (knee below 30 deg), the late-swing target
# holds the ankle near 15 deg (above the 6 deg threshold), and the machine goes
# through all four phases in the air before heel strike.  Early swing keeps "any":
# with "all", a knee that peaks below 50 deg holds the machine in early swing for good.
DEFAULT_THRESHOLDS = {
    "e_stance": ("all", (("load", 0.25, "above"), ("ankle_angle", np.deg2rad(6.0), "above"))),
    "l_stance": ("any", (("load", 0.15, "below"),)),
    "e_swing": ("any", (("knee_angle", np.deg2rad(50.0), "above"), ("knee_vel", np.deg2rad(3.0), "below"))),
    "l_swing": ("any", (("load", 0.4, "above"), ("knee_angle", np.deg2rad(30.0), "below"))),
}

# Denormalization ranges of the optimized gains.  The stiffness range holds the
# OSL defaults and the stance quasi-stiffness of a biological knee or ankle.  The
# target-angle range is the range of the prosthetic joint in the model; the values
# below apply only to a joint that has no range.
STIFFNESS_RANGE = {"knee": (0.0, 400.0), "ankle": (0.0, 400.0)}  # Nm/rad
DAMPING_RANGE = {"knee": (0.0, 10.0), "ankle": (0.0, 1.0)}  # Nm s/rad
FALLBACK_ANGLE_RANGE = {"knee": (0.0, np.deg2rad(120.0)), "ankle": (np.deg2rad(-30.0), np.deg2rad(30.0))}

# The biological joint names of the MSK models.  A prosthetic joint replaces one
# of them, so the device joint name ends with the same suffix.
BIO_JOINT = {"knee": "knee_angle_{side}", "ankle": "ankle_angle_{side}"}
SIDES = ("r", "l")


def _next_phase(phase: str) -> str:
    return PHASES[(PHASES.index(phase) + 1) % len(PHASES)]


class ImpedanceStateMachine:
    """The four-phase state machine and the torque law, without MuJoCo.

    ``gains[joint_type][phase]`` is ``(K, B, theta_eq)`` in the controller convention.
    """

    def __init__(self, joint_types: Sequence[str], thresholds=DEFAULT_THRESHOLDS, init_phase: str = "e_stance"):
        unknown = set(joint_types) - set(JOINT_TYPES)
        if unknown:
            raise ValueError(f"joint types must be in {JOINT_TYPES}; got {sorted(unknown)}")
        self.joint_types = tuple(joint_types)
        self.thresholds = thresholds
        self.gains = {jt: dict(DEFAULT_GAINS[jt]) for jt in self.joint_types}
        self.reset(init_phase)

    def reset(self, init_phase: str = "e_stance") -> None:
        if init_phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}; got {init_phase!r}")
        self.phase = init_phase

    def set_gain(self, joint_type: str, phase: str, stiffness: float, damping: float, target_angle: float) -> None:
        self.gains[joint_type][phase] = (float(stiffness), float(damping), float(target_angle))

    def update(self, sensors: Dict[str, Optional[float]]) -> str:
        """Check the transition conditions of the current phase and return the phase.

        The machine moves at most one phase per call.  A condition whose sensor is
        absent or ``None`` is skipped.
        """
        rule, conditions = self.thresholds[self.phase]
        results = []
        for key, threshold, condition in conditions:
            value = sensors.get(key)
            if value is None:
                continue
            results.append(value > threshold if condition == "above" else value < threshold)
        if results and (all(results) if rule == "all" else any(results)):
            self.phase = _next_phase(self.phase)
        return self.phase

    def torque(self, joint_type: str, angle: float, velocity: float, capacity: float = np.inf) -> float:
        """Impedance torque of one joint in the current phase, clipped to +-capacity."""
        k, b, theta_eq = self.gains[joint_type][self.phase]
        return float(np.clip(k * (theta_eq - angle) - b * velocity, -capacity, capacity))


@dataclass(frozen=True)
class PoweredJoint:
    """A powered prosthetic joint and its actuator, read from the composed model."""

    name: str
    joint_type: str  # "knee" or "ankle"
    side: str  # "r" or "l"
    sign: float  # controller angle = sign * qpos
    qpos_adr: int
    dof_adr: int
    actuator_id: int
    actuator_name: str
    gain: float  # joint torque per unit of control
    capacity: float  # largest torque magnitude, Nm
    angle_range: Tuple[float, float]  # controller convention, rad


def _lateral_axes(model) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """World joint axes at qpos0 and the hip-flexion axis per side."""
    raw = getattr(model, "ptr", model)  # the env gives a dm_control wrapper; MjData needs the raw MjModel
    data = mujoco.MjData(raw)
    mujoco.mj_kinematics(raw, data)
    hip = {}
    for side in SIDES:
        try:
            hip[side] = data.xaxis[model.joint(f"hip_flexion_{side}").id].copy()
        except KeyError:
            pass
    return data.xaxis.copy(), hip


def _joint_sign(xaxis: np.ndarray, hip_axis: np.ndarray, joint_id: int, joint_type: str) -> float:
    """Sign that maps ``qpos`` to the controller convention.

    Hip flexion, knee extension and ankle dorsiflexion turn about the same lateral
    axis.  So an ankle axis parallel to the hip-flexion axis is dorsiflexion
    positive, and a knee axis parallel to it is extension positive.
    """
    dot = float(np.dot(xaxis[joint_id], hip_axis))
    if abs(dot) < 0.5:
        raise ValueError(f"joint {joint_id} is not a sagittal hinge (axis . hip-flexion axis = {dot:+.2f})")
    parallel = 1.0 if dot > 0 else -1.0
    return parallel if joint_type == "ankle" else -parallel


def joint_sign(model, joint_name: str, joint_type: str) -> float:
    """Sign that maps the qpos of a knee or ankle joint to the controller convention
    (knee flexion positive, ankle dorsiflexion positive).  The side comes from the
    joint name suffix (``_r`` / ``_l``)."""
    side = joint_name[-1]
    xaxis, hip = _lateral_axes(model)
    if side not in hip:
        raise ValueError(f"no hip_flexion_{side} joint to fix the sign of {joint_name}")
    return _joint_sign(xaxis, hip[side], model.joint(joint_name).id, joint_type)


def _joint_type_and_side(name: str) -> Optional[Tuple[str, str]]:
    for joint_type, pattern in BIO_JOINT.items():
        for side in SIDES:
            if name.endswith(pattern.format(side=side)):
                return joint_type, side
    return None


def _capacity(model, act_id: int) -> Tuple[float, float]:
    """Return (gain, capacity) of a joint actuator: torque per control unit and the
    largest torque magnitude (gain x largest control magnitude)."""
    gain = float(model.actuator_gainprm[act_id, 0] * model.actuator_gear[act_id, 0])
    ctrl_max = np.inf
    if model.actuator_ctrllimited[act_id]:
        ctrl_max = float(np.max(np.abs(model.actuator_ctrlrange[act_id])))
    capacity = abs(gain) * ctrl_max
    if model.actuator_forcelimited[act_id]:
        force_max = float(np.max(np.abs(model.actuator_forcerange[act_id])))
        capacity = min(capacity, force_max * abs(float(model.actuator_gear[act_id, 0])))
    return gain, capacity


def find_powered_joints(model) -> List[PoweredJoint]:
    """Find the powered prosthetic joints of a composed model.

    A powered prosthetic joint is a hinge that a non-muscle actuator drives through
    a joint transmission, whose name ends like a biological knee or ankle
    (``knee_angle_r``, ``ankle_angle_l``, ...), and whose biological joint is absent
    (the device amputated it).  Exoskeleton actuators on intact joints do not match.
    The list is in parameter order: right side first, then knee before ankle.
    """
    joint_names = {model.joint(i).name for i in range(model.njnt)}
    found = []
    for act_id in range(model.nu):
        if model.actuator_dyntype[act_id] == mujoco.mjtDyn.mjDYN_MUSCLE:
            continue
        if model.actuator_trntype[act_id] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        jid = int(model.actuator_trnid[act_id, 0])
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = model.joint(jid).name
        match = _joint_type_and_side(name)
        if match is None:
            continue
        joint_type, side = match
        if name == BIO_JOINT[joint_type].format(side=side) or BIO_JOINT[joint_type].format(side=side) in joint_names:
            continue  # an intact biological joint, not a prosthesis
        found.append((side, joint_type, jid, act_id))

    if not found:
        return []
    xaxis, hip = _lateral_axes(model)
    joints = []
    for side, joint_type, jid, act_id in found:
        if side not in hip:
            raise ValueError(f"no hip_flexion_{side} joint to fix the sign of {model.joint(jid).name}")
        sign = _joint_sign(xaxis, hip[side], jid, joint_type)
        if model.jnt_limited[jid]:
            lo, hi = sorted(sign * model.jnt_range[jid])
        else:
            lo, hi = FALLBACK_ANGLE_RANGE[joint_type]
        gain, capacity = _capacity(model, act_id)
        joints.append(
            PoweredJoint(
                name=model.joint(jid).name,
                joint_type=joint_type,
                side=side,
                sign=sign,
                qpos_adr=int(model.jnt_qposadr[jid]),
                dof_adr=int(model.jnt_dofadr[jid]),
                actuator_id=act_id,
                actuator_name=model.actuator(act_id).name,
                gain=gain,
                capacity=capacity,
                angle_range=(float(lo), float(hi)),
            )
        )
    joints.sort(key=lambda j: (SIDES.index(j.side), JOINT_TYPES.index(j.joint_type)))
    return joints


def _angle_source(model, side: str, joint_type: str, powered: List[PoweredJoint], xaxis, hip_axis):
    """Joints whose summed angle is the knee or ankle sensor of one side.

    The order of preference is: the powered prosthetic joint, the biological joint,
    then passive device joints of that type (summed, as for a df/pf spring ankle).
    Returns a list of ``(qpos_adr, dof_adr, sign)``, or ``None`` if the side has no
    such joint.
    """
    for j in powered:
        if j.side == side and j.joint_type == joint_type:
            return [(j.qpos_adr, j.dof_adr, j.sign)]
    bio = BIO_JOINT[joint_type].format(side=side)
    names = [model.joint(i).name for i in range(model.njnt)]
    candidates = [bio] if bio in names else [n for n in names if n.endswith(bio)]
    source = []
    for name in candidates:
        jid = model.joint(name).id
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        sign = _joint_sign(xaxis, hip_axis, jid, joint_type)
        source.append((int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]), sign))
    return source or None


def _read(data, source) -> Tuple[Optional[float], Optional[float]]:
    if source is None:
        return None, None
    angle = sum(sign * float(data.qpos[q]) for q, _, sign in source)
    vel = sum(sign * float(data.qvel[d]) for _, d, sign in source)
    return angle, vel


def init_phase_for_pose(init_pose: str, side: str) -> str:
    """Initial phase of a leg, matched to the reflex controller's initial phases.

    ``walk_left`` starts the right leg in swing and the left leg in stance;
    ``walk_right`` is the mirror.  Other poses start in early stance.
    """
    swing_leg = {"walk_left": "r", "walk_right": "l"}.get(init_pose)
    if swing_leg is None:
        return "e_stance"
    return "e_swing" if side == swing_leg else "e_stance"


class ProsthesisImpedanceController:
    """Drive every powered prosthetic joint of a composed model.

    Use one state machine per prosthetic side.  ``apply`` updates the phases from
    the sensors and writes the actuator controls into the control vector.  Call it
    once per control step, after the reflex controls are set.

    Call ``write_controls`` before each physics substep of that control step.  It
    applies the torque law to the current joint state with the phases unchanged.
    The prosthetic foot is light (about 0.003 kg m2 about the ankle), so a torque
    held for a full 20 ms control step is not an impedance: the default late-stance
    torque drives the unloaded foot through its range in one step.
    """

    def __init__(self, model):
        self.joints = find_powered_joints(model)
        if not self.joints:
            raise ValueError(
                "no powered prosthetic joint found: the model has no non-muscle actuator on a "
                "device knee or ankle joint that replaces an amputated biological joint"
            )
        self.sides = tuple(s for s in SIDES if any(j.side == s for j in self.joints))
        self.fsm = {side: ImpedanceStateMachine([j.joint_type for j in self.joints if j.side == side]) for side in self.sides}
        xaxis, hip = _lateral_axes(model)
        self.sources = {}
        for side in self.sides:
            self.sources[side] = {jt: _angle_source(model, side, jt, self.joints, xaxis, hip[side]) for jt in JOINT_TYPES}
            if self.sources[side]["knee"] is None:
                raise ValueError(f"side {side!r} has no knee joint; the swing transitions need a knee angle")
        self.last_torque = {j.name: 0.0 for j in self.joints}

    # -- parameters -------------------------------------------------------------
    @property
    def n_params(self) -> int:
        return N_PARAMS_PER_JOINT * len(self.joints)

    def _ranges(self, joint: PoweredJoint):
        return (STIFFNESS_RANGE[joint.joint_type], DAMPING_RANGE[joint.joint_type], joint.angle_range)

    def denormalize(self, params) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
        """Map the normalized tail to ``{joint name: {phase: (K, B, theta_eq)}}``.

        The tail is joint-major (the order of ``self.joints``), then phase (``PHASES``),
        then ``(K, B, theta_eq)``.  Each value in ``[0, 1]`` (clipped) maps linearly
        onto its range.
        """
        params = np.asarray(params, dtype=float)
        if params.shape != (self.n_params,):
            raise ValueError(f"expected {self.n_params} impedance parameters, got {params.shape}")
        gains = {}
        for j_idx, joint in enumerate(self.joints):
            ranges = self._ranges(joint)
            gains[joint.name] = {}
            for p_idx, phase in enumerate(PHASES):
                base = j_idx * N_PARAMS_PER_JOINT + p_idx * len(GAIN_NAMES)
                gains[joint.name][phase] = tuple(denormalize(params[base + g], *ranges[g]) for g in range(len(GAIN_NAMES)))
        return gains

    def default_params(self) -> np.ndarray:
        """The ``initDefaults`` gains as a normalized tail (the CMA-ES start)."""
        out = np.zeros(self.n_params)
        for j_idx, joint in enumerate(self.joints):
            ranges = self._ranges(joint)
            for p_idx, phase in enumerate(PHASES):
                base = j_idx * N_PARAMS_PER_JOINT + p_idx * len(GAIN_NAMES)
                for g, value in enumerate(DEFAULT_GAINS[joint.joint_type][phase]):
                    lo, hi = ranges[g]
                    out[base + g] = np.clip((value - lo) / (hi - lo), 0.0, 1.0)
        return out

    def set_params(self, params) -> None:
        for name, phases in self.denormalize(params).items():
            joint = next(j for j in self.joints if j.name == name)
            for phase, (k, b, theta) in phases.items():
                self.fsm[joint.side].set_gain(joint.joint_type, phase, k, b, theta)

    # -- run time ---------------------------------------------------------------
    def reset(self, init_pose: str) -> None:
        for side in self.sides:
            self.fsm[side].reset(init_phase_for_pose(init_pose, side))
        self.last_torque = {j.name: 0.0 for j in self.joints}

    def sensors(self, data, side: str, load: float) -> Dict[str, Optional[float]]:
        knee_angle, knee_vel = _read(data, self.sources[side]["knee"])
        ankle_angle, ankle_vel = _read(data, self.sources[side]["ankle"])
        return {
            "load": load,
            "knee_angle": knee_angle,
            "knee_vel": knee_vel,
            "ankle_angle": ankle_angle,
            "ankle_vel": ankle_vel,
        }

    def update_phases(self, data, loads: Dict[str, float]) -> None:
        """Update the state machine of each side.  ``loads`` maps a side (``"r"`` /
        ``"l"``) to the vertical ground force under that foot, in body weights."""
        for side in self.sides:
            self.fsm[side].update(self.sensors(data, side, loads[side]))

    def write_controls(self, data, act: np.ndarray) -> np.ndarray:
        """Apply the torque law of the current phase to the current joint state and
        write the controls into ``act`` (in place): ctrl = sign * tau / gain."""
        for joint in self.joints:
            angle = joint.sign * float(data.qpos[joint.qpos_adr])
            vel = joint.sign * float(data.qvel[joint.dof_adr])
            tau = self.fsm[joint.side].torque(joint.joint_type, angle, vel, joint.capacity)
            self.last_torque[joint.name] = tau
            act[joint.actuator_id] = joint.sign * tau / joint.gain
        return act

    def apply(self, data, loads: Dict[str, float], act: np.ndarray) -> np.ndarray:
        """Update the phases, then write the prosthesis controls into ``act`` (in place)."""
        self.update_phases(data, loads)
        return self.write_controls(data, act)

    @property
    def phases(self) -> Dict[str, str]:
        return {side: self.fsm[side].phase for side in self.sides}

    def describe(self) -> dict:
        """Joints, conventions and ranges, to store next to the results."""
        return {
            "phases": list(PHASES),
            "param_order": "joint (listed order), then phase, then (stiffness, damping, target_angle)",
            "thresholds": {
                p: {"rule": DEFAULT_THRESHOLDS[p][0], "conditions": [list(c) for c in DEFAULT_THRESHOLDS[p][1]]} for p in PHASES
            },
            "joints": [
                {
                    "name": j.name,
                    "type": j.joint_type,
                    "side": j.side,
                    "sign": j.sign,
                    "actuator": j.actuator_name,
                    "capacity_nm": j.capacity,
                    "stiffness_range_nm_per_rad": list(STIFFNESS_RANGE[j.joint_type]),
                    "damping_range_nm_s_per_rad": list(DAMPING_RANGE[j.joint_type]),
                    "target_angle_range_rad": list(j.angle_range),
                }
                for j in self.joints
            ],
        }


@lru_cache(maxsize=8)
def _env_tail(msk_key: str, device_key: str) -> Tuple[int, Tuple[float, ...]]:
    from myoassist_utils.compose import compose_env_model

    # The prosthetic joints do not depend on the terrain, so compose on flat ground.
    model = mujoco.MjModel.from_xml_string(compose_env_model(msk_key, device_key, terrain=None, planar_root=True))
    ctrl = ProsthesisImpedanceController(model)
    return ctrl.n_params, tuple(ctrl.default_params())


def n_params_for_env(msk_key: str, device_key: str) -> int:
    """Number of impedance parameters of a composed env (12 per powered joint)."""
    return _env_tail(msk_key, device_key)[0]


def default_params_for_env(msk_key: str, device_key: str) -> np.ndarray:
    """Normalized ``initDefaults`` tail of a composed env (the CMA-ES start)."""
    return np.array(_env_tail(msk_key, device_key)[1])
