"""Reproduce four measurements on the composed device models.

Findings from RL work on `myolegs22 + <device>`, written so the model owner can re-run them
rather than take the numbers on trust. Nothing here modifies anything; it composes each model
through the shared pipeline and measures it.

    .venv/bin/python tools/model_checks/check_prosthesis_model.py

1. SEATING. `compose._model_ground_candidates` estimates a primitive's underside as
   `centre_z - max(geom_size)`. For a capsule `size` is `(radius, half-length)` about its local z
   axis, so a horizontal capsule reaches only its radius below centre. The OpenSourceLeg foot
   contacts are capsules of radius 0.012 m and half-length 0.11 m laid flat, so the estimate is
   9.8 cm too low and the seater rests the model on a point that is not there. Both OpenSourceLeg
   compositions open with no ground contact at all. Every other device's foot contact is a mesh
   and is measured from its transformed vertices.

2. JOINT DEFAULTS. `myolegs26_assets.xml` puts `<joint armature="1e-05" damping="0.05"/>` in its
   `<default class="main">`, so every human joint inherits it. `NEUankle/L1model.xml` and
   `KFoot/L1model.xml` repeat that line in their own `class="main"` and their joints inherit it
   too. `OpenSourceLeg/A_L1model.xml` and `KA_L1model.xml` declare `class="main"` holding only
   `coll` and `myo_leg_touch`, with no `<joint>`, so on merge their joints fall back to MuJoCo's
   built-ins: armature 0, damping 0. The legacy `myo_sim/models/legacy/osl` model gave the same
   joints armature 0.01 and damping 0.5, so this is a regression against that model as well as an
   inconsistency with the two sibling devices.

3. JOINT LIMIT UNDER ACTUATOR TORQUE. Holding an actuator at full scale drives its joint well past
   its own range and keeps it there, on every powered device including the exos on human joints.
   The implied limit stiffness is near 120-130 N*m/rad against device torques of 50-168, so this
   is MuJoCo's default solref/solimp meeting torques the muscle-driven model never produced. Not
   an inertia effect: the overshoot is unchanged across two decades of added armature, because at
   steady state inertia does not enter.

4. DISTAL INERTIA. Reported without a cause attached: the prosthetic ankles carry about a fifth of
   the intact ankle's rotational inertia, one distal body against four.
"""

import numpy as np

try:
    import mujoco as mj
except ImportError as exc:  # pragma: no cover
    raise SystemExit("needs the project venv: .venv/bin/python tools/model_checks/check_prosthesis_model.py") from exc

from myoassist_utils.compose import compose_env_model

MSK = "myolegs22"
DEVICES = ["Tutorial_L1", "STRIDE_L2", "KFoot_L1", "NEUankle_L1", "OpenSourceLeg_A_L1", "OpenSourceLeg_KA_L1"]
POWERED = [
    ("DephyExoBoot_L1", "Exo_R", "ankle_angle_r"),
    ("Humotech_L1", "Exo_R", "ankle_angle_r"),
    ("OpenExo_L1", "Exo_R", "ankle_angle_r"),
    ("Hippo_L1", "Exo_R", "hip_flexion_r"),
    ("NEUankle_L1", "neuankle_ankle_torque_actuator", "NEUankle_L1_neuankle_ankle_angle_r"),
    ("OpenSourceLeg_A_L1", "osl_ankle_torque_actuator", "OSL_A_L1_osl_ankle_angle_r"),
    ("OpenSourceLeg_KA_L1", "osl_ankle_torque_actuator", "OSL_KA_L1_osl_ankle_angle_r"),
]
ANKLE = {
    "Tutorial_L1": ("ankle_angle_r", None),
    "STRIDE_L2": ("ankle_angle_r", None),
    "KFoot_L1": ("KFoot_L1_df_ankle_angle_r", None),
    "NEUankle_L1": ("NEUankle_L1_neuankle_ankle_angle_r", "neuankle_ankle_torque_actuator"),
    "OpenSourceLeg_A_L1": ("OSL_A_L1_osl_ankle_angle_r", "osl_ankle_torque_actuator"),
    "OpenSourceLeg_KA_L1": ("OSL_KA_L1_osl_ankle_angle_r", "osl_ankle_torque_actuator"),
}


def check_seating():
    print("1. SEATING -- ground contact at the model's own standing keyframe")
    print(f"   {'device':22} {'pelvis world z':>15} {'contacts':>9}")
    for dev in DEVICES:
        m = mj.MjModel.from_xml_string(compose_env_model(MSK, dev))
        d = mj.MjData(m)
        mj.mj_resetDataKeyframe(m, d, 0)
        mj.mj_forward(m, d)
        flag = "" if d.ncon else "   <-- opens in the air"
        print(f"   {dev:22} {float(d.body('pelvis').xpos[2]):15.5f} {d.ncon:9}{flag}")


def check_joint_defaults():
    print("\n2. JOINT DEFAULTS -- what each device's own joints compile to")
    print(f"   {'device':22} {'joint':36} {'armature':>9} {'damping':>8}")
    for dev, _, joint in POWERED + [("KFoot_L1", None, "KFoot_L1_df_ankle_angle_r")]:
        m = mj.MjModel.from_xml_string(compose_env_model(MSK, dev))
        jid = m.joint(joint).id
        dof = m.jnt_dofadr[jid]
        human = "" if joint.startswith(("ankle_", "hip_", "knee_")) else ""
        flag = "   <-- not inheriting the human default" if m.dof_armature[dof] == 0 else ""
        print(f"   {dev:22} {joint:36} {m.dof_armature[dof]:9.5f} {m.dof_damping[dof]:8.3f}{human}{flag}")


def check_limit_under_torque():
    print("\n3. JOINT LIMIT -- hold the actuator at full scale for 1 s, measure the overshoot")
    print(f"   {'device':22} {'torque':>8} {'overshoot':>10} {'% of range':>11} {'stiffness':>10}")
    for dev, actuator, joint in POWERED:
        m = mj.MjModel.from_xml_string(compose_env_model(MSK, dev))
        aid = m.actuator(actuator).id
        jid = m.joint(joint).id
        lo, hi = m.jnt_range[jid]
        best, best_torque = 0.0, 0.0
        for ctrl in (float(m.actuator_ctrlrange[aid][0]), float(m.actuator_ctrlrange[aid][1])):
            if ctrl == 0:
                continue
            d = mj.MjData(m)
            mj.mj_resetDataKeyframe(m, d, 0)
            m.opt.timestep = 1 / 1200
            d.ctrl[aid] = ctrl
            for _ in range(1200):
                mj.mj_step(m, d)
            over = max(float(d.qpos[m.jnt_qposadr[jid]]) - hi, lo - float(d.qpos[m.jnt_qposadr[jid]]), 0.0)
            if over > best:
                best, best_torque = over, abs(float(m.actuator_gainprm[aid][0]) * ctrl)
        span = float(hi - lo)
        stiff = best_torque / best if best > 1e-9 else float("inf")
        print(f"   {dev:22} {best_torque:8.1f} {best:+10.3f} {best / span * 100:10.1f}% {stiff:10.0f}")
    print("   Adding armature does not change these; at steady state inertia does not enter.")
    for armature in (0.0, 0.01, 0.1):
        dev, actuator, joint = POWERED[-2]
        m = mj.MjModel.from_xml_string(compose_env_model(MSK, dev))
        jid = m.joint(joint).id
        m.dof_armature[m.jnt_dofadr[jid]] = armature
        m.opt.timestep = 1 / 1200
        aid = m.actuator(actuator).id
        d = mj.MjData(m)
        mj.mj_resetDataKeyframe(m, d, 0)
        d.ctrl[aid] = float(m.actuator_ctrlrange[aid][1])
        for _ in range(1200):
            mj.mj_step(m, d)
        over = float(d.qpos[m.jnt_qposadr[jid]]) - float(m.jnt_range[jid][1])
        print(f"      {dev} armature {armature:5.2f} -> overshoot {over:+.3f} rad")


def distal_inertia(m, joint_name):
    jid = m.joint(joint_name).id
    root = m.jnt_bodyid[jid]
    subtree = []
    for i in range(m.nbody):
        b = i
        while b != 0:
            if b == root:
                subtree.append(i)
                break
            b = m.body_parentid[b]
    d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, 0)
    mj.mj_forward(m, d)
    axis, anchor = d.xaxis[jid], d.xanchor[jid]
    total = 0.0
    for i in subtree:
        mass = float(m.body_mass[i])
        if mass < 1e-9:
            continue
        r = d.xipos[i] - anchor
        perp = r - np.dot(r, axis) * axis
        rot = d.ximat[i].reshape(3, 3)
        inertia_world = rot @ np.diag(m.body_inertia[i]) @ rot.T
        total += float(axis @ inertia_world @ axis) + mass * float(perp @ perp)
    return total, len(subtree), sum(float(m.body_mass[i]) for i in subtree)


def check_distal_inertia():
    print("\n4. DISTAL INERTIA -- what the ankle joint has to accelerate")
    print(f"   {'device':22} {'I (kg m^2)':>11} {'mass (kg)':>10} {'bodies':>7}")
    for dev in DEVICES:
        joint, _ = ANKLE[dev]
        m = mj.MjModel.from_xml_string(compose_env_model(MSK, dev))
        inertia, n_bodies, mass = distal_inertia(m, joint)
        print(f"   {dev:22} {inertia:11.5f} {mass:10.3f} {n_bodies:7}")


if __name__ == "__main__":
    check_seating()
    check_joint_defaults()
    check_limit_under_torque()
    check_distal_inertia()
