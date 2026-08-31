"""Reproduce three measurements on the composed prosthesis models.

Findings from RL work on `myolegs22 + <prosthetic device>`, written so the model owner can
re-run them rather than take the numbers on trust. Nothing here modifies anything; it composes
each model through the shared pipeline and measures it.

    python tools/model_checks/check_prosthesis_model.py

1. SEATING. `compose._model_ground_candidates` estimates a primitive's underside as
   `centre_z - max(geom_size)`. For a capsule `size` is `(radius, half-length)` about its local z
   axis, so a horizontal capsule reaches only its radius below centre. The OpenSourceLeg foot
   contacts are capsules of radius 0.012 m and half-length 0.11 m laid flat, so the estimate is
   9.8 cm too low and the seater rests the model on a point that is not there. Both OpenSourceLeg
   compositions open with no ground contact at all. Every other device's foot contact is a mesh
   and is measured from its transformed vertices.

2. JOINT LIMIT STIFFNESS. A held actuator torque pushes the ankle a long way past its own range
   and keeps it there. Not an inertia effect: the overshoot is unchanged across two orders of
   magnitude of added armature, because at steady state inertia does not enter. It is the
   constraint stiffness, which works out near 120-130 N*m/rad against device torques of 50-168.

3. DISTAL INERTIA. Reported without a cause attached: the prosthetic ankles carry about a fifth
   of the intact ankle's rotational inertia, one distal body against four.
"""

import numpy as np

try:
    import mujoco as mj
except ImportError as exc:  # pragma: no cover
    raise SystemExit("needs the project venv: .venv/bin/python tools/model_checks/check_prosthesis_model.py") from exc

from myoassist_utils.compose import compose_env_model

MSK = "myolegs22"
DEVICES = ["Tutorial_L1", "STRIDE_L2", "KFoot_L1", "NEUankle_L1", "OpenSourceLeg_A_L1", "OpenSourceLeg_KA_L1"]
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


def check_limit_stiffness():
    print("\n2. JOINT LIMIT -- steady-state overshoot under a held torque, versus added armature")
    for dev in DEVICES:
        joint, actuator = ANKLE[dev]
        if actuator is None:
            continue
        print(f"   {dev}")
        for armature in (0.0, 0.01, 0.1):
            m = mj.MjModel.from_xml_string(compose_env_model(MSK, dev))
            d = mj.MjData(m)
            jid = m.joint(joint).id
            m.dof_armature[m.jnt_dofadr[jid]] = armature
            m.opt.timestep = 1 / 1200
            aid = m.actuator(actuator).id
            torque = float(m.actuator_gainprm[aid][0] * abs(m.actuator_ctrlrange[aid]).max())
            mj.mj_resetDataKeyframe(m, d, 0)
            d.ctrl[aid] = float(abs(m.actuator_ctrlrange[aid]).max())
            for _ in range(1200):
                mj.mj_step(m, d)
            hi = float(m.jnt_range[jid][1])
            over = float(d.qpos[m.jnt_qposadr[jid]]) - hi
            print(
                f"      armature {armature:5.2f}   {torque:6.1f} N*m held   overshoot {over:+.3f} rad "
                f"past a +-{hi:.3f} limit   implied stiffness {torque / max(over, 1e-9):6.0f} N*m/rad"
            )


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
    print("\n3. DISTAL INERTIA -- what the ankle joint has to accelerate")
    print(f"   {'device':22} {'I (kg m^2)':>11} {'mass (kg)':>10} {'bodies':>7}")
    for dev in DEVICES:
        joint, _ = ANKLE[dev]
        m = mj.MjModel.from_xml_string(compose_env_model(MSK, dev))
        inertia, n_bodies, mass = distal_inertia(m, joint)
        print(f"   {dev:22} {inertia:11.5f} {mass:10.3f} {n_bodies:7}")


if __name__ == "__main__":
    check_seating()
    check_limit_stiffness()
    check_distal_inertia()
