"""Hard range-of-motion (ROM) clamp for named joints, applied each control step.

ROM here is a study *constraint*, not an optimized parameter: the optimization
config sets a joint range and this holds the joint inside it with a **hard stop**
-- the qpos is clamped into ``[lo, hi]`` and the qvel is zeroed at the limit.
MuJoCo's soft ``range`` limit alone overshoots the intended ROM under load, so
the clamp is applied on top of it (see the MYOASSIST_FEATURES decision log).

Missing joints are skipped (a prosthetic side may lack ``ankle_angle_r``), so the
same call is safe across intact and amputee models.
"""


def clamp_joint_rom(sim, joint_names, lo, hi):
    """Clamp each named joint's qpos into ``[lo, hi]``; zero its qvel at a limit.

    Returns the number of joints actually clamped this step (0 if none reached a
    limit), mostly for tests/telemetry.
    """
    data = sim.data
    hit = 0
    for name in joint_names:
        try:
            joint = data.joint(name)
        except KeyError:
            continue  # joint amputated away / not in this model
        q = float(joint.qpos[0])
        if q < lo:
            joint.qpos[0] = lo
            joint.qvel[0] = 0.0
            hit += 1
        elif q > hi:
            joint.qpos[0] = hi
            joint.qvel[0] = 0.0
            hit += 1
    return hit
