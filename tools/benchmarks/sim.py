"""MuJoCo simulation speed of the MSK models on one shared walking motion.

Every model replays the same 3D walking reference (`short_reference_gait.npz`, OpenSim
coordinates at 500 Hz, ~1.28 m/s). At each control step (30 Hz) the state is re-anchored to
the reference, then `frame_skip` physics steps run at the RL timestep (1/1200 s) and only
those are timed. No env or policy code runs, so the number is the simulator's cost alone.

Mapping the reference onto a model:
- hinge/slide joints the reference names are set from it (clipped to the joint range, and
  the clipping is reported). The reference uses the Gait2392 convention of negative knee
  flexion; a knee whose range starts at 0 (Rajagopal convention: myolegs, myofullbody) gets
  the sign flipped;
- joints coupled by a joint equality (knee translations, via points, spine levels) are set
  from the coupling polynomial so the anchor does not violate the constraint;
- every other joint is held at keyframe 0 (the full-body model's arms and hands, whose joint
  names the reference does not share);
- a freejoint root gets the pelvis pose rotated from the OpenSim frame (y up) into the
  model's frame, whose heading is read from the keyframe-0 foot direction;
- each anchor is seated: the root is shifted vertically so the lowest collidable geom sits
  exactly on the floor. Walking has no flight phase, so one foot is always down, and this
  absorbs leg-length differences and clipped joints without a per-model constant.

Muscle input is a constant activation on every muscle (`--activation`, several values run
one after another). At zero the legs are limp: the stance foot lifts off within the control
step and contacts are undercounted. `--validate` measures how far the replay is from
closed-loop walking; pass a range of activations to see how much the choice moves each model.

Device: the models are composed with `--device` (Tutorial_L1 by default, two ankle motors held
at zero command). `--device none` builds the bare MSK instead: the same compose pipeline
(ground, render defaults, seating) with the device step skipped. Only the device step adds
keyframes, so the bare model gets keyframe 0 of the Tutorial_L1 build (same joints and qpos
layout); both builds then replay from the same pose and differ only by the device.

`--xml SPEC` times an external MJCF instead of a composed MSK, as shipped (no device, no compose).
SPEC is a path, or `package:relative/path` to read a file from an installed package without copying
it, for example `gymnasium:envs/mujoco/assets/humanoid.xml` (Gymnasium's Humanoid model, the same
file in Humanoid-v4 and -v5) or `myosuite:simhive/myo_sim/leg/myolegs.xml` (MyoSuite's MyoLeg). The
model keeps its own MuJoCo options except the timestep, set to 1/1200 s as for the composed models;
`--native` keeps the timestep too.
  * `--protocol replay` (default) replays the walking reference; the model needs its joint names.
  * `--protocol random` runs Gymnasium's Humanoid-v5 rollout instead: uniform random controls held
    for `--frame-skip` physics steps, reset to qpos0 plus uniform noise (`--reset-noise`) when the
    root height leaves `--healthy-z` or after `--max-episode-steps`. Resets are not timed. With
    `--native --frame-skip 5` this is Humanoid-v5's physics; without `--native`, add
    `--myoassist-solver` to also use the composed models' solver (Euler, Newton, 100 iterations).

`--validate SESSION_DIR MODEL_ZIP` checks the replay against closed-loop walking, on a model
with a trained walking policy (only the 22-muscle model has one). It drives the policy through
the RL env, records state and controls at every control step, and times the recorded segments
with the same re-anchor-and-step loop on the env's own MjModel, interleaved with the reference
replay at activation 0 and 0.05. MuJoCo is deterministic, so re-anchoring to the recorded state
is a no-op and the timed loop replays the rollout exactly (the drift is reported). The recording
loop also times `model.predict` and `env.step`, which gives the RL-side overhead.

Usage (repo root, project venv):
    python tools/benchmarks/sim.py                              # all models, 1 pinned core, 20 reps
    python tools/benchmarks/sim.py --device none                # bare MSKs
    python tools/benchmarks/sim.py --concurrent 32 --no-pin     # 32 copies at once
    python tools/benchmarks/sim.py --xml myosuite:simhive/myo_sim/leg/myolegs.xml
    python tools/benchmarks/sim.py --xml gymnasium:envs/mujoco/assets/humanoid.xml --protocol random --native --frame-skip 5
    python tools/benchmarks/sim.py --validate <session_dir> <session_dir>/trained_models/<model>.zip
"""

import sys

import common

# One thread everywhere: mj_step is single-threaded, and a worker must not fan numpy out over
# cores that other pinned workers own. Set before numpy is imported.
if __name__ == "__main__":
    common.set_threads(1)

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402
from myoassist_utils.compose import compose_env_model  # noqa: E402

REF_PATH = common.REPO / "rl_train/reference_data/short_reference_gait.npz"
# RL defaults from the training configs: physics_sim_framerate 1200, control_framerate 30.
PHYSICS_HZ = 1200
CONTROL_HZ = 30

# OpenSim ground (x forward, y up, z right) -> MuJoCo world with z up: +90 deg about x.
R_OSIM_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def load_reference():
    ref_npz = np.load(REF_PATH, allow_pickle=True)
    assert ref_npz["metadata"].item()["sample_rate"] == 500
    return {k: np.asarray(v) for k, v in ref_npz["series_data"].item().items()}


def bare_model_with_device_keyframe(msk, device=common.DEVICE):
    """The bare MSK, with keyframe 0 copied from the `device` build (see the module docstring)."""
    with_device = mujoco.MjModel.from_xml_string(compose_env_model(msk, device))
    spec = mujoco.MjSpec.from_string(common.compose_bare_model(msk))
    bare = spec.compile()
    assert [bare.joint(j).name for j in range(bare.njnt)] == [with_device.joint(j).name for j in range(with_device.njnt)], (
        f"{msk}: the bare and {device} builds have different joints"
    )
    for key in list(spec.keys):
        spec.delete(key)
    spec.add_key(name="from_" + device, qpos=with_device.key_qpos[0])
    return spec.compile()


def load_model(msk, device, planar_root=False):
    if device == "none":
        assert not planar_root, "the bare MSK is built with its freejoint root"
        m = bare_model_with_device_keyframe(msk)
    else:
        m = mujoco.MjModel.from_xml_string(compose_env_model(msk, device, planar_root=planar_root))
    m.opt.timestep = 1.0 / PHYSICS_HZ
    assert m.nkey > 0, f"{msk}: no keyframe to hold the unreferenced joints at"
    return m


def heading_from_keyframe(m):
    """Rotation about z taking +x to the direction the model faces at keyframe 0."""
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_kinematics(m, d)
    foot = d.body("toes_r").xpos[:2] - d.body("calcn_r").xpos[:2]
    yaw = np.arctan2(foot[1], foot[0])
    snapped = np.round(yaw / (np.pi / 2)) * (np.pi / 2)
    assert abs(yaw - snapped) < np.deg2rad(10), f"foot heading {np.rad2deg(yaw):.1f} deg is not axis-aligned"
    return rot_z(snapped)


def apply_joint_equalities(m, q):
    """Set every joint-equality dependent from its primary: y - y0 = sum_i c_i (x - x0)^i."""
    eq_ids = [e for e in range(m.neq) if m.eq_active0[e]]
    assert all(m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT for e in eq_ids), "only joint equalities are handled"
    # Two passes so a dependent whose primary is itself a dependent still lands right.
    for _ in range(2):
        for e in eq_ids:
            a1 = m.jnt_qposadr[m.eq_obj1id[e]]
            c = m.eq_data[e][:5]
            if m.eq_obj2id[e] < 0:
                q[a1] = m.qpos0[a1] + c[0]
                continue
            a2 = m.jnt_qposadr[m.eq_obj2id[e]]
            x = q[a2] - m.qpos0[a2]
            q[a1] = m.qpos0[a1] + c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4


class ReferenceMapper:
    """Turns reference sample indices into model qpos/qvel."""

    def __init__(self, m, ref):
        self.m = m
        self.ref = ref
        self.free_root = m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        ref_names = {k[2:] for k in ref if k.startswith("q_")}
        self.driven = []  # (qpos adr, reference key, joint id, sign)
        for j in range(m.njnt):
            name = m.joint(j).name
            if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) and name in ref_names:
                sign = -1.0 if name.startswith("knee_angle") and m.jnt_range[j][0] >= 0 else 1.0
                self.driven.append((m.jnt_qposadr[j], f"q_{name}", j, sign))
        self.flipped = [m.joint(j).name for _, _, j, sign in self.driven if sign < 0]
        coupled = {m.eq_obj1id[e] for e in range(m.neq) if m.eq_active0[e]}
        driven_ids = {j for _, _, j, _ in self.driven}
        self.n_coupled = len(coupled - driven_ids)
        self.n_held = sum(
            1 for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE and j not in driven_ids and j not in coupled
        )
        self.heading = heading_from_keyframe(m)
        if not self.free_root:
            assert np.allclose(self.heading, np.eye(3)), "a planar root must face +x"
            self.ty_adr = m.jnt_qposadr[m.joint("pelvis_ty").id]
            self.tx_adr = m.jnt_qposadr[m.joint("pelvis_tx").id]
        self.tx0 = ref["q_pelvis_tx"][0]
        self.tz0 = ref["q_pelvis_tz"][0]
        self.n_clipped = 0
        self.n_set = 0
        planes = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
        assert len(planes) == 1, "seating assumes one flat plane floor"
        self.floor = planes[0]
        self.contact_geoms = [  # the model's own geoms that collide with the floor
            g
            for g in range(m.ngeom)
            if g != self.floor
            and m.geom_bodyid[g] != 0
            and ((m.geom_contype[g] & m.geom_conaffinity[self.floor]) or (m.geom_contype[self.floor] & m.geom_conaffinity[g]))
        ]
        self._d = mujoco.MjData(m)
        self._fromto = np.zeros(6)

    def qpos(self, i):
        m, ref = self.m, self.ref
        q = m.key_qpos[0].copy()
        for adr, key, j, sign in self.driven:
            v = sign * ref[key][i]
            if m.jnt_limited[j]:
                lo, hi = m.jnt_range[j]
                self.n_clipped += int(v < lo or v > hi)
                v = min(max(v, lo), hi)
            self.n_set += 1
            q[adr] = v
        if self.free_root:
            frame = self.heading @ R_OSIM_TO_ZUP
            r_osim = rot_z(ref["q_pelvis_tilt"][i]) @ rot_x(ref["q_pelvis_list"][i]) @ rot_y(ref["q_pelvis_rotation"][i])
            p_osim = np.array([ref["q_pelvis_tx"][i] - self.tx0, ref["q_pelvis_ty"][i], ref["q_pelvis_tz"][i] - self.tz0])
            q[0:3] = frame @ p_osim
            mujoco.mju_mat2Quat(q[3:7], (frame @ r_osim @ frame.T).flatten())
        else:
            q[self.tx_adr] -= self.tx0
        apply_joint_equalities(m, q)
        return q

    def seated_qpos(self, i):
        """qpos(i) shifted vertically so the lowest collidable geom touches the floor."""
        q = self.qpos(i)
        d = self._d
        d.qpos[:] = q
        mujoco.mj_kinematics(self.m, d)
        lowest = min(mujoco.mj_geomDistance(self.m, d, g, self.floor, 2.0, self._fromto) for g in self.contact_geoms)
        q[2 if self.free_root else self.ty_adr] -= lowest
        return q, lowest


def build_anchors(m, ref, t_start, duration):
    """qpos/qvel at each control step of [t_start, t_start + duration)."""
    ref_hz = 500  # metadata sample_rate of short_reference_gait.npz, asserted in load_reference
    mapper = ReferenceMapper(m, ref)
    n = int(round(duration * CONTROL_HZ))
    idx = np.round((t_start + np.arange(n) / CONTROL_HZ) * ref_hz).astype(int)
    assert idx[0] >= 1 and idx[-1] + 1 < len(ref["q_pelvis_tx"]), "anchor window runs off the reference"
    qpos = np.empty((n, m.nq))
    qvel = np.empty((n, m.nv))
    shift = np.empty(n)
    for k, i in enumerate(idx):
        qpos[k], shift[k] = mapper.seated_qpos(i)
        # Central difference on the seated poses, so the coupled joints, the freejoint angular
        # velocity and the vertical velocity all come out consistent with qpos.
        mujoco.mj_differentiatePos(m, qvel[k], 2.0 / ref_hz, mapper.seated_qpos(i - 1)[0], mapper.seated_qpos(i + 1)[0])
    n_set = mapper.n_set // 3  # seated_qpos ran three times per anchor
    info = {
        "n_driven": len(mapper.driven),
        "n_coupled": mapper.n_coupled,
        "n_held": mapper.n_held,
        "sign_flipped": mapper.flipped,
        "clipped_frac": round(mapper.n_clipped / 3 / max(n_set, 1), 4),
        # How far seating moved the root, in mm: a large spread means the reference
        # kinematics do not fit this model's segment lengths and joint limits well.
        "seat_shift_mm_mean": round(float(-shift.mean()) * 1000, 1),
        "seat_shift_mm_sd": round(float(shift.std()) * 1000, 1),
    }
    return qpos, qvel, info


def constant_inputs(m, n, activation):
    """act/ctrl arrays: every muscle at `activation`, device motors at zero command."""
    muscles = np.where(m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
    act = np.zeros((n, m.na))
    act[:, m.actuator_actadr[muscles]] = activation
    ctrl = np.zeros((n, m.nu))
    ctrl[:, muscles] = activation
    return act, ctrl, m.nu - len(muscles)


def time_anchors(m, qpos, qvel, act, ctrl, frame_skip, n_warmup):
    """Re-anchor, then time frame_skip mj_steps per anchor."""
    d = mujoco.MjData(m)
    n = len(qpos)
    wall_ns = np.empty(n, dtype=np.int64)
    ncon = np.empty(n, dtype=np.int64)
    nefc = np.empty(n, dtype=np.int64)
    niter = np.empty(n, dtype=np.int64)
    for k in range(n):
        d.qpos[:] = qpos[k]
        d.qvel[:] = qvel[k]
        d.act[:] = act[k]
        d.ctrl[:] = ctrl[k]
        t0 = time.perf_counter_ns()
        mujoco.mj_step(m, d, nstep=frame_skip)
        wall_ns[k] = time.perf_counter_ns() - t0
        ncon[k] = d.ncon
        nefc[k] = d.nefc
        niter[k] = d.solver_niter[0]
    s = slice(n_warmup, n)
    n_steps = (n - n_warmup) * frame_skip
    sim_s = n_steps * m.opt.timestep
    wall_s = wall_ns[s].sum() / 1e9
    return {
        # mj_step calls per wall second, and RL control steps (frame_skip mj_steps each) per second.
        "physics_steps_per_s": round(n_steps / wall_s, 1),
        "control_steps_per_s": round((n - n_warmup) / wall_s, 2),
        "sim_s": round(sim_s, 4),
        "wall_s": round(wall_s, 6),
        "wall_per_sim_s": round(wall_s / sim_s, 6),
        "rtf": round(sim_s / wall_s, 3),
        "us_per_step": round(wall_ns[s].sum() / 1e3 / n_steps, 3),
        "mean_ncon": round(float(ncon[s].mean()), 2),
        "frac_anchor_with_contact": round(float((ncon[s] > 0).mean()), 3),
        "mean_nefc": round(float(nefc[s].mean()), 1),
        "mean_solver_niter": round(float(niter[s].mean()), 2),
    }


def model_info(m):
    nmus = int(sum(m.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE for i in range(m.nu)))
    return {
        "muscles": nmus,
        "nu": m.nu,
        "nq": m.nq,
        "nv": m.nv,
        "nbody": m.nbody,
        "ngeom": m.ngeom,
        "ntendon": m.ntendon,
        "nwrap": m.nwrap,
        "neq": m.neq,
    }


def resolve_xml(spec):
    """A path, or `package:relative/path` inside an installed package (read in place, not copied)."""
    if ":" in spec and not pathlib.Path(spec).exists():
        import importlib.util

        package, rel = spec.split(":", 1)
        path = pathlib.Path(importlib.util.find_spec(package).origin).parent / rel
    else:
        path = pathlib.Path(spec)
    assert path.is_file(), f"no model file at {path}"
    return path


def xml_label(spec):
    return (spec.split(":", 1)[0] + ":" if ":" in spec and not pathlib.Path(spec).exists() else "") + pathlib.Path(spec).stem


def load_external(spec, native, myoassist_solver):
    m = mujoco.MjModel.from_xml_path(str(resolve_xml(spec)))
    if not native:
        m.opt.timestep = 1.0 / PHYSICS_HZ
    if myoassist_solver:  # the composed models' options
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        m.opt.iterations = 100
        m.opt.ls_iterations = 50
        m.opt.tolerance = 1e-8
    return m


def time_random(m, frame_skip, n_control, n_warmup, healthy_z, reset_noise, max_episode_steps, seed):
    """Gymnasium Humanoid-v5's rollout with a random policy; only the mj_step calls are timed."""
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    assert m.actuator_ctrllimited.all(), "random controls are drawn from the control range"

    def reset():
        d.qpos[:] = m.qpos0 + rng.uniform(-reset_noise, reset_noise, m.nq)
        d.qvel[:] = rng.uniform(-reset_noise, reset_noise, m.nv)
        d.act[:] = 0
        mujoco.mj_forward(m, d)

    reset()
    wall_ns = np.empty(n_control, dtype=np.int64)
    ncon = np.empty(n_control, dtype=np.int64)
    nefc = np.empty(n_control, dtype=np.int64)
    niter = np.empty(n_control, dtype=np.int64)
    resets, episode = 0, 0
    for k in range(n_control):
        d.ctrl[:] = rng.uniform(lo, hi)
        t0 = time.perf_counter_ns()
        mujoco.mj_step(m, d, nstep=frame_skip)
        wall_ns[k] = time.perf_counter_ns() - t0
        ncon[k], nefc[k], niter[k] = d.ncon, d.nefc, d.solver_niter[0]
        episode += 1
        if not healthy_z[0] < d.qpos[2] < healthy_z[1] or episode >= max_episode_steps:
            reset()
            resets += int(k >= n_warmup)
            episode = 0
    s = slice(n_warmup, n_control)
    n_steps = (n_control - n_warmup) * frame_skip
    sim_s = n_steps * m.opt.timestep
    wall_s = wall_ns[s].sum() / 1e9
    return {
        "physics_steps_per_s": round(n_steps / wall_s, 1),
        "control_steps_per_s": round((n_control - n_warmup) / wall_s, 2),
        "sim_s": round(sim_s, 4),
        "wall_s": round(wall_s, 6),
        "wall_per_sim_s": round(wall_s / sim_s, 6),
        "rtf": round(sim_s / wall_s, 3),
        "us_per_step": round(wall_ns[s].sum() / 1e3 / n_steps, 3),
        "mean_ncon": round(float(ncon[s].mean()), 2),
        "frac_anchor_with_contact": round(float((ncon[s] > 0).mean()), 3),
        "mean_nefc": round(float(nefc[s].mean()), 1),
        "mean_solver_niter": round(float(niter[s].mean()), 2),
        "resets": resets,
        "mean_episode_steps": round((n_control - n_warmup) / max(resets, 1), 1),
    }


def solver_info(m, frame_skip):
    return {
        "timestep": m.opt.timestep,
        "frame_skip": frame_skip,
        "integrator": mujoco.mjtIntegrator(m.opt.integrator).name,
        "solver": mujoco.mjtSolver(m.opt.solver).name,
        "iterations": int(m.opt.iterations),
    }


# --- speed ------------------------------------------------------------------------------------


def worker(args):
    common.pin(args.core)
    if args.xml:
        m = load_external(args.xml, args.native, args.myoassist_solver)
    else:
        m = load_model(args.msk, args.device, args.planar_root)
    n_motors = m.nu - int(sum(m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE))
    if args.protocol == "replay":
        frame_skip = PHYSICS_HZ // CONTROL_HZ
        assert abs(m.opt.timestep - 1.0 / PHYSICS_HZ) < 1e-12, "the replay steps at 1/1200 s"
        qpos, qvel, info = build_anchors(m, load_reference(), args.t_start, args.warmup + args.duration)
        act, ctrl, _ = constant_inputs(m, len(qpos), args.activation)
        assert args.xml or args.device != "none" or n_motors == 0
        res = time_anchors(m, qpos, qvel, act, ctrl, frame_skip, int(round(args.warmup * CONTROL_HZ)))
    else:
        frame_skip = args.frame_skip
        control_dt = m.opt.timestep * frame_skip
        n_control = int(round((args.warmup + args.duration) / control_dt))
        n_warmup = int(round(args.warmup / control_dt))
        res = time_random(
            m, frame_skip, n_control, n_warmup, args.healthy_z, args.reset_noise, args.max_episode_steps, args.rep
        )
        info = {}
    common.emit(
        {
            "msk": xml_label(args.xml) if args.xml else args.msk,
            "xml": str(resolve_xml(args.xml)) if args.xml else None,
            "device": None if args.xml else args.device,
            "device_motors": n_motors,
            "protocol": args.protocol,
            "activation": args.activation,
            "planar_root": args.planar_root,
            "rep": args.rep,
            "core": args.core,
            **model_info(m),
            **solver_info(m, frame_skip),
            **info,
            **res,
        }
    )


def worker_cmd(args, model, activation, rep, core):
    cmd = [sys.executable, __file__, "--worker", "--xml" if args.xml else "--msk", model, "--activation", str(activation)]
    cmd += ["--rep", str(rep), "--device", args.device, "--duration", str(args.duration), "--warmup", str(args.warmup)]
    cmd += ["--t-start", str(args.t_start), "--protocol", args.protocol, "--frame-skip", str(args.frame_skip)]
    cmd += ["--healthy-z", *map(str, args.healthy_z), "--reset-noise", str(args.reset_noise)]
    cmd += ["--max-episode-steps", str(args.max_episode_steps)]
    cmd += ["--native"] if args.native else []
    cmd += ["--myoassist-solver"] if args.myoassist_solver else []
    if core is not None:
        cmd += ["--core", str(core)]
    if args.planar_root:
        cmd += ["--planar-root"]
    return cmd


def speed(args):
    # Pin to physical cores. psutil numbers logical CPUs with SMT siblings adjacent on
    # Windows and Linux, so every second logical CPU is a distinct physical core.
    smt = psutil.cpu_count(logical=True) // psutil.cpu_count(logical=False)
    cores = [args.first_core + smt * i for i in range(args.concurrent)] if args.pin else [None] * args.concurrent
    assert not args.pin or cores[-1] < psutil.cpu_count(logical=True), "not enough cores for --concurrent"
    runs = []
    models = args.xml or args.msk
    cases = [(model, a) for model in models for a in args.activation]
    for rep in range(args.reps):
        order = cases[rep % len(cases) :] + cases[: rep % len(cases)]  # rotate to spread drift
        for msk, a in order:
            load_before = psutil.cpu_percent(interval=1.0)
            procs = [common.spawn_worker(worker_cmd(args, msk, a, rep, c)) for c in cores]
            results = [common.collect(p) for p in procs]
            for r in results:
                r["concurrent"] = args.concurrent
                r["system_cpu_pct_before"] = load_before
            runs.extend(results)
            print(
                f"rep {rep:>2} {msk:<12} act={a:.2f} steps/s={np.mean([r['physics_steps_per_s'] for r in results]):8.0f} "
                f"RTF={np.mean([r['rtf'] for r in results]):7.2f} ncon={results[0]['mean_ncon']:5.2f} "
                f"load_before={load_before:4.1f}%",
                flush=True,
            )

    summary = {}
    for model, a in cases:
        msk = xml_label(model) if args.xml else model
        rs = [r for r in runs if r["msk"] == msk and r["activation"] == a]
        # Across concurrent copies, a rep's value is the mean over copies; variability is across reps.
        per_rep = {
            k: [float(np.mean([r[k] for r in rs if r["rep"] == rep])) for rep in range(args.reps)]
            for k in ("physics_steps_per_s", "control_steps_per_s", "us_per_step", "wall_per_sim_s", "rtf")
        }
        static = ("muscles", "nu", "nv", "ngeom", "nwrap", "timestep", "frame_skip", "integrator", "solver", "iterations")
        static += ("n_driven", "n_coupled", "n_held", "sign_flipped", "clipped_frac") if args.protocol == "replay" else ()
        observed = ("mean_ncon", "mean_nefc", "mean_solver_niter")
        observed += ("seat_shift_mm_mean",) if args.protocol == "replay" else ("mean_episode_steps",)
        summary[f"{msk}@act{a:g}"] = {
            "msk": msk,
            "activation": a,
            **{k: rs[0][k] for k in static},
            **{k: round(float(np.mean([r[k] for r in rs])), 2) for k in observed},
            **{k: common.stats(v) for k, v in per_rep.items()},
            "aggregate_rtf": round(float(np.mean(per_rep["rtf"])) * args.concurrent, 2),
            "aggregate_control_steps_per_s": round(float(np.mean(per_rep["control_steps_per_s"])) * args.concurrent, 1),
        }
    settings = {
        "device": None if args.xml else args.device,
        "xml": args.xml,
        "protocol": args.protocol,
        "native": args.native,
        "myoassist_solver": args.myoassist_solver,
        "frame_skip": args.frame_skip if args.protocol == "random" else PHYSICS_HZ // CONTROL_HZ,
        "healthy_z": args.healthy_z if args.protocol == "random" else None,
        "reset_noise": args.reset_noise if args.protocol == "random" else None,
        "max_episode_steps": args.max_episode_steps if args.protocol == "random" else None,
        "physics_hz": PHYSICS_HZ,
        "control_hz": CONTROL_HZ,
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "t_start_s": args.t_start,
        "reps": args.reps,
        "concurrent": args.concurrent,
        "pinned_cores": cores,
        "activation": args.activation,
        "planar_root": args.planar_root,
        "reference": REF_PATH.name,
    }
    common.write_results("sim", settings, summary, runs, args.out)
    for key, s in summary.items():
        print(
            f"  {key:<20} {s['muscles']:>3} muscles  physics steps/s {s['physics_steps_per_s']['mean']:9.0f} "
            f"± {s['physics_steps_per_s']['sd']}  RTF {s['rtf']['mean']:.2f} ± {s['rtf']['sd']}"
        )


# --- validate ---------------------------------------------------------------------------------


def record_closed_loop(config, model_zip, n_steps, velocity):
    import torch
    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.envs.myoassist_leg_base import MyoAssistLegBase

    torch.set_num_threads(1)
    config.env_params.num_envs = 1  # a single env comes back unwrapped, not as a SubprocVecEnv
    # The out-of-trajectory check ends upright walking episodes on a drift from the reference
    # phase. It changes neither the dynamics nor the policy input, so turn it off; the fall
    # check (pelvis below safe_height) still terminates and is asserted below.
    config.env_params.out_of_trajectory_threshold = float("inf")
    # reset() draws the start index from an unseeded RNG; start at 0 so the rollout repeats.
    config.env_params.flag_random_ref_index = False
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=True)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env, trained_model_path=str(model_zip))
    base = env.unwrapped
    base.set_target_velocity_mode_manually(MyoAssistLegBase.VelocityMode.UNIFORM, 0, velocity, velocity, velocity)
    m = base.sim.model.ptr
    d = base.sim.data.ptr
    assert isinstance(m, mujoco.MjModel) and isinstance(d, mujoco.MjData)

    obs, _ = env.reset()
    rec = {k: [] for k in ("qpos", "qvel", "act", "ctrl")}
    predict_ns, step_ns = [], []
    for k in range(n_steps):
        rec["qpos"].append(d.qpos.copy())
        rec["qvel"].append(d.qvel.copy())
        rec["act"].append(d.act.copy())
        t0 = time.perf_counter_ns()
        action, _ = model.predict(obs, deterministic=True)
        t1 = time.perf_counter_ns()
        obs, _, done, truncated, _ = env.step(action)
        t2 = time.perf_counter_ns()
        rec["ctrl"].append(d.ctrl.copy())
        predict_ns.append(t1 - t0)
        step_ns.append(t2 - t1)
        assert not done, f"policy fell at control step {k}; pick a checkpoint that walks {n_steps} steps"
        assert not truncated, f"episode truncated at control step {k}; lower --duration"
    rec = {k: np.array(v) for k, v in rec.items()}
    tx_adr = m.jnt_qposadr[m.joint("pelvis_tx").id]
    rl = {
        "predict_ms": float(np.mean(predict_ns[30:])) / 1e6,
        "env_step_ms": float(np.mean(step_ns[30:])) / 1e6,
        "mean_speed_m_s": round(float(d.qpos[tx_adr] - rec["qpos"][0][tx_adr]) / (n_steps * base.dt), 3),
    }
    return m, base.frame_skip, rec, rl


def replay_drift(m, rec, frame_skip):
    """Max |qpos| error after stepping each recorded segment, against the next recorded state."""
    d = mujoco.MjData(m)
    worst = 0.0
    for k in range(len(rec["qpos"]) - 1):
        d.qpos[:], d.qvel[:], d.act[:], d.ctrl[:] = rec["qpos"][k], rec["qvel"][k], rec["act"][k], rec["ctrl"][k]
        mujoco.mj_step(m, d, nstep=frame_skip)
        worst = max(worst, float(np.abs(d.qpos - rec["qpos"][k + 1]).max()))
    return worst


def validate(args):
    # Pin before torch starts its threads: on Windows with more than 64 logical CPUs those can
    # land in another processor group, and the affinity call then fails (WinError 87).
    common.pin(args.core)
    from rl_train.envs.environment_handler import EnvironmentHandler

    session_dir, model_zip = map(pathlib.Path, args.validate)
    cfg_path = session_dir / "session_config.json"
    env_id = json.loads(cfg_path.read_text())["env_params"]["env_id"]
    config = EnvironmentHandler.get_session_config_from_path(
        str(cfg_path), EnvironmentHandler.get_config_type_from_session_id(env_id)
    )
    assert config.env_params.physics_sim_framerate == PHYSICS_HZ
    assert config.env_params.control_framerate == CONTROL_HZ

    n_steps = int(round((args.warmup + args.duration) * CONTROL_HZ))
    n_warmup = int(round(args.warmup * CONTROL_HZ))
    m, frame_skip, rec, rl = record_closed_loop(config, model_zip, n_steps, args.velocity)
    drift = replay_drift(m, rec, frame_skip)

    ref = load_reference()
    ref_qpos, ref_qvel, map_info = build_anchors(m, ref, args.t_start, args.warmup + args.duration)
    composed = load_model(config.env_params.msk_key, config.env_params.device_key)
    comp_qpos, comp_qvel, _ = build_anchors(composed, ref, args.t_start, args.warmup + args.duration)
    sources = {"closed_loop_policy": (m, rec["qpos"], rec["qvel"], rec["act"], rec["ctrl"])}
    for a in args.activation:
        sources[f"replay_act{a:g}_env_model"] = (m, ref_qpos, ref_qvel, *constant_inputs(m, n_steps, a)[:2])
        # The composed model without the env's setup-time edits, as `speed` builds it.
        sources[f"replay_act{a:g}_composed_model"] = (
            composed,
            comp_qpos,
            comp_qvel,
            *constant_inputs(composed, n_steps, a)[:2],
        )
    runs = {name: [] for name in sources}
    for rep in range(args.reps):
        for name, (mm, q, v, a, c) in sources.items():  # interleaved so drift hits all sources alike
            runs[name].append(time_anchors(mm, q, v, a, c, frame_skip, n_warmup))
        print(f"rep {rep:>2} " + "  ".join(f"{n}={runs[n][-1]['us_per_step']:.2f}us" for n in sources), flush=True)

    keys = ("us_per_step", "rtf", "mean_ncon", "frac_anchor_with_contact", "mean_nefc", "mean_solver_niter")
    summary = {name: {k: common.stats([r[k] for r in rs]) for k in keys} for name, rs in runs.items()}
    physics_ms = summary["closed_loop_policy"]["us_per_step"]["mean"] * frame_skip / 1e3
    settings = {
        "session_dir": str(session_dir),
        "model_zip": str(model_zip),
        "msk": config.env_params.msk_key,
        "device": config.env_params.device_key,
        "physics_hz": PHYSICS_HZ,
        "control_hz": CONTROL_HZ,
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "reps": args.reps,
        "activation": args.activation,
        "velocity": args.velocity,
        "core": args.core,
    }
    extra = {
        "closed_loop_replay_max_qpos_drift": drift,
        "reference_mapping": map_info,
        # One control step in the RL loop, single core: policy inference + env.step, where
        # env.step = physics (frame_skip mj_steps) + observation/reward/bookkeeping.
        "rl_step_breakdown_ms": {
            **{k: round(v, 4) for k, v in rl.items()},
            "physics_ms": round(physics_ms, 4),
            "env_overhead_ms": round(rl["env_step_ms"] - physics_ms, 4),
        },
    }
    common.write_results("validate", settings, summary, runs, args.out, **extra)
    print(f"closed-loop replay drift (max |qpos| error): {drift:.2e}")
    for name, s in summary.items():
        print(
            f"  {name:<32} us/step {s['us_per_step']['mean']:8.2f} ± {s['us_per_step']['sd']}  ncon {s['mean_ncon']['mean']:5.2f}"
        )
    print("  RL step breakdown (ms):", extra["rl_step_breakdown_ms"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--msk", nargs="+", default=common.MSKS)
    ap.add_argument("--device", default=common.DEVICE, help='"none" builds the bare MSK')
    ap.add_argument(
        "--activation",
        type=float,
        nargs="+",
        default=None,
        help="constant muscle activation(s); default 0.05 (speed), 0 and 0.05 (--validate)",
    )
    ap.add_argument("--reps", type=int, default=None, help="default 20 (speed), 10 (--validate)")
    ap.add_argument(
        "--planar-root",
        action="store_true",
        help="compose 3D models with the six named pelvis DOFs instead of the freejoint (the RL env's build)",
    )
    ap.add_argument("--duration", type=float, default=10.0, help="timed simulated seconds per run")
    ap.add_argument("--warmup", type=float, default=1.0, help="untimed simulated seconds before timing")
    ap.add_argument("--t-start", type=float, default=1.0, help="start time in the reference [s]")
    ap.add_argument("--concurrent", type=int, default=1, help="identical workers run at once, one per core")
    ap.add_argument("--first-core", type=int, default=2, help="first logical CPU to pin to (skip CPU 0)")
    ap.add_argument("--no-pin", dest="pin", action="store_false", help="leave workers unpinned (required on macOS)")
    ap.add_argument("--xml", nargs="+", default=None, help="external MJCF files (path or package:relative/path)")
    ap.add_argument("--protocol", choices=["replay", "random"], default="replay")
    ap.add_argument("--native", action="store_true", help="--xml: keep the model's own timestep")
    ap.add_argument("--myoassist-solver", action="store_true", help="--xml: Euler, Newton, 100 iterations")
    ap.add_argument("--frame-skip", type=int, default=40, help="--protocol random: physics steps per action")
    ap.add_argument("--healthy-z", type=float, nargs=2, default=[1.0, 2.0], help="--protocol random: root height range")
    ap.add_argument("--reset-noise", type=float, default=0.01, help="--protocol random: uniform reset noise")
    ap.add_argument("--max-episode-steps", type=int, default=1000, help="--protocol random: episode truncation")
    ap.add_argument("--validate", nargs=2, metavar=("SESSION_DIR", "MODEL_ZIP"), help="check the replay against a policy")
    ap.add_argument("--velocity", type=float, default=1.25, help="--validate: target speed for the policy [m/s]")
    ap.add_argument("--out", default=None)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--rep", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--core", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        assert len(args.activation) == 1 and (len(args.xml) == 1 if args.xml else len(args.msk) == 1)
        args.activation = args.activation[0]
        args.msk, args.xml = (None, args.xml[0]) if args.xml else (args.msk[0], None)
        worker(args)
    elif args.validate:
        args.activation = args.activation or [0.0, 0.05]
        args.reps = args.reps or 10
        args.core = args.first_core if args.pin else None
        validate(args)
    else:
        args.activation = args.activation or [0.05]
        args.reps = args.reps or 20
        speed(args)


if __name__ == "__main__":
    main()
