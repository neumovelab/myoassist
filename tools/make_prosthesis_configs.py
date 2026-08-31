"""Generate one imitation-training config per prosthetic device.

The exo sweep (`tools/make_device_sweep_configs.py`) covers the bilateral drop-in devices:
an intact 22-muscle model, two device actuators, and a left/right mirror pair. The
prostheses are the other class, and its own generator, because every assumption that sweep
rests on is false here:

  * **The human model is asymmetric.** The device performs the amputation, so the composed
    model is missing muscles, joints and sensors on the prosthetic side. `myolegs22 +
    OpenSourceLeg_A_L1` has 18 muscles, not 22, and no `ankle_angle_r` or `mtp_angle_r`.
    Every absolute observation index in an exo config is therefore wrong here, and the
    counts differ per device, so the layout has to be read off the composed model.
  * **There is no mirror map.** `mirror.py:_swap_name` requires every actuator to have a
    partner on the other side; on an amputee model it raises, correctly. `mirror_coef` is
    pinned to 0 and the per-side shared exo network (`exo_actor_r`/`exo_actor_l`) is not
    used -- a single `exo_actor` drives the one-sided device.
  * **The reference is a healthy walker.** `short_reference_gait.npz` has `q_ankle_angle_r`,
    but the model it would be written to has no such joint, and more fundamentally the whole
    amputated side cannot reach a healthy, near-symmetric trajectory: without an ankle
    push-off the residual limb's knee does not produce the healthy swing-phase flexion. The
    qpos/qvel imitation weights therefore keep only the pelvis and the intact leg. Because
    `MyoAssistLegImitation.step` reads that same dict for its `out_of_trajectory_threshold`
    check, this also stops the affected side from terminating episodes; the threshold itself
    is relaxed from the intact configs' 0.2 because the intact leg compensates and does not
    track a healthy walker as tightly either. `reference_data_keys` still *places* the
    residual limb from the reference at reset -- dropping a joint from the reward is not the
    same as starting it from an arbitrary pose.

    This is measured, not assumed. On the first 30M runs (imitation on the affected side,
    threshold 0.2) `knee_angle_r` was both the largest tracking error and the most frequent
    cause of termination, and both prosthesis runs stopped improving at ~15M steps -- episode
    length 4.1 -> 4.7 for `OpenSourceLeg_A_L1` -- while the intact `Tutorial_L1` control,
    sharing all of this infrastructure, kept improving to the end (4.25 -> 7.51) and walked
    39 m in evaluation against the prostheses' 1 m.

Devices, all right-sided (`python -m assist_sim list` for the registry):

    OpenSourceLeg_A_L1    transtibial, powered ankle          18 muscle + 1 motor
    NEUankle_L1           transtibial, powered ankle          18 muscle + 1 motor
    OpenSourceLeg_KA_L1   transfemoral, powered knee+ankle    15 muscle + 2 motor
    KFoot_L1              transtibial, passive spring foot    18 muscle + 0 motor

`KFoot_L1` has no motor, so there is nothing for an exo sub-policy to drive and
`EnvironmentHandler._validate_action_layout` rejects the exo policy for it. Its config uses
`myoAssistLegImitation-v0` and the muscle-only `HumanActorCriticPolicy`, which reads the whole
observation and writes the whole action, so it carries no `net_indexing_info`.

Nothing about the layout is tabulated here: the muscle count, the actuator names, the
prosthetic joints that stand in for the missing human ones, and the joint-limit sensors that
survive the amputation are all read from the composed model, so a device or model change
cannot leave a stale index behind.
"""

import argparse
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
CFG_DIR = REPO / "rl_train" / "train" / "train_configs"
TEMPLATE = CFG_DIR / "imitation_tutorial_22_separated_net_partial_obs.json"
OUT_DIR = CFG_DIR / "prosthesis"

MSK = "myolegs22"

# The prosthetic devices in the assist_sim registry. Value is the amputation, used only in the
# generated config's comments and the console summary; everything structural is read from the
# composed model. Anatomics_L1 is not here: it is a passive *orthotic* insole on an intact
# model, so it belongs to the exo sweep's class, not this one.
PROSTHESES = {
    "OpenSourceLeg_A_L1": "transtibial (below knee), right",
    "NEUankle_L1": "transtibial (below knee), right",
    "OpenSourceLeg_KA_L1": "transfemoral (above knee), right",
    "KFoot_L1": "transtibial (below knee), right",
}


def _model_facts(device: str) -> dict:
    """Actuator, joint and sensor names of the composed `MSK + device` model.

    Composed rather than assumed, because the amputation is performed by the device: which
    muscles, joints and sensors survive is a property of the pair, and the configs address
    observations by absolute index.
    """
    import mujoco

    from myoassist_utils.compose import compose_env_model

    model = mujoco.MjModel.from_xml_string(compose_env_model(MSK, device))
    muscle = [i for i in range(model.nu) if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE]
    motor = [i for i in range(model.nu) if i not in set(muscle)]
    return {
        "nu": model.nu,
        "n_muscle": len(muscle),
        "n_motor": len(motor),
        "actuators": [model.actuator(i).name for i in range(model.nu)],
        "motor_names": [model.actuator(i).name for i in motor],
        "joints": {model.joint(i).name for i in range(model.njnt)},
        "sensors": {model.sensor(i).name for i in range(model.nsensor)},
    }


def _substitute(keys: list[str], joints: set[str]) -> tuple[list[str], list[str]]:
    """Replace each absent human joint key with the device joints that stand in for it.

    The amputation removes `ankle_angle_r` (and `knee_angle_r`, transfemorally) and the device
    reintroduces the DOF under its own name -- `OSL_A_L1_osl_ankle_angle_r`, or the two spring
    joints `KFoot_L1_df_ankle_angle_r` / `KFoot_L1_pf_ankle_angle_r`. Matched by suffix rather
    than tabulated, so a new device that follows the same naming needs no edit here; a device
    that does not will fail the assertion rather than silently drop the joint from observation.

    Returns the substituted key list and, separately, the device joints that were substituted
    in -- those are the prosthetic side, which is what the exo sub-policy observes.
    """
    out: list[str] = []
    prosthetic: list[str] = []
    for key in keys:
        if key in joints:
            out.append(key)
            continue
        stand_ins = sorted(j for j in joints if j.endswith(key))
        assert stand_ins, (
            f"joint {key!r} is absent from the composed model and no device joint ends with that "
            f"name, so it cannot be observed. Model joints: {sorted(joints)}"
        )
        out.extend(stand_ins)
        prosthetic.extend(stand_ins)
    return out, prosthetic


def _keep_present(keys: list[str], joints: set[str]) -> list[str]:
    """The subset of `keys` the composed model actually has, order preserved.

    Used for the reference and imitation keys, where an absent joint must be *dropped* rather
    than substituted: the reference is a healthy walker and has nothing to say about a
    prosthetic joint. Substituting would write a human ankle trajectory onto a device DOF whose
    range is narrower (OSL's ankle is +-0.52 rad against the human's -1.13..0.35) and, on the
    transfemoral knee, whose sign convention is reversed ([0, 2.09] against [-2.53, 0]).
    """
    return [k for k in keys if k in joints]


def _indices(keys: list[str], names: list[str], offset: int) -> list[int]:
    """Global observation indices of `names` within a block starting at `offset`."""
    index = {k: i + offset for i, k in enumerate(keys)}
    return [index[n] for n in names]


def _standing_height(device: str) -> float:
    """The `pelvis_ty` qpos at which the composed model's feet reach the ground.

    Same rule as `MyoAssistLegImitation._standing_pelvis_height`: the keyframe when it is a
    standing pose, otherwise measured by lowering the pelvis until something touches. The two
    OpenSourceLeg compositions need the second branch -- they hang 8.6 cm above the floor at
    their own keyframe.
    """
    import mujoco

    from myoassist_utils.compose import compose_env_model

    model = mujoco.MjModel.from_xml_string(compose_env_model(MSK, device))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ty_adr = model.jnt_qposadr[model.joint("pelvis_ty").id]
    keyframe = float(data.qpos[ty_adr])
    mujoco.mj_forward(model, data)
    if data.ncon > 0:
        return keyframe
    low, high = keyframe - 0.4, keyframe
    for _ in range(40):
        mid = 0.5 * (low + high)
        data.qpos[ty_adr] = mid
        mujoco.mj_forward(model, data)
        low, high = (mid, high) if data.ncon > 0 else (low, mid)
    return 0.5 * (low + high)


def _amputated_side(prosthetic_joints: list[str]) -> str:
    """The body side the device amputates, read off the joints it substituted in.

    Every shipped prosthesis is right-sided, but reading it from the composed model rather than
    assuming keeps a future left-sided device from silently getting the wrong side dropped.
    """
    sides = {j[-1] for j in prosthetic_joints}
    assert sides <= {"l", "r"} and len(sides) == 1, (
        f"cannot tell which side the amputation is on from {prosthetic_joints}; expected every "
        f"substituted joint to end in the same '_l'/'_r' suffix"
    )
    return sides.pop()


def _residual_and_replaced(keys: list[str], joints: set[str], side: str) -> tuple[list[str], list[str]]:
    """Split the amputated side's human joint keys into what survives and what the device took.

    A transtibial device replaces only the ankle: `knee_angle_r` and `hip_flexion_r` are still
    the person's own joints, driven by their own muscles. A transfemoral device takes the knee
    too. The distinction matters because the two want opposite treatment -- a joint the person
    still has should be shaped by the reward (without one, nothing in the reward cares about it
    and the knee simply folds), while a joint the device replaced has no healthy counterpart to
    be shaped towards at all.
    """
    residual = [k for k in keys if k.endswith(f"_{side}") and k in joints]
    replaced = [k for k in keys if k.endswith(f"_{side}") and k not in joints]
    return residual, replaced


def _drop_amputated_side(weights: dict, side: str) -> dict:
    """Remove the amputated limb's joints from an imitation weight dict.

    The reference is a healthy, near-symmetric walker, and on the amputated side that target is
    not reachable: without an ankle push-off the residual limb's knee cannot produce the healthy
    swing-phase flexion, so the term asks for a trajectory no policy can deliver. Measured on the
    30M `OpenSourceLeg_A_L1` run, `knee_angle_r` was the single largest tracking error and the
    most frequent cause of episode termination -- the same dict drives the reward and the
    `out_of_trajectory_threshold` check in `MyoAssistLegImitation.step`.

    Both prosthesis runs flattened out after ~15M steps (episode length 4.1 -> 4.7 for
    `OpenSourceLeg_A_L1`) while the intact `Tutorial_L1` control, sharing all of this
    infrastructure, kept improving to the end (4.25 -> 7.51). Dropping the affected side leaves
    the pelvis and the intact leg -- the parts of an amputee gait a healthy reference can still
    speak to -- and lets the residual limb be shaped by the forward, foot-force and effort terms
    instead.
    """
    return {k: v for k, v in weights.items() if not k.endswith(f"_{side}")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="*", default=None, help="Subset of device keys (default: all four).")
    ap.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=OUT_DIR,
        help="Where to write the configs. Defaults to the shipped prosthesis directory; point it at "
        "a temporary directory when generating throwaway variants.",
    )
    ap.add_argument(
        "--muscle-activation-penalty",
        type=float,
        default=None,
        help="Override reward_keys_and_weights.muscle_activation_penalty (template value: 0.1). "
        "Filenames gain an _actpen<value> suffix.",
    )
    ap.add_argument(
        "--device-ctrl-scale",
        type=float,
        default=1.0,
        help="Fraction of its own ctrlrange the policy may command on the device actuators. "
        "1.0, the default, is the device's own specification and is what should normally be used: "
        "the ctrlrange and gain are the manufacturer's, not ours to reduce. Kept as an escape "
        "hatch only. A run at 0.15 showed why a cap is the wrong instrument -- it is a fraction "
        "of each device's own range, so the same number meant 25 N*m on OpenSourceLeg_A_L1 "
        "(ctrlrange 2.88) and 7.5 N*m on NEUankle_L1 (ctrlrange 1.0), and at 7.5 N*m the ankle is "
        "worth so little that the policy left it alone: mean commanded torque 1.08 N*m, never "
        "saturating, net mechanical power -0.74 W. Price device effort through "
        "--device-activation-penalty instead. Filenames gain a _cap<value> suffix when set.",
    )
    ap.add_argument(
        "--device-activation-penalty",
        type=float,
        default=1.0,
        help="Price of prosthesis effort, same units as --muscle-activation-penalty (both are dt "
        "times a mean dimensionless effort; device ctrl is normalised by its own ctrlrange). "
        "Filenames gain a _devpen<value> suffix.",
    )
    ap.add_argument("--human-net", type=int, default=None, help="Width of both human_actor hidden layers (template: 64).")
    ap.add_argument("--device-net", type=int, default=None, help="Width of both prosthesis-actor hidden layers (template: 8).")
    ap.add_argument("--total-timesteps", type=float, default=None, help="Override total_timesteps (template: 3e7).")
    ap.add_argument(
        "--out-of-trajectory-threshold",
        type=float,
        default=0.4,
        help="Radians of imitation tracking error, on any joint the imitation dict still names, "
        "that ends the episode (`MyoAssistLegImitation.step`). The intact configs use 0.2; the "
        "default here is 0.4 because an amputee's intact leg compensates for the missing "
        "push-off and does not follow a healthy walker as closely. Filenames gain an _oot<value> "
        "suffix when this differs from the default.",
    )
    ap.add_argument(
        "--forward-reward",
        type=float,
        default=20.0,
        help="Weight on the forward-velocity term (template: 1.0). On an intact model tracking "
        "the reference is walking, so an imitation-dominant reward produces gait; on an amputee "
        "the imitation term covers only the pelvis and the intact leg, and its optimum is "
        "reachable while the prosthetic leg drags. Measured on the 30M runs: imitation 77.4%% of "
        "the weighted total against forward's 5.7%%. Filenames gain a _fwd<value> suffix when "
        "this differs from the default.",
    )
    ap.add_argument(
        "--residual-imitation-weight",
        type=float,
        default=0.2,
        help="qpos/qvel imitation weight for the amputated side's *surviving* joints -- the "
        "residual limb's own hip, and its knee for a transtibial device. A weak posture term: "
        "without one nothing in the reward refers to those joints and the knee folds. They are "
        "excluded from the out-of-trajectory check regardless. 0 removes the term entirely.",
    )
    ap.add_argument(
        "--fall-margin",
        type=float,
        default=0.21,
        help="Metres the pelvis may drop below the composition's standing height before "
        "safe_height ends the episode. The intact configs work out to ~0.21 (0.915 standing "
        "against safe_height 0.7); this keeps that margin whatever height the device stands at.",
    )
    ap.add_argument(
        "--curriculum-start-velocity",
        type=float,
        default=0.3,
        help="Target velocity the run starts at, ramping to the config's min/max over "
        "--curriculum-fraction of training. 0 disables the curriculum. Default 0.3 because that "
        "is the speed at which the amputee models actually produced sustained stepping. Setting "
        "it also turns on scale_reference_playback, without which a slow target contradicts the "
        "imitation term. Filenames gain a _curr<value> suffix when this differs from the default.",
    )
    ap.add_argument(
        "--curriculum-fraction",
        type=float,
        default=0.5,
        help="Fraction of total_timesteps over which the target velocity ramps to full; held there afterwards.",
    )
    ap.add_argument(
        "--joint-limit-penalty",
        type=float,
        default=20.0,
        help="Weight on joint_constraint_force_penalty (template: 1.0). At 1.0 it was 1.5% of the "
        "weighted reward while the prosthetic ankle spent 43% of steps past its own limit. 20 puts "
        "it at roughly 1.4x the muscle activation penalty and 40% of forward_reward, which is the "
        "range where a term in this reward has been observed to shape behaviour. Filenames gain a "
        "_jlim<value> suffix when this differs from the default.",
    )
    ap.add_argument(
        "--anneal-imitation",
        action="store_true",
        help="Ramp the imitation weights to zero and the forward/effort weights up, so imitation "
        "bootstraps and then leaves. Needs a much longer run than 30M, since the annealed "
        "objective has to rediscover gait once the scaffold is gone. Filenames gain _anneal.",
    )
    ap.add_argument("--anneal-start", type=float, default=0.2, help="Fraction of the run where the ramp begins.")
    ap.add_argument("--anneal-end", type=float, default=0.6, help="Fraction of the run where the ramp completes.")
    args = ap.parse_args()

    template = json.loads(TEMPLATE.read_text())
    template_env = template["env_params"]

    devices = args.devices or sorted(PROSTHESES)
    unknown = set(devices) - set(PROSTHESES)
    assert not unknown, f"unknown or non-prosthetic device keys: {sorted(unknown)}"

    suffix = ""
    if args.human_net:
        suffix += f"_h{args.human_net}"
    if args.device_net:
        suffix += f"_d{args.device_net}"
    if args.device_activation_penalty != 1.0:
        suffix += f"_devpen{f'{args.device_activation_penalty:g}'.replace('.', 'p')}"
    if args.device_ctrl_scale != 1.0:
        suffix += f"_cap{f'{args.device_ctrl_scale:g}'.replace('.', 'p')}"
    if args.curriculum_start_velocity != 0.3:
        suffix += f"_curr{f'{args.curriculum_start_velocity:g}'.replace('.', 'p')}"
    if args.anneal_imitation:
        suffix += "_anneal"
    if args.joint_limit_penalty != 20.0:
        suffix += f"_jlim{f'{args.joint_limit_penalty:g}'.replace('.', 'p')}"
    if args.out_of_trajectory_threshold != 0.4:
        suffix += f"_oot{f'{args.out_of_trajectory_threshold:g}'.replace('.', 'p')}"
    if args.forward_reward != 20.0:
        suffix += f"_fwd{f'{args.forward_reward:g}'.replace('.', 'p')}"
    if args.muscle_activation_penalty is not None:
        suffix += f"_actpen{f'{args.muscle_activation_penalty:g}'.replace('.', 'p')}"

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for device in devices:
        facts = _model_facts(device)
        cfg = json.loads(TEMPLATE.read_text())
        env = cfg["env_params"]

        # `safe_height` is an absolute pelvis_ty, so a composition that stands lower gets a
        # smaller fall margin from the same number. The intact configs allow ~0.21 m of drop
        # (0.915 standing against 0.7); left at 0.7 the OpenSourceLeg models, which stand at
        # 0.823, would be declared fallen after 0.12 m -- roughly half the room to recover.
        # Derived so every composition gets the same margin.
        env["safe_height"] = round(_standing_height(device) - args.fall_margin, 4)

        env["msk_key"] = MSK
        env["device_key"] = device
        env["comment"] = f"amputee model: {MSK} + {device}, {PROSTHESES[device]}"

        # -- observation keys ------------------------------------------------------------
        pos_keys, prosthetic_pos = _substitute(template_env["observation_joint_pos_keys"], facts["joints"])
        vel_keys, prosthetic_vel = _substitute(template_env["observation_joint_vel_keys"], facts["joints"])
        sens_keys = template_env["observation_joint_sensor_keys"]
        missing_sensors = [s for s in sens_keys if s not in facts["sensors"]]
        assert not missing_sensors, f"{device}: composed model has no sensor {missing_sensors}"
        env["observation_joint_pos_keys"] = pos_keys
        env["observation_joint_vel_keys"] = vel_keys

        # The amputation takes the joint-limit sensors of the removed joints with it -- every
        # prosthesis here drops `r_mtp_sensor`. The constraint-force penalty reads each name
        # unguarded, so the class default would raise on the first step.
        from rl_train.envs.myoassist_leg_base import MyoAssistLegBase

        env["joint_limit_sensor_keys"] = [s for s in MyoAssistLegBase.JOINT_LIMIT_SENSOR_NAMES if s in facts["sensors"]]

        # -- reference and imitation keys ------------------------------------------------
        env["reference_data_keys"] = _keep_present(template_env["reference_data_keys"], facts["joints"])
        # The reference writes nothing to the prosthetic DOFs, and `reset` seeds the next episode
        # from the current qpos, so without this they start each episode wherever the last fall
        # left them -- measured well outside their own limits. See
        # MyoAssistLegImitation._reset_keyframe_joints.
        env["reset_keyframe_joint_keys"] = sorted(set(prosthetic_pos) | set(prosthetic_vel))
        rewards = env["reward_keys_and_weights"]
        side = _amputated_side(prosthetic_pos)
        for block in ("qpos_imitation_rewards", "qvel_imitation_rewards"):
            present = {k: v for k, v in rewards[block].items() if k in facts["joints"]}
            intact = _drop_amputated_side(present, side)
            residual, _ = _residual_and_replaced(list(rewards[block]), facts["joints"], side)
            # The residual limb gets a weak posture term, not the template's tracking weight: the
            # point is to stop the knee folding, not to demand healthy swing-phase kinematics the
            # missing push-off cannot produce.
            rewards[block] = {**intact, **{k: args.residual_imitation_weight for k in residual}}

        # Termination watches the pelvis and the intact leg only. The residual limb is in the
        # reward above but deliberately not here: it is the joint that deviates most from a
        # healthy walker, and on the first 30M runs `knee_angle_r` was the most frequent cause of
        # termination. The joints the device replaced appear in neither.
        env["out_of_trajectory_joint_keys"] = sorted(rewards["qpos_imitation_rewards"].keys() - set(residual))
        env["out_of_trajectory_threshold"] = args.out_of_trajectory_threshold

        # Forward progress has to be the dominant objective here, which it is not on an intact
        # model. There, tracking the reference *is* walking, so an imitation-dominant reward
        # (measured 84% of the intact control's total against forward's 3.7%) produces gait. On an
        # amputee the imitation term covers only the pelvis and the intact leg, so its optimum is
        # reachable while the prosthetic leg drags: measured on the 30M run, imitation was 77.4%
        # of the total and forward 5.7%, and the learned policies dragged the foot or folded the
        # residual knee.
        rewards["forward_reward"] = args.forward_reward

        # Device effort is priced, not free. At zero the policy saturates the actuator, which on
        # OpenSourceLeg_A_L1 means 168 N*m into an ankle with no damping and no armature: measured
        # on the 30M run the command sat at saturation and the joint ran 4.3 rad past its +-0.52
        # rad limit. The muscle term averages over `n_muscle` actuators and this one over
        # `n_motor`, so the per-actuator price here is deliberately above a muscle's
        # (10/18 = 0.56 for the transtibial models).
        rewards["exo_activation_penalty"] = args.device_activation_penalty
        env["device_ctrl_scale"] = args.device_ctrl_scale

        # The limit penalty exists in the template at weight 1.0 but is far too small to matter
        # here. Measured on the 100M spec-torque policy: the prosthetic ankle sat 0.74 rad past
        # its +-0.52 rad range for 43% of steps, drawing a mean 71.7 N*m of constraint force, and
        # the term still came to 1.5% of the weighted reward against forward_reward's 73%. Per
        # step that is 0.0027 against 0.1485, so violating the limit paid about 55 times what it
        # cost. The normalisation is part of why -- it divides a joint torque by body weight,
        # which is dimensionally a length, so the scale is arbitrary rather than chosen.
        rewards["joint_constraint_force_penalty"] = args.joint_limit_penalty

        # Speed curriculum. The amputee models only ever produced sustained stepping at
        # 0.2-0.35 m/s while the reward demanded 1.25 from the first step; measured on the 30M
        # NEUankle_L1 run, the four episodes that lasted past 5 s all travelled at 0.21-0.29 m/s.
        # Ramping the demand lets that regime be reached before it is asked to be fast.
        if args.anneal_imitation:
            # Imitation as a scaffold, not the objective. On an amputee the reference cannot
            # describe the affected side, so what it can teach is a posture to bootstrap from;
            # annealing it away leaves forward progress against effort, which the measured
            # effort cost can distinguish (one-legged hopping ran 0.237 per metre against
            # 0.110-0.131 for the policies that used the prosthesis).
            rewards["forward_reward"] = args.forward_reward
            rewards["muscle_activation_penalty"] = args.muscle_activation_penalty or 10.0
            env["reward_curriculum"] = {
                "qpos_imitation_rewards": [1.0, 0.0],
                "qvel_imitation_rewards": [1.0, 0.0],
                "forward_reward": [0.2, 1.0],
                "muscle_activation_penalty": [0.2, 1.0],
            }
            env["reward_curriculum_start"] = args.anneal_start
            env["reward_curriculum_end"] = args.anneal_end

        env["curriculum_start_velocity"] = args.curriculum_start_velocity
        env["curriculum_fraction"] = args.curriculum_fraction
        # Required for the curriculum to mean anything: the reference walks at 1.281 m/s and its
        # index advances one frame per control step, so at a 0.3 m/s target the qpos terms would
        # still demand full-stride angles at full cadence while forward_reward asks for 0.3 m/s.
        # No gait satisfies both.
        env["scale_reference_playback"] = args.curriculum_start_velocity > 0
        if args.muscle_activation_penalty is not None:
            rewards["muscle_activation_penalty"] = args.muscle_activation_penalty

        if args.total_timesteps is not None:
            cfg["total_timesteps"] = args.total_timesteps

        # The mirror penalty needs a left/right actuator permutation, which does not exist on an
        # asymmetric model. Written explicitly rather than left absent: this is a property of the
        # amputee model, not an oversight.
        cfg["ppo_params"]["mirror_coef"] = 0.0

        # -- observation layout ----------------------------------------------------------
        # qpos | qvel | act (muscles only) | sensor | target_velocity
        qpos_offset = 0
        qvel_offset = len(pos_keys)
        act_offset = qvel_offset + len(vel_keys)
        sens_offset = act_offset + facts["n_muscle"]
        tv_offset = sens_offset + len(sens_keys)
        obs_len = tv_offset + 1

        full_obs = [
            {"type": "range", "range": [qpos_offset, qvel_offset], "comment": f"{len(pos_keys)} qpos"},
            {"type": "range", "range": [qvel_offset, act_offset], "comment": f"{len(vel_keys)} qvel"},
            {"type": "range", "range": [act_offset, sens_offset], "comment": f"{facts['n_muscle']} muscle activation"},
            {"type": "range", "range": [sens_offset, tv_offset], "comment": f"{len(sens_keys)} foot force"},
            {"type": "range", "range": [tv_offset, obs_len], "comment": "target velocity"},
        ]

        custom = cfg["policy_params"]["custom_policy_params"]
        net_arch = custom["net_arch"]
        if args.human_net:
            net_arch["human_actor"] = [args.human_net, args.human_net]

        if facts["n_motor"] == 0:
            # Muscle-only model: HumanActorCriticPolicy reads the whole observation and writes
            # the whole action, and `net_indexing_info` is unused. Leaving the exo entries in
            # would describe a layout nothing implements, and an exo env id would be rejected by
            # `_validate_action_layout` for having no motor actuator to drive.
            env["env_id"] = "myoAssistLegImitation-v0"
            custom["net_indexing_info"] = {}
            net_arch.pop("exo_actor", None)
        else:
            env["env_id"] = "myoAssistLegImitationExo-v0"
            if args.device_net:
                net_arch["exo_actor"] = [args.device_net, args.device_net]
            # The prosthesis actor sees the DOFs it drives plus both feet's contact force: a
            # one-sided device still has to time its push-off against the gait cycle, and the
            # contralateral contact is what tells it where in that cycle it is.
            device_obs_idx = _indices(pos_keys, prosthetic_pos, qpos_offset) + _indices(vel_keys, prosthetic_vel, qvel_offset)
            custom["net_indexing_info"] = {
                "human_actor": {
                    "observation": full_obs,
                    "action": [
                        {
                            "type": "range_mapping",
                            "range_net": [0, facts["n_muscle"]],
                            "range_action": [0, facts["n_muscle"]],
                            "comment": f"{facts['n_muscle']} muscle activation (amputee: "
                            f"the prosthetic side has fewer muscles than the intact side)",
                        }
                    ],
                },
                "exo_actor": {
                    "observation": [
                        {
                            "type": "index",
                            "index": device_obs_idx,
                            "comment": f"prosthetic joint angle and angular velocity ({', '.join(prosthetic_pos)})",
                        },
                        {
                            "type": "range",
                            "range": [sens_offset, tv_offset],
                            "comment": f"{len(sens_keys)} foot contact ({', '.join(sens_keys)})",
                        },
                    ],
                    "action": [
                        {
                            "type": "range_mapping",
                            "range_net": [0, facts["n_motor"]],
                            "range_action": [facts["n_muscle"], facts["nu"]],
                            "comment": f"{facts['n_motor']} command(s) for {', '.join(facts['motor_names'])}",
                        }
                    ],
                },
                "common_critic": {"observation": full_obs},
            }

        out = out_dir / f"imitation_22_{device}{suffix}.json"
        out.write_text(json.dumps(cfg, indent=4) + "\n")
        dropped = sorted(
            set(template_env["reward_keys_and_weights"]["qpos_imitation_rewards"]) - set(rewards["qpos_imitation_rewards"])
        )
        print(
            f"  {out.name:52} obs={obs_len:3} act={facts['nu']:3} "
            f"({facts['n_muscle']} muscle + {facts['n_motor']} motor)  oot={args.out_of_trajectory_threshold}"
        )
        print(f"       prosthetic_dof={prosthetic_pos}")
        print(f"       imitation drops the '{side}' side: {dropped}  -> tracks {sorted(rewards['qpos_imitation_rewards'])}")


if __name__ == "__main__":
    main()
