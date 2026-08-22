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
        "--device-activation-penalty",
        type=float,
        default=None,
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
    if args.device_activation_penalty is not None:
        suffix += f"_devpen{f'{args.device_activation_penalty:g}'.replace('.', 'p')}"
    if args.out_of_trajectory_threshold != 0.4:
        suffix += f"_oot{f'{args.out_of_trajectory_threshold:g}'.replace('.', 'p')}"
    if args.muscle_activation_penalty is not None:
        suffix += f"_actpen{f'{args.muscle_activation_penalty:g}'.replace('.', 'p')}"

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for device in devices:
        facts = _model_facts(device)
        cfg = json.loads(TEMPLATE.read_text())
        env = cfg["env_params"]

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
            rewards[block] = _drop_amputated_side(present, side)

        # The same dict is the episode's out-of-trajectory check, so this threshold now applies
        # only to the pelvis and the intact leg. It is still relaxed relative to the intact
        # configs' 0.2: the intact side of an amputee gait compensates for the missing push-off
        # and does not track a healthy walker as tightly either -- `ankle_angle_l` was the third
        # most frequent termination cause on the 30M run.
        env["out_of_trajectory_threshold"] = args.out_of_trajectory_threshold
        if args.muscle_activation_penalty is not None:
            rewards["muscle_activation_penalty"] = args.muscle_activation_penalty
        # Always written, even when zero, so a generated config states every weight it runs under.
        rewards["exo_activation_penalty"] = args.device_activation_penalty or 0.0

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
