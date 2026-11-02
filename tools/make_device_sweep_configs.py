"""Generate one imitation-training config per drop-in exo device.

A device sweep is only interpretable if the *only* thing that varies is the device, so
these are copies of the maintained `imitation_tutorial_22_separated_net_partial_obs`
config with two things changed: `device_key`, and the observation slice the exo sub-policy
reads.

That second change is the reason this is not a for-loop over `device_key`. The exo policy
is given the joint it actuates, and the drop-in devices do not all actuate the same joint:

    ankle   DephyExoBoot_L1, Humotech_L1, OpenExo_L1, Tutorial_L1, STRIDE_L2
    hip     Hippo_L1, HMEDI_L1

Verified against the composed models rather than assumed:

  * the four ankle `joint` devices drive `ankle_angle_r` / `ankle_angle_l` directly.
  * `Hippo_L1` drives `hip_flexion_r` / `hip_flexion_l` directly.
  * `HMEDI_L1` is tendon-driven; its cable runs from the device torso to the thigh and the
    only human joint it crosses is `hip_flexion`, so it is a hip device.
  * `STRIDE_L2` is tendon-driven through its own linkage, so the tendon path names no human
    joint. It attaches to `tibia`, `calcn` and `toes` and its linkage drives the foot, so it
    is an ankle (and forefoot) device.

Not varied, deliberately:

  * `ctrlrange` differs across devices (`[-1,0]` for the unidirectional ankle devices,
    `[-1,1]` for Hippo, `[0,400]` N for STRIDE), but the env normalises the action space to
    `[-1,1]` for every one of them, so the `Tanh` policy output needs no per-device scaling.
  * `STRIDE_L2` composes to `nq=49` instead of 39 because the linkage carries its own DOF.
    Observations are built from named human joints, so the extra DOF change nothing here.
  * rewards, velocity target, episode length, PPO hyperparameters and `total_timesteps`
    stay identical, so any difference in outcome is attributable to the device.
"""

import argparse
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
CFG_DIR = REPO / "rl_train" / "train" / "train_configs"
TEMPLATE = CFG_DIR / "imitation_tutorial_22_separated_net_partial_obs.json"
OUT_DIR = CFG_DIR / "device_sweep"

# joint the device assists -> the qpos key pair the exo policy should observe.
# These are the bilateral powered devices, the class this sweep is about: two device actuators,
# the full 22-muscle model, and a left/right mirror pair the symmetry machinery can use.
ASSISTED_JOINT = {
    "DephyExoBoot_L1": "ankle_angle",
    "Humotech_L1": "ankle_angle",
    "OpenExo_L1": "ankle_angle",
    "Tutorial_L1": "ankle_angle",
    "STRIDE_L2": "ankle_angle",
    "UTAnkleExo_L2": "ankle_angle",
    "Hippo_L1": "hip_flexion",
    "HMEDI_L1": "hip_flexion",
}

# `python -m assist_sim list` offers 13 devices for myolegs22; the five not above are excluded
# because they are a different class, not because they were overlooked:
#
#   Anatomics_L1, KFoot_L1   no device actuator at all (nu == na == the muscle count), so there
#                            is nothing for an exo sub-policy to drive. An exo config would fail
#                            _validate_action_layout, which is the correct outcome.
#   NEUankle_L1              unilateral: one right-ankle actuator, and the right leg is missing
#                            edl/fdl/soleus/tibant (18 muscles, nq 38). The human model itself is
#                            asymmetric, so no left/right mirror map exists -- `action_permutation`
#                            raises on it, again correctly.
#   OpenSourceLeg_A_L1       unilateral ankle prosthesis, 18 muscles, ctrlrange [-2.88, 2.88].
#   OpenSourceLeg_KA_L1      unilateral knee+ankle prosthesis, 15 muscles, nq 30.
#
# The three unilateral models need their own action layout and cannot use the mirror penalty or
# the per-side shared network, so they are not a row in this sweep. They have their own generator:
# tools/make_prosthesis_configs.py, which also covers the passive KFoot_L1.


def _mirror_name(name: str) -> str:
    """`ankle_angle_l` -> `ankle_angle_r`, `r_foot` -> `l_foot`; side-less names unchanged.

    Same rule as `rl_train/train/policies/mirror.py`, restated here because this generator
    runs without composing a model. The two must agree, which the emitted config makes
    checkable: a per-side exo config is only exactly symmetric if the left ordering really is
    the mirror permutation of the right.
    """
    other = {"l": "r", "r": "l", "L": "R", "R": "L"}
    if len(name) > 2 and name[-2] == "_" and name[-1] in other:
        return name[:-1] + other[name[-1]]
    if len(name) > 2 and name[1] == "_" and name[0] in other:
        return other[name[0]] + name[1:]
    return name


def _own_side_first(keys: list[str], wanted: list[str], offset: int, side: str) -> list[int]:
    """Global observation indices for `wanted`, that leg's entry first within each pair.

    `wanted` names the right-side channels; passing side "l" returns their mirrors in the
    same relative order, which is exactly the ordering the shared per-side network needs.
    """
    index = {k: i + offset for i, k in enumerate(keys)}
    out = []
    for name in wanted:
        own = name if side == "r" else _mirror_name(name)
        contra = _mirror_name(own)
        for n in (own, contra):
            assert n in index, f"{n} not in {keys}"
            out.append(index[n])
    return out


def _make_shared_side_exo(cfg: dict, env: dict, joint: str, exo_net_width: int | None) -> None:
    """Swap the single `exo_actor` for a weight-shared pair of per-leg actors.

    Each side reads the same channels in mirrored order and emits one command, so the shared
    weights make the two exo outputs exact mirrors of each other. The contact sensors are
    included unconditionally: a per-leg network with only its own joint angle cannot tell
    stance from swing, so it has nothing to time push-off against.
    """
    custom = cfg["policy_params"]["custom_policy_params"]
    nets, arch = custom["net_indexing_info"], custom["net_arch"]

    action_range = nets["exo_actor"]["action"][0]["range_action"]
    assert action_range[1] - action_range[0] == 2, (
        f"expected exactly two exo action slots (Exo_R then Exo_L), got {action_range}"
    )
    right_slot, left_slot = action_range[0], action_range[0] + 1

    pos_keys, vel_keys = env["observation_joint_pos_keys"], env["observation_joint_vel_keys"]
    sens_keys = env["observation_joint_sensor_keys"]
    # Observation layout: qpos | qvel | act | sensor | target_velocity.
    sens_offset = len(pos_keys) + len(vel_keys) + 22  # 22 muscle activations
    right_sensors = [k for k in sens_keys if _mirror_name(k) != k and k.startswith("r")]

    for side, slot in (("r", right_slot), ("l", left_slot)):
        indices = (
            _own_side_first(pos_keys, [f"{joint}_r"], 0, side)
            + _own_side_first(vel_keys, [f"{joint}_r"], len(pos_keys), side)
            + _own_side_first(sens_keys, right_sensors, sens_offset, side)
        )
        nets[f"exo_actor_{side}"] = {
            "observation": [
                {
                    "type": "index",
                    "index": indices,
                    "comment": f"{joint} angle, angular velocity and foot contact, "
                    f"{'right' if side == 'r' else 'left'} leg first then contralateral",
                }
            ],
            "action": [
                {
                    "type": "range_mapping",
                    "range_net": [0, 1],
                    "range_action": [slot, slot + 1],
                    "comment": f"1 command for Exo_{side.upper()}",
                }
            ],
        }
        if exo_net_width:
            arch[f"exo_actor_{side}"] = [exo_net_width, exo_net_width]
        else:
            arch[f"exo_actor_{side}"] = list(arch["exo_actor"])

    # The two orderings must be elementwise mirrors, or shared weights do not give symmetry.
    right_idx = nets["exo_actor_r"]["observation"][0]["index"]
    left_idx = nets["exo_actor_l"]["observation"][0]["index"]
    all_keys = pos_keys + vel_keys + ["act"] * 22 + sens_keys
    for a, b in zip(right_idx, left_idx):
        assert _mirror_name(all_keys[a]) == all_keys[b], (
            f"per-side orderings are not mirrors at this position: {all_keys[a]} vs {all_keys[b]}"
        )

    del nets["exo_actor"]
    del arch["exo_actor"]


def _obs_slice(keys: list[str], base: str, offset: int) -> list[int]:
    """Contiguous [start, end) of `<base>_l` / `<base>_r` within an observation block."""
    idx = sorted(i for i, k in enumerate(keys) if k in (f"{base}_l", f"{base}_r"))
    assert idx, f"{base} not found in {keys}"
    assert idx == list(range(idx[0], idx[-1] + 1)), f"{base} keys are not contiguous: {idx}"
    return [offset + idx[0], offset + idx[-1] + 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--muscle-activation-penalty",
        type=float,
        default=None,
        help="Override reward_keys_and_weights.muscle_activation_penalty (template value: 0.1). "
        "Measured contribution at 0.1 is ~0.4%% of total reward magnitude, so raising it is how "
        "you make muscle effort expensive enough that the (unpenalised) exo is worth using. "
        "Output filenames gain an _actpen<value> suffix so variants sit alongside the baseline.",
    )
    ap.add_argument(
        "--exo-activation-penalty",
        type=float,
        default=None,
        help="Price of device effort, same units as --muscle-activation-penalty (both are dt "
        "times a mean dimensionless effort). Device ctrl is normalised by each actuator's own "
        "ctrlrange, so one weight means the same thing on every device in the sweep. The muscle "
        "mean is over 22 actuators and the device mean over 2, so at muscle weight 10 a value of "
        "0.1 makes one device actuator about ten times cheaper than one muscle. Motivation: "
        "without it the device's torque is free and the policy adds assistance wherever it helps "
        "even marginally, which shows up as exo torque in early stance. Filenames gain "
        "_exopen<value>.",
    )
    ap.add_argument("--devices", nargs="*", default=None, help="Subset of device keys (default: all).")
    ap.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=OUT_DIR,
        help="Where to write the configs. Defaults to the shipped device_sweep directory; point it "
        "at a temporary directory when generating throwaway variants, so tests and experiments do "
        "not leave files in the set that is meant to be deployed.",
    )
    ap.add_argument(
        "--mirror-coef",
        type=float,
        default=None,
        help="Weight on the left/right mirror-symmetry penalty (see rl_train/train/mirror_ppo.py). "
        "0 or unset leaves training as plain PPO. Filenames gain a _mirror<value> suffix.",
    )
    ap.add_argument("--human-net", type=int, default=None, help="Width of both human_actor hidden layers (template: 64).")
    ap.add_argument("--exo-net", type=int, default=None, help="Width of both exo_actor hidden layers (template: 8).")
    ap.add_argument(
        "--exo-contact",
        action="store_true",
        help="Also feed the exo sub-policy the four foot-contact sensors. Without them it sees "
        "only its own joint's angle and velocity, so it cannot tell which foot is loaded -- "
        "the information a push-off assist needs to time each side independently.",
    )
    ap.add_argument(
        "--exo-shared-side-net",
        action="store_true",
        help="Replace the single exo sub-policy with one shared network applied to each leg in "
        "turn, that leg's own inputs first (`exo_actor_r` / `exo_actor_l` in the config). Left "
        "and right exo commands then satisfy Exo_L(s) = Exo_R(mirror(s)) exactly, by "
        "construction, instead of being nudged there by the mirror penalty. Filenames gain a "
        "_sidenet suffix. Implies the contact inputs, since a per-leg network that cannot see "
        "which foot is loaded has no way to time its own push-off.",
    )
    ap.add_argument(
        "--exo-obs-right-first",
        action="store_true",
        help="Feed the exo sub-policy its joint right-side-first, matching its action slots "
        "(22 is Exo_R, 23 is Exo_L) instead of the left-first order the observation keys "
        "happen to be in. Emits `index` mappings rather than `range` ones. Filenames gain an "
        "_exoRL suffix. Motivation: with the left-first default the sub-policy has to learn a "
        "crossed mapping, and at penalty 10 it drives the right exo four times harder than the "
        "left even though the model, the reference and the learned gait are all symmetric.",
    )
    args = ap.parse_args()

    template = json.loads(TEMPLATE.read_text())
    env = template["env_params"]
    pos_keys, vel_keys = env["observation_joint_pos_keys"], env["observation_joint_vel_keys"]
    # Observation layout: qpos | qvel | act | sensor | target_velocity.
    qpos_offset, qvel_offset = 0, len(pos_keys)

    devices = args.devices or sorted(ASSISTED_JOINT)
    unknown = set(devices) - set(ASSISTED_JOINT)
    assert not unknown, f"unknown device keys: {sorted(unknown)}"

    suffix = "_exoRL" if args.exo_obs_right_first else ""
    if args.human_net:
        suffix += f"_h{args.human_net}"
    if args.exo_net:
        suffix += f"_e{args.exo_net}"
    if args.exo_contact:
        suffix += "_contact"
    if args.exo_shared_side_net:
        suffix += "_sidenet"
    if args.mirror_coef:
        suffix += f"_mirror{f'{args.mirror_coef:g}'.replace('.', 'p')}"
    if args.exo_activation_penalty is not None:
        suffix += f"_exopen{f'{args.exo_activation_penalty:g}'.replace('.', 'p')}"
    if args.muscle_activation_penalty is not None:
        # 10.0 -> "10", 2.5 -> "2p5", so the filename stays shell- and glob-friendly.
        pretty = f"{args.muscle_activation_penalty:g}".replace(".", "p")
        suffix += f"_actpen{pretty}"

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for device in devices:
        joint = ASSISTED_JOINT[device]
        cfg = json.loads(TEMPLATE.read_text())
        cfg["env_params"]["device_key"] = device
        if args.muscle_activation_penalty is not None:
            cfg["env_params"]["reward_keys_and_weights"]["muscle_activation_penalty"] = args.muscle_activation_penalty
        # Always written, even when zero: a generated config should state every reward weight it
        # runs under rather than leaving a reader to find the dataclass default.
        cfg["env_params"]["reward_keys_and_weights"]["exo_activation_penalty"] = args.exo_activation_penalty or 0.0

        exo = cfg["policy_params"]["custom_policy_params"]["net_indexing_info"]["exo_actor"]
        qpos_range = _obs_slice(pos_keys, joint, qpos_offset)
        qvel_range = _obs_slice(vel_keys, joint, qvel_offset)
        if args.exo_obs_right_first:
            # qpos_range/qvel_range are [left, right]; reverse each so the sub-policy sees
            # right before left, the order its action slots are in.
            exo["observation"] = [
                {
                    "type": "index",
                    "index": [qpos_range[1] - 1, qpos_range[0]],
                    "comment": f"2 {joint} angle, right then left (matches Exo_R, Exo_L)",
                },
                {
                    "type": "index",
                    "index": [qvel_range[1] - 1, qvel_range[0]],
                    "comment": f"2 {joint} angular velocity, right then left",
                },
            ]
        else:
            exo["observation"] = [
                {"type": "range", "range": qpos_range, "comment": f"2 {joint} angle (device assists this joint)"},
                {"type": "range", "range": qvel_range, "comment": f"2 {joint} angular velocity"},
            ]

        if args.exo_contact:
            # Sensor block sits after qpos | qvel | act in the observation vector.
            n_act = 22  # muscles; the exo actuators carry no activation state on these devices
            sens_start = len(pos_keys) + len(vel_keys) + n_act
            n_sens = len(env["observation_joint_sensor_keys"])
            exo["observation"].append(
                {
                    "type": "range",
                    "range": [sens_start, sens_start + n_sens],
                    "comment": f"{n_sens} foot contact ({', '.join(env['observation_joint_sensor_keys'])})",
                }
            )

        if args.mirror_coef:
            cfg["ppo_params"]["mirror_coef"] = args.mirror_coef

        net_arch = cfg["policy_params"]["custom_policy_params"]["net_arch"]
        if args.human_net:
            net_arch["human_actor"] = [args.human_net, args.human_net]
        if args.exo_net:
            net_arch["exo_actor"] = [args.exo_net, args.exo_net]

        # Last, so it can consume the widths set above and remove the `exo_actor` entries it
        # replaces without them being written back.
        if args.exo_shared_side_net:
            _make_shared_side_exo(cfg, env, joint, args.exo_net)

        out = out_dir / f"imitation_22_{device}{suffix}.json"
        out.write_text(json.dumps(cfg, indent=4) + "\n")
        pen = cfg["env_params"]["reward_keys_and_weights"]["muscle_activation_penalty"]
        print(f"  {out.name:46} joint={joint:12} qpos={qpos_range} qvel={qvel_range} act_penalty={pen}")


if __name__ == "__main__":
    main()
