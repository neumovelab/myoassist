"""Every shipped prosthesis config builds an amputee env, keeps its layout, and trains a step.

The amputee configs are the ones a layout change breaks silently. The device performs the
amputation, so the muscle count, the observation length and the joint and sensor names all
differ per device and none of them match the intact 22-muscle configs the rest of the suite
covers. A config addresses observations by absolute index, so a device model that gains or
loses a muscle repoints every sub-policy without raising.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tools.tests.conftest import take_a_few_ppo_steps

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROSTHESIS_DIR = REPO_ROOT / "rl_train/train/train_configs/prosthesis"

# device -> (observation length, action length, muscle count, joints with no reference trajectory)
# Read off the composed models; see tools/make_prosthesis_configs.py for how they are derived.
EXPECTED = {
    "OpenSourceLeg_A_L1": (40, 19, 18, ["OSL_A_L1_osl_ankle_angle_r"]),
    "NEUankle_L1": (40, 19, 18, ["NEUankle_L1_neuankle_ankle_angle_r"]),
    "OpenSourceLeg_KA_L1": (37, 17, 15, ["OSL_KA_L1_osl_ankle_angle_r", "OSL_KA_L1_osl_knee_angle_r"]),
    "KFoot_L1": (42, 18, 18, ["KFoot_L1_df_ankle_angle_r", "KFoot_L1_pf_ankle_angle_r"]),
}


def shipped_prosthesis_configs() -> list[pathlib.Path]:
    configs = sorted(PROSTHESIS_DIR.glob("imitation_22_*.json"))
    assert configs, f"no shipped prosthesis configs found in {PROSTHESIS_DIR}"
    return configs


def build_prosthesis_env(config_path: pathlib.Path):
    """Compose the amputee model and build the env described by a config.

    Not `conftest.build_env`: that one hardcodes the exo config class, and the passive-foot
    config is a muscle-only session whose config class has no exo policy parameters.
    """
    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.utils.data_types import DictionableDataclass

    raw = json.loads(config_path.read_text())
    config_class = EnvironmentHandler.get_config_type_from_session_id(raw["env_params"]["env_id"])
    config = DictionableDataclass.create(config_class, raw)
    config.env_params.num_envs = 1
    config.ppo_params.n_steps = 64
    config.ppo_params.batch_size = 64
    return config, EnvironmentHandler.create_environment(config, is_rendering_on=False)


def _device_of(config_path: pathlib.Path) -> str:
    return json.loads(config_path.read_text())["env_params"]["device_key"]


@pytest.mark.parametrize("config_path", shipped_prosthesis_configs(), ids=lambda p: p.stem.replace("imitation_22_", ""))
def test_shipped_prosthesis_config_trains(config_path):
    device = _device_of(config_path)
    expected_obs, expected_act, n_muscle, _ = EXPECTED[device]

    config, env = build_prosthesis_env(config_path)
    try:
        assert env.observation_space.shape[0] == expected_obs, (
            f"{config_path.name}: observation is {env.observation_space.shape[0]}, expected {expected_obs}. "
            "The config addresses observations by absolute index, so a changed length repoints them."
        )
        assert env.action_space.shape[0] == expected_act, (
            f"{config_path.name}: action is {env.action_space.shape[0]}, expected {expected_act} "
            f"({n_muscle} muscle + {expected_act - n_muscle} device)"
        )
        # The mirror penalty needs a left/right actuator permutation, which an amputee model does
        # not have -- `mirror.action_permutation` raises on one. A config that set it would fail
        # at model construction, so this is a guard against a copy-paste from the exo sweep.
        assert config.ppo_params.mirror_coef == 0, f"{config_path.name}: an amputee model has no mirror map"
        take_a_few_ppo_steps(config, env)
    finally:
        env.close()


@pytest.mark.parametrize("config_path", shipped_prosthesis_configs(), ids=lambda p: p.stem.replace("imitation_22_", ""))
def test_reference_and_imitation_keys_exist_in_model(config_path):
    """No reference or imitation key names a joint the amputation removed.

    The reference is a healthy walker, so `short_reference_gait.npz` still carries
    `q_ankle_angle_r` for a model that has no such joint. `_follow_reference_motion` writes each
    reference key straight to `sim.data.joint(key)`, and the imitation reward reads it back, so a
    leftover key raises on the first reset -- but only for the device that removed that joint,
    which is exactly the case a hand-edited config gets wrong.
    """
    env_params = json.loads(config_path.read_text())["env_params"]
    _, env = build_prosthesis_env(config_path)
    try:
        joints = {env.sim.model.joint(i).name for i in range(env.sim.model.njnt)}
        rewards = env_params["reward_keys_and_weights"]
        named = (
            set(env_params["reference_data_keys"])
            | set(env_params["reset_keyframe_joint_keys"])
            | set(rewards["qpos_imitation_rewards"])
            | set(rewards["qvel_imitation_rewards"])
            | set(env_params["observation_joint_pos_keys"])
            | set(env_params["observation_joint_vel_keys"])
        )
        assert named <= joints, f"{config_path.name}: names joints absent from the composed model: {sorted(named - joints)}"

        sensors = {env.sim.model.sensor(i).name for i in range(env.sim.model.nsensor)}
        named_sensors = set(env_params["observation_joint_sensor_keys"]) | set(env_params["joint_limit_sensor_keys"])
        assert named_sensors <= sensors, (
            f"{config_path.name}: names sensors absent from the composed model: {sorted(named_sensors - sensors)}. "
            "Every prosthesis drops r_mtp_sensor with the amputated forefoot."
        )
    finally:
        env.close()


@pytest.mark.parametrize("config_path", shipped_prosthesis_configs(), ids=lambda p: p.stem.replace("imitation_22_", ""))
def test_imitation_excludes_the_amputated_side(config_path):
    """No imitation weight names a joint on the amputated side, and the wall is relaxed.

    The reference is a healthy, near-symmetric walker and the amputated side cannot reach it:
    without an ankle push-off the residual limb's knee does not produce the healthy swing-phase
    flexion. Because `MyoAssistLegImitation.step` reads the same dict for the
    `out_of_trajectory_threshold` check, an entry there is also a hard episode terminator. On the
    first 30M runs `knee_angle_r` was both the largest tracking error and the most frequent
    termination cause, and both prosthesis runs stopped improving at ~15M while the intact
    `Tutorial_L1` control kept going -- so a `_r` key reappearing here is a real regression.
    """
    env_params = json.loads(config_path.read_text())["env_params"]
    rewards = env_params["reward_keys_and_weights"]
    side = "r"  # every shipped prosthesis amputates the right leg
    for block in ("qpos_imitation_rewards", "qvel_imitation_rewards"):
        offending = [k for k in rewards[block] if k.endswith(f"_{side}")]
        assert not offending, (
            f"{config_path.name}: {block} still tracks the amputated side: {offending}. "
            "That both rewards an unreachable trajectory and terminates the episode on it."
        )
        assert rewards[block], f"{config_path.name}: {block} is empty; nothing would be imitated"

    assert env_params["out_of_trajectory_threshold"] > 0.2, (
        f"{config_path.name}: out_of_trajectory_threshold is "
        f"{env_params['out_of_trajectory_threshold']}, the intact-config value. The intact leg of "
        "an amputee compensates for the missing push-off and does not track a healthy walker as "
        "tightly, so the wall has to be looser here."
    )

    # The residual limb is still placed from the reference at reset -- dropping it from the
    # imitation reward is not the same as starting it from an arbitrary pose.
    assert any(k.endswith(f"_{side}") for k in env_params["reference_data_keys"]) or "KA" in config_path.name, (
        f"{config_path.name}: reference_data_keys no longer initialises any joint on the "
        "amputated side, so the residual limb would start each episode away from the gait pose"
    )


@pytest.mark.parametrize("config_path", shipped_prosthesis_configs(), ids=lambda p: p.stem.replace("imitation_22_", ""))
def test_prosthetic_joints_reset_to_keyframe(config_path):
    """The prosthesis' own DOFs start each episode at the keyframe, not where the last fall left them.

    `reset` seeds the next episode from `sim.data.qpos`, so a joint the reference does not write
    carries across the episode boundary. The prosthetic joint is the one the device actuator
    drives and the healthy reference says nothing about it, so before
    `reset_keyframe_joint_keys` existed it started successive episodes at +1.43 and +1.94 rad
    against its own +-0.52 rad limit.
    """
    device = _device_of(config_path)
    _, prosthetic_joints = EXPECTED[device][2], EXPECTED[device][3]

    _, env = build_prosthesis_env(config_path)
    try:
        keyframe = {
            j: float(env.sim.model.key_qpos[0][env.sim.model.jnt_qposadr[env.sim.model.joint(j).id]]) for j in prosthetic_joints
        }
        rng = np.random.default_rng(0)
        for episode in range(3):
            env.reset()
            for joint, expected in keyframe.items():
                actual = float(env.sim.data.joint(joint).qpos[0])
                assert actual == pytest.approx(expected, abs=1e-9), (
                    f"{config_path.name}: {joint} starts episode {episode} at {actual:+.4f}, "
                    f"not its keyframe value {expected:+.4f} -- it carried over from the previous episode"
                )
            # Drive the model into a fall, which is what leaves the joint at an extreme value.
            for _ in range(80):
                env.step(rng.uniform(-1, 1, env.sim.model.nu))
    finally:
        env.close()
