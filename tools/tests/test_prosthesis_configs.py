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
def test_amputated_side_guides_but_does_not_terminate(config_path):
    """The residual limb is in the reward, the amputated side is out of the termination check.

    Three separate rules, and the first 30M runs broke each of them in a way that showed up in
    the learned gait:

    * A joint the *device* replaced has no healthy counterpart, so it appears in neither the
      reward nor the check -- covered by `test_reference_and_imitation_keys_exist_in_model`,
      since those joint names are absent from the composed model entirely.
    * A joint the person still has (the residual hip, and the knee for a transtibial device) does
      belong in the reward. With the whole side dropped, nothing in the reward referred to it and
      the policies folded the residual knee or dragged the foot.
    * Neither belongs in the termination check. `knee_angle_r` was the single most frequent cause
      of episode termination, because a residual limb without an ankle push-off cannot produce
      healthy swing-phase kinematics.
    """
    env_params = json.loads(config_path.read_text())["env_params"]
    rewards = env_params["reward_keys_and_weights"]
    oot = env_params["out_of_trajectory_joint_keys"]
    side = "r"  # every shipped prosthesis amputates the right leg

    assert oot, f"{config_path.name}: empty out_of_trajectory_joint_keys falls back to the whole imitation dict"
    offending = [k for k in oot if k.endswith(f"_{side}")]
    assert not offending, f"{config_path.name}: the termination check still watches the amputated side: {offending}"
    assert set(oot) <= set(rewards["qpos_imitation_rewards"]), (
        f"{config_path.name}: out_of_trajectory_joint_keys names joints the imitation reward does "
        f"not track, so there is no reference to compare them against: "
        f"{sorted(set(oot) - set(rewards['qpos_imitation_rewards']))}"
    )

    residual = [k for k in rewards["qpos_imitation_rewards"] if k.endswith(f"_{side}")]
    assert residual, (
        f"{config_path.name}: no residual-limb joint is in the imitation reward. The amputation "
        "leaves the person their own hip (and knee, transtibially); with nothing in the reward "
        "referring to those joints the policy has no reason not to fold them."
    )

    assert env_params["out_of_trajectory_threshold"] > 0.2, (
        f"{config_path.name}: threshold is the intact-config value; the intact leg of an amputee "
        "compensates for the missing push-off and does not track a healthy walker as tightly."
    )
    assert rewards["forward_reward"] > sum(rewards["qpos_imitation_rewards"].values()), (
        f"{config_path.name}: forward_reward ({rewards['forward_reward']}) does not outweigh the "
        f"imitation block ({sum(rewards['qpos_imitation_rewards'].values())}). On an amputee the "
        "imitation optimum is reachable with the prosthetic leg dragging, so forward progress has "
        "to be the dominant objective."
    )


@pytest.mark.parametrize("config_path", shipped_prosthesis_configs(), ids=lambda p: p.stem.replace("imitation_22_", ""))
def test_reset_places_the_model_on_the_ground(config_path):
    """A reset pose sits at the height the composition can actually stand at.

    `reset` takes the initial pose from the reference, whose `q_pelvis_ty` describes a bare
    musculoskeletal model. `myolegs22 + OpenSourceLeg_A_L1` stands at 0.823 m against the
    reference's 0.906 m mean, so before `_standing_pelvis_height` learned to detect it, every
    episode began with the pelvis 8.3 cm above the height at which the feet reach the floor --
    the model free-fell into each episode and the imitation term rewarded holding it up there.
    The keyframe could not reveal this: those two compositions report no ground contact at their
    own keyframe, which is exactly the signal the fix keys off.

    Checked as a fall distance rather than a contact count, because the reference is a walking
    trajectory and a mid-swing pose legitimately has both feet off the ground.
    """
    import numpy as np

    _, env = build_prosthesis_env(config_path)
    try:
        standing = env._standing_pelvis_height()
        drops = []
        for _ in range(12):
            env.reset()
            drops.append(float(env.sim.data.joint("pelvis_ty").qpos[0]) - standing)
        worst = max(drops)
        assert worst < 0.06, (
            f"{config_path.name}: a reset starts up to {worst * 100:.1f} cm above the height this "
            f"composition stands at ({standing:.3f} m). The reference pelvis height is not being "
            "corrected for this model."
        )
        assert np.mean(drops) < 0.04, f"{config_path.name}: resets average {np.mean(drops) * 100:.1f} cm above standing height"
    finally:
        env.close()


@pytest.mark.parametrize("config_path", shipped_prosthesis_configs(), ids=lambda p: p.stem.replace("imitation_22_", ""))
def test_fall_margin_matches_the_intact_configs(config_path):
    """`safe_height` leaves the same room to recover whatever height the device stands at.

    It is an absolute `pelvis_ty`, so the shipped 0.7 means 0.21 m of drop on an intact model
    (standing 0.915) but only 0.12 m on the OpenSourceLeg compositions (standing 0.823) -- they
    would be called fallen after roughly half the descent.
    """
    env_params = json.loads(config_path.read_text())["env_params"]
    _, env = build_prosthesis_env(config_path)
    try:
        margin = env._standing_pelvis_height() - env_params["safe_height"]
    finally:
        env.close()
    assert 0.18 < margin < 0.24, (
        f"{config_path.name}: safe_height {env_params['safe_height']} leaves {margin:.3f} m of "
        "fall margin; the intact configs allow ~0.21 m."
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
