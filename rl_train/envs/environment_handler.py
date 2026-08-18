import numpy as np
import json
from stable_baselines3.common.vec_env import SubprocVecEnv
from myosuite.utils import gym
from rl_train.utils.data_types import DictionableDataclass
from rl_train.train.train_configs.config import TrainSessionConfigBase


# Env ids whose policy drives the device's non-muscle actuators (HumanExoActorCriticPolicy).
# Every other env id selects the muscle-only HumanActorCriticPolicy. Single source of truth
# for both the policy switch in get_stable_baselines3_model and the guard below, which is
# only correct as long as the two agree.
_EXO_ENV_IDS = ("myoAssistLegImitationExo-v0",)


class EnvironmentHandler:
    @staticmethod
    def _validate_action_layout(model_xml: str, config) -> None:
        """Fail fast when a config's action layout disagrees with the composed model.

        Actuator counts are a property of the composed model (msk + device), not of the
        config, so a config can declare an action layout for a model it no longer builds.
        Without this check such a mismatch surfaces as a tensor-shape error inside the
        policy forward pass, with nothing naming the config or the keys that caused it.

        Two checks:
          1. a muscle-only policy paired with a device that contributes motor actuators --
             those actuators would be left permanently unaddressed;
          2. which actuator indices the net indexing claims, against ``[0, nu)``.
             ``NetworkIndexHandler.map_network_to_action`` writes each mapping into an
             ``nu``-wide tensor as ``result[:, start:end]``, so what matters is the *set* of
             indices covered, not the summed width -- summing double-counts a ``constant``
             override of a range a ``range_mapping`` already covers, which ``exo_off`` does
             on ``[22,24]``. Overlap is therefore legal; an index claimed past ``nu`` or an
             actuator no mapping claims is not. Checking the covered set rather than just
             its maximum is what catches an interior gap, where the widest index still
             lands on ``nu`` but some actuator in the middle is left at zero.
        """
        import mujoco

        model = mujoco.MjModel.from_xml_string(model_xml)
        # Count muscles by actuator dynamics rather than as nu - na: a device's `general`
        # actuator may declare its own activation dynamics, which would inflate na.
        n_muscle = int((model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE).sum())
        n_motor = model.nu - n_muscle

        env_id = config.env_params.env_id
        composed = f"env_id={env_id!r}, msk_key={config.env_params.msk_key!r}, device_key={config.env_params.device_key!r}"

        if n_motor > 0 and env_id not in _EXO_ENV_IDS:
            raise ValueError(
                f"Composed model has {n_motor} motor actuator(s) but {composed} selects the "
                f"muscle-only policy, so nothing would drive them. Use an exo env id "
                f"({', '.join(_EXO_ENV_IDS)}) or a device that adds no motor actuators."
            )

        # Always present (config.py CustomPolicyParams), empty for a config that does not
        # use custom net indexing -- in which case there is no declared layout to check.
        net_indexing_info = config.policy_params.custom_policy_params.net_indexing_info
        claimed: set[int] = set()
        for net_info in net_indexing_info.values():
            for mapping in net_info.get("action", []):
                action_range = mapping.get("range_action")
                if action_range is not None:
                    claimed.update(range(action_range[0], action_range[1]))
        if not claimed:
            return

        model_desc = f"the composed model has nu={model.nu} ({n_muscle} muscle + {n_motor} motor)"
        beyond = sorted(i for i in claimed if i >= model.nu)
        if beyond:
            raise ValueError(
                f"Action layout mismatch: net_indexing_info claims actuator index "
                f"{beyond[0]}{f' (and {len(beyond) - 1} more)' if len(beyond) > 1 else ''} but "
                f"{model_desc}. {composed}. Narrow the config's range_action entries to the "
                f"model's actuators."
            )
        unclaimed = sorted(set(range(model.nu)) - claimed)
        if unclaimed:
            raise ValueError(
                f"Action layout mismatch: no net_indexing_info mapping drives actuator "
                f"index {unclaimed if len(unclaimed) <= 8 else f'{unclaimed[:8]} (+{len(unclaimed) - 8} more)'}, "
                f"so {'it would stay' if len(unclaimed) == 1 else 'they would stay'} at zero; "
                f"{model_desc}. {composed}. Extend the config's range_action entries to cover "
                f"every actuator."
            )

    @staticmethod
    def create_environment(config, is_rendering_on: bool, is_evaluate_mode: bool = False):

        ref_data_dict = EnvironmentHandler.load_reference_data(config)

        # Compose pipeline: when both msk_key and device_key are set, route the
        # {msk, device, terrain} triple through the shared env-spec front-door --
        # the same validated path the CO pipeline uses -- to compose the model into
        # an XML string, passed as model_path (myosuite's SimScene routes "<mujoco"
        # -> from_xml_string). A literal model_path is the escape hatch for a
        # pre-built MJCF. Composed once here; the XML string pickles fine into
        # SubprocVecEnv workers.
        msk_key = config.env_params.msk_key
        device_key = config.env_params.device_key
        model_path = config.env_params.model_path

        # Compose takes both keys or neither. Half a spec used to fall through to
        # model_path, which is None in every migrated config, and failed inside
        # gym.make without mentioning the key that was actually missing.
        if bool(msk_key) != bool(device_key):
            missing, present = ("device_key", f"msk_key={msk_key!r}") if msk_key else ("msk_key", f"device_key={device_key!r}")
            raise ValueError(
                f"Incomplete compose spec: {present} is set but {missing} is not. Set both to "
                f"compose a model (run `python -m assist_sim list` for the valid keys), or "
                f"neither and give an explicit model_path."
            )

        if msk_key and device_key:
            from myoassist_utils.env_spec import EnvSpec

            model_path = (
                EnvSpec(
                    msk=msk_key,
                    device=device_key,
                    terrain=config.env_params.terrain,
                )
                .validate()
                .compose()
            )
            EnvironmentHandler._validate_action_layout(model_path, config)
        elif not model_path:
            raise ValueError(
                "No model specified: set msk_key + device_key to compose a model, or "
                "model_path to load a pre-built MJCF. The shipped configs use the "
                "compose keys; a config carrying neither is a migration oversight."
            )

        # Base gym.make arguments
        gym_make_args = {
            "seed": config.env_params.seed,
            "model_path": model_path,
            "env_params": config.env_params,
            "is_evaluate_mode": is_evaluate_mode,
        }

        # Add reference_data only if it exists
        if ref_data_dict is not None:
            gym_make_args["reference_data"] = ref_data_dict

        try:
            if is_rendering_on or config.env_params.num_envs == 1:
                print(f"{config.env_params.env_id=}")
                env = gym.make(config.env_params.env_id, **gym_make_args).unwrapped
                if is_rendering_on:
                    env.mujoco_render_frames = True
                config.env_params.num_envs = 1
                config.ppo_params.n_steps = config.ppo_params.batch_size
            else:
                env = SubprocVecEnv(
                    [
                        lambda: (gym.make(config.env_params.env_id, **gym_make_args)).unwrapped
                        for _ in range(config.env_params.num_envs)
                    ]
                )
        except Exception as e:
            new_message = str(e)[:1000]
            e.args = (new_message,)
            raise e
        return env

    @staticmethod
    def load_reference_data(config):
        # Check if config has reference_data_path attribute
        print("===================================================================")
        if not hasattr(config.env_params, "reference_data_path"):
            print("No reference data path provided.")
            print("===================================================================")
            return None

        if not config.env_params.reference_data_path:
            print("No reference data path provided.")
            print("===================================================================")
            return None
        print(f"Loading reference data from {config.env_params.reference_data_path}")
        print("===================================================================")
        if config.env_params.reference_data_path.endswith(".npz"):
            ref_data_npz = np.load(config.env_params.reference_data_path, allow_pickle=True)
            ref_data_dict = {key: ref_data_npz[key].item() for key in ref_data_npz.files}
        elif config.env_params.reference_data_path.endswith(".json"):
            with open(config.env_params.reference_data_path, "r") as f:
                ref_data_dict = json.load(f)
        else:
            raise ValueError("Unsupported file format. Please use either .npz or .json.")

        if "resampled_series_data" not in ref_data_dict:
            ref_data_dict["resampled_series_data"] = {}
            for key in ref_data_dict["series_data"].keys():
                original_data_length = len(ref_data_dict["series_data"][key])
                original_sample_rate = ref_data_dict["metadata"]["sample_rate"]
                original_x = np.linspace(0, original_data_length - 1, original_data_length)

                new_sample_rate = config.env_params.control_framerate
                new_length = int(original_data_length * new_sample_rate / original_sample_rate)
                new_x = np.linspace(0, original_data_length - 1, new_length)
                ref_data_dict["series_data"][key] = np.interp(new_x, original_x, ref_data_dict["series_data"][key])
                ref_data_dict["metadata"]["resampled_data_length"] = new_length
                ref_data_dict["metadata"]["resampled_sample_rate"] = new_sample_rate

        return ref_data_dict

    def get_config_type_from_session_id(session_id):
        # from rl_train.envs import myo_leg_18_reward_per_step
        from rl_train.train.train_configs.config import TrainSessionConfigBase
        from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
        from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig

        # Create appropriate config based on env_id
        print(f"session_id: {session_id}")
        if session_id == "myoAssistLeg-v0":
            return TrainSessionConfigBase
        elif session_id in ["myoAssistLegImitation-v0"]:
            return ImitationTrainSessionConfig
        elif session_id == "myoAssistLegImitationExo-v0":
            return ExoImitationTrainSessionConfig
        raise ValueError(f"Invalid session id: {session_id}")

    @staticmethod
    def get_session_config_from_path(config_path, class_type):
        print(f"Loading config from {config_path}")
        config_file_path = config_path
        with open(config_file_path, "r") as f:
            config_dict = json.load(f)
            session_config = DictionableDataclass.create(class_type, config_dict)
        return session_config

    @staticmethod
    def get_callback(config, train_log_handler):
        from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig

        from rl_train.envs import myoassist_leg_imitation
        from rl_train.utils import learning_callback

        if isinstance(config, ImitationTrainSessionConfig):
            custom_callback = myoassist_leg_imitation.ImitationCustomLearningCallback(
                log_rollout_freq=config.logger_params.logging_frequency,
                evaluate_freq=config.logger_params.evaluate_frequency,
                log_handler=train_log_handler,
                original_reward_weights=config.env_params.reward_keys_and_weights,
                auto_reward_adjust_params=config.auto_reward_adjust_params,
            )
        else:
            custom_callback = learning_callback.BaseCustomLearningCallback(
                log_rollout_freq=config.logger_params.logging_frequency,
                evaluate_freq=config.logger_params.evaluate_frequency,
                log_handler=train_log_handler,
            )

        return custom_callback

    @staticmethod
    def get_stable_baselines3_model(config: TrainSessionConfigBase, env, trained_model_path: str | None = None):
        import stable_baselines3
        from rl_train.train.policies.rl_agent_human import HumanActorCriticPolicy
        from rl_train.train.policies.rl_agent_exo import HumanExoActorCriticPolicy

        if config.env_params.env_id in _EXO_ENV_IDS:
            policy_class = HumanExoActorCriticPolicy
            print("Using HumanExoActorCriticPolicy")
        else:
            policy_class = HumanActorCriticPolicy
            print("Using HumanActorCriticPolicy")
        if trained_model_path is not None:
            print(f"Loading trained model from {trained_model_path}")
            model = stable_baselines3.PPO.load(
                trained_model_path,
                env=env,
                custom_objects={"policy_class": policy_class},
            )
        elif config.env_params.prev_trained_policy_path:
            print(f"Loading previous trained policy from {config.env_params.prev_trained_policy_path}")
            # when should I reset the (value)network?
            model = stable_baselines3.PPO.load(
                config.env_params.prev_trained_policy_path,
                env=env,
                custom_objects={"policy_class": policy_class},
                # policy_kwargs=DictionableDataclass.to_dict(config.policy_params),
                verbose=2,
                **DictionableDataclass.to_dict(config.ppo_params),
            )
            # print(f"Resetting network: {config.custom_policy_params.reset_shared_net_after_load=}, {config.custom_policy_params.reset_policy_net_after_load=}, {config.custom_policy_params.reset_value_net_after_load=}")
            model.policy.reset_network(
                reset_shared_net=config.policy_params.custom_policy_params.reset_shared_net_after_load,
                reset_policy_net=config.policy_params.custom_policy_params.reset_policy_net_after_load,
                reset_value_net=config.policy_params.custom_policy_params.reset_value_net_after_load,
            )
        else:
            ppo_kwargs = DictionableDataclass.to_dict(config.ppo_params)
            mirror_coef = ppo_kwargs.pop("mirror_coef", 0.0)
            if mirror_coef > 0:
                from rl_train.train.mirror_ppo import MirrorPPO

                obs_perm, act_perm, n_muscle = EnvironmentHandler._mirror_permutations(config)
                model = MirrorPPO(
                    policy=policy_class,
                    env=env,
                    policy_kwargs=DictionableDataclass.to_dict(config.policy_params),
                    verbose=2,
                    mirror_coef=mirror_coef,
                    obs_perm=obs_perm,
                    act_perm=act_perm,
                    n_muscle_actuators=n_muscle,
                    **ppo_kwargs,
                )
            else:
                model = stable_baselines3.PPO(
                    policy=policy_class,
                    env=env,
                    policy_kwargs=DictionableDataclass.to_dict(config.policy_params),
                    verbose=2,
                    **ppo_kwargs,
                )
        return model

    @staticmethod
    def _mirror_permutations(config):
        """Left/right mirror maps for the observation and action vectors, and the muscle count.

        Derived from the composed model's actuator names plus the config's own observation
        keys, so a config that changes either cannot end up with a stale map. The muscle count
        comes back too because the action vector is muscles followed by device actuators, and
        the split is what lets the penalty report how much of itself reaches the device.
        """
        import mujoco

        from myoassist_utils.compose import compose_env_model
        from rl_train.train.policies.mirror import action_permutation, observation_permutation

        model = mujoco.MjModel.from_xml_string(
            compose_env_model(
                config.env_params.msk_key,
                config.env_params.device_key,
                terrain=config.env_params.terrain,
            )
        )
        n_muscle = int((model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE).sum())
        names = [model.actuator(i).name for i in range(model.nu)]
        act_perm = action_permutation(names)
        # A device actuator that mirrors to itself contributes nothing to the mirror penalty, and
        # a self-map passes the involution check, so an unrecognised side-naming convention fails
        # silently on exactly the pair the penalty exists to constrain. UTAnkleExo_L2's `_dx`/`_sx`
        # did this before the rule was extended. One self-mapping device actuator is legitimate --
        # the unilateral prostheses have a single one -- so only flag a set of two or more where
        # none found a partner.
        device_slots = list(range(n_muscle, model.nu))
        if len(device_slots) >= 2 and all(int(act_perm[i]) == i for i in device_slots):
            raise ValueError(
                f"Mirror map leaves every device actuator mapped to itself, so the mirror penalty "
                f"cannot constrain them: {[names[i] for i in device_slots]}. Their side naming is "
                f"not recognised by rl_train/train/policies/mirror.py:_swap_name -- add the "
                f"convention there."
            )
        return observation_permutation(config.env_params, n_muscle, names), act_perm, n_muscle

    @staticmethod
    def updateconfig_from_model_policy(config, model):
        pass
        # config.policy_info.extractor_policy_net = f"{model.policy.mlp_extractor.policy_net}"
        # config.policy_info.extractor_value_net = f"{model.policy.mlp_extractor.value_net}"
        # config.policy_info.action_net = f"{model.policy.action_net}"
        # config.policy_info.value_net = f"{model.policy.value_net}"
        # config.policy_info.ortho_init = f"{model.policy.ortho_init}"
        # config.policy_info.share_features_extractor = f"{model.policy.share_features_extractor}"
