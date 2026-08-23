from dataclasses import dataclass, field


@dataclass
class TrainSessionConfigBase:
    total_timesteps: int = 1000

    @dataclass
    class LoggerParams:
        logging_frequency: int = int(1)
        evaluate_frequency: int = int(64)

    logger_params: LoggerParams = field(default_factory=LoggerParams)

    @dataclass
    class EnvParams:
        @dataclass
        class RewardWeights:
            forward_reward: float = 0.01
            muscle_activation_penalty: float = 0.1
            # Price of device effort, in the same units as muscle_activation_penalty: both terms
            # are dt times a mean dimensionless effort. Because the muscle mean is taken over 22
            # actuators and the device mean over 2, the per-actuator price is
            # muscle_activation_penalty/22 against exo_activation_penalty/2 -- so at the shipped
            # muscle weight of 10, an exo weight of 0.1 makes one device actuator's effort about
            # ten times cheaper than one muscle's. Default 0 keeps existing configs unchanged.
            exo_activation_penalty: float = 0.0
            muscle_activation_diff_penalty: float = 0.1

            # for reward per step
            footstep_delta_time: float = 0.0
            average_velocity_per_step: float = 0.0
            muscle_activation_penalty_per_step: float = 0.0

            joint_constraint_force_penalty: float = 0.0

            foot_force_penalty: float = 0.0

        reward_keys_and_weights: RewardWeights = field(default_factory=RewardWeights)

        env_id: str = ""
        num_envs: int = 1
        seed: int = 0
        safe_height: float = 0.65
        control_framerate: int = 30
        physics_sim_framerate: int = 1200

        min_target_velocity: float = 0.5
        max_target_velocity: float = 3.0

        # Speed curriculum. 0 disables it, which is what every existing config does. Set it and
        # the target velocity ramps linearly from this value up to min/max_target_velocity over
        # `curriculum_fraction` of the run, then holds. The amputee models only ever produced
        # sustained stepping at 0.2-0.35 m/s while being asked for 1.25 from the first step.
        curriculum_start_velocity: float = 0.0
        curriculum_fraction: float = 0.5

        # Reward annealing. Maps a reward key to [start_scale, end_scale]; the key's configured
        # weight is multiplied by a scale that moves linearly between them across
        # [reward_curriculum_start, reward_curriculum_end] of the run, held flat outside. Empty
        # disables it, which is what every existing config does.
        #
        # The intended use is to let imitation bootstrap and then get out of the way. On an
        # amputee the reference cannot describe the affected side, so imitation is a scaffold
        # rather than the objective: it teaches a posture to stand in, and what remains once it
        # is gone is forward progress against effort. Measured on the 30M policies, the effort
        # term already prices one-legged hopping at 0.237 per metre against 0.110-0.131 for the
        # policies that use the prosthesis, so that terminal objective can tell them apart --
        # provided forward does not outweigh effort, which at 20 against 10 it did.
        reward_curriculum: dict = field(default_factory=dict)
        reward_curriculum_start: float = 0.2
        reward_curriculum_end: float = 0.6
        min_target_velocity_period: float = 3
        max_target_velocity_period: float = 5

        custom_max_episode_steps: int = 500
        model_path: str = None
        prev_trained_policy_path: str = None
        reference_data_path: str = ""

        enable_lumbar_joint: bool = False
        lumbar_joint_fixed_angle: float = 0.0
        lumbar_joint_damping_value: float = 0.05

        # Geom groups to hide from rendering. Which group holds clutter and which holds hardware
        # is an authoring convention of whoever built the model, not something the environment can
        # derive, so it belongs here rather than in code. This replaces an unconditional "hide
        # group 1" whose comment said it removed the musculoskeletal skin; myolegs22 and myolegs26
        # put no geom in group 1, so its only remaining effect was to hide STRIDE_L2's entire
        # six-bar linkage -- 14 geoms -- leaving just the shoe visible. Rendering only: alpha does
        # not enter contact, mass or constraint computation.
        hidden_geom_groups: list[int] = field(default_factory=list)

        observation_joint_pos_keys: list[str] = field(default_factory=list)
        observation_joint_vel_keys: list[str] = field(default_factory=list)
        observation_joint_sensor_keys: list[str] = field(default_factory=list)
        # Joint-limit (constraint-force) sensor names used by the constraint-force
        # penalty. Empty -> MyoAssistLegBase.JOINT_LIMIT_SENSOR_NAMES default.
        joint_limit_sensor_keys: list[str] = field(default_factory=list)

        # A10 compose pipeline. When both msk_key and device_key are set, the
        # model is composed via myoassist_utils.compose.compose_env_model(...)
        # (human MSK + assistive device + terrain) and the resulting XML string is
        # used as the model. Leave them None to fall back to the literal
        # model_path above (escape hatch). terrain is a path to a
        # myoassist_terrains JSON config, or None for a flat default ground.
        msk_key: str = None
        device_key: str = None
        terrain: str = None

        # Fraction of its own ctrlrange the policy may command on the device actuators, applied
        # by narrowing `actuator_ctrlrange` at setup. 1.0 is the model's full authority and is
        # what every intact config uses. Below 1.0 for a device whose joint cannot absorb its own
        # actuator: see MyoAssistLegBase._setup.
        device_ctrl_scale: float = 1.0

    env_params: EnvParams = field(default_factory=EnvParams)

    """
    used in TrainAnalyzer
        total_timesteps: int = 300
        min_target_velocity: float = 1.25
        max_target_velocity: float = 1.25
        target_velocity_period: float = 3
        velocity_mode: str = "SINUSOIDAL"
        cam_type: str = "follow"
        cam_distance: float = 2.5
        visualize_activation: bool = True
    """
    evaluate_param_list: list[dict] = field(default_factory=list[dict])

    @dataclass
    class PolicyParams:
        """
        ActorCriticPolicy parameters:
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
            activation_fn: type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[dict[str, Any]] = None,
        """

        # @dataclass
        # class CustomPolicyParams:
        #     reset_shared_net: bool = False
        #     reset_policy_net: bool = False
        #     reset_value_net: bool = False
        # custom_policy_params: CustomPolicyParams = field(default_factory=CustomPolicyParams)
        @dataclass
        class CustomPolicyParams:
            # For curriculum learning
            reset_shared_net_after_load: bool = False
            reset_policy_net_after_load: bool = False
            reset_value_net_after_load: bool = False
            # reset_log_std_after_load: bool = False

            net_arch: dict = field(default_factory=dict)
            log_std_init: float = field(default=-2.0)

            net_indexing_info: dict = field(default_factory=dict)

        custom_policy_params: CustomPolicyParams = field(default_factory=CustomPolicyParams)

    policy_params: PolicyParams = field(default_factory=PolicyParams)

    @dataclass
    class PPOParams:
        learning_rate: float = 3e-4
        n_steps: int = 4096
        batch_size: int = 2048
        n_epochs: int = 10
        gamma: float = 0.99
        gae_lambda: float = 0.95
        clip_range: float = 0.2
        clip_range_vf: float = 0.2
        ent_coef: float = 0.01
        vf_coef: float = 0.5
        max_grad_norm: float = 0.5
        use_sde: bool = False
        sde_sample_freq: int = -1
        target_kl: float = None
        device: str = "cpu"
        # Weight on the left/right mirror-symmetry penalty (see rl_train/train/mirror_ppo.py).
        # 0 disables it, and PPO then behaves exactly as before.
        mirror_coef: float = 0.0

    ppo_params: PPOParams = field(default_factory=PPOParams)
