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
            muscle_activation_diff_penalty: float = 0.1

            # for reward per step
            footstep_delta_time:float = 0.0
            average_velocity_per_step:float = 0.0
            muscle_activation_penalty_per_step:float = 0.0

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
        min_target_velocity_period: float = 3
        max_target_velocity_period: float = 5

        custom_max_episode_steps: int = 500
        model_path: str = None
        prev_trained_policy_path: str = None
        reference_data_path: str = ""

        enable_lumbar_joint: bool = False
        lumbar_joint_fixed_angle: float = 0.0
        lumbar_joint_damping_value: float = 0.05

        observation_joint_pos_keys: list[str] = field(default_factory=list)
        observation_joint_vel_keys: list[str] = field(default_factory=list)
        observation_joint_sensor_keys: list[str] = field(default_factory=list)

        # terrain type: flat, random, sinusoidal, harmonic_sinusoidal, uphill, downhill, dev
        terrain_type: str = "flat"
        terrain_params: str = ""
        
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
        '''
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
        '''
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
    ppo_params: PPOParams = field(default_factory=PPOParams)

    
