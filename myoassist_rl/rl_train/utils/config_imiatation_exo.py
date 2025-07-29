
from myoassist_rl.rl_train.utils.config_imitation import ImitationTrainSessionConfig
from dataclasses import dataclass, field

@dataclass
class ExoImitationTrainSessionConfig(ImitationTrainSessionConfig):

    @dataclass
    class PolicyParams(ImitationTrainSessionConfig.PolicyParams):
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
        
        # This actually does nothing
        @dataclass
        class CustomPolicyParams(ImitationTrainSessionConfig.PolicyParams.CustomPolicyParams):
            human_observation_indices: list[int] = field(default_factory=list[int])
            exo_observation_indices: list[int] = field(default_factory=list[int])
            human_action_size: int = 0
            exo_action_size: int = 0
        custom_policy_params: CustomPolicyParams = field(default_factory=CustomPolicyParams)
    policy_params: PolicyParams = field(default_factory=PolicyParams)
    # @dataclass
    # class EnvParams(ImitationTrainSessionConfig.EnvParams):
    #     pass
    # env_params: EnvParams = field(default_factory=EnvParams)
##############################################################################
