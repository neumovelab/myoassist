import collections
import numpy as np
from dataclasses import dataclass, field
from myoassist_rl.envs.myoassist_leg_imitation import MyoAssistLegImitation
from myoassist_rl.envs.myoassist_leg_imitation import ImitationTrainSessionConfig
from myoassist_rl.envs.myoassist_leg_imitation import ImitationCustomLearningCallback


from myoassist_rl.rl_train.utils.config import TrainSessionConfigBase
from myoassist_rl.rl_train.utils.data_types import DictionableDataclass
from myoassist_rl.rl_train.utils.handlers import train_log_handler
from stable_baselines3.common.vec_env import SubprocVecEnv
from myoassist_rl.rl_train.utils.learning_callback import BaseCustomLearningCallback
################################################################
@dataclass
class ExoImitationTrainSessionConfig(ImitationTrainSessionConfig):
    @dataclass
    class CustomPolicyParams(ImitationTrainSessionConfig.PolicyParams.CustomPolicyParams):
        human_observation_indices: list[int] = field(default_factory=list[int])
        exo_observation_indices: list[int] = field(default_factory=list[int])
        human_action_size: int = 0
        exo_action_size: int = 0
    custom_policy_params: CustomPolicyParams = field(default_factory=CustomPolicyParams)
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
        @dataclass
        class CustomPolicyParams(ImitationTrainSessionConfig.PolicyParams.CustomPolicyParams):
            human_observation_indices: list[int] = field(default_factory=list[int])
            exo_observation_indices: list[int] = field(default_factory=list[int])
            human_action_size: int = 0
            exo_action_size: int = 0
        custom_policy_params: CustomPolicyParams = field(default_factory=CustomPolicyParams)
    policy_params: PolicyParams = field(default_factory=PolicyParams)
##############################################################################



class MyoAssistLegImitationExo(MyoAssistLegImitation):
    # override
    def step(self, a, **kwargs):
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        
        return (next_obs, reward, terminated, truncated, info)
        