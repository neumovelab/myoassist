import torch
import torch.nn as nn
import torch.nn.init as init

from abc import ABC, abstractmethod
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
from typing import Any, ClassVar, Optional, TypeVar, Union
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict

from gymnasium import spaces
import torch as th
from torch import nn
from typing import Callable, Tuple

import gymnasium as gym
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from myoassist_rl.rl_train.utils.data_types import DictionableDataclass
import myoassist_rl.rl_train.utils.config as myoassist_config

import torch
torch.autograd.set_detect_anomaly(True)
################################## set device ##################################
# set device to cpu or cuda
# device = torch.device('cpu')
# if(torch.cuda.is_available()): 
#     device = torch.device('cuda:0') 
#     torch.cuda.empty_cache()
#     print("Device: " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device: cpu")

class BasePPOCustomNetwork(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        custom_policy_params: myoassist_config.TrainSessionConfigBase.PolicyParams.CustomPolicyParams,
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space

        self.net_arch = custom_policy_params.net_arch
        self.net_indexing_info = custom_policy_params.net_indexing_info
        
        self.reset_policy_networks()
        self.reset_value_network()

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Forward pass through both networks
        :return: (action_mean, value)
        """
        return self.forward_actor(obs), self.forward_critic(obs)
    def forward_actor(self, obs: th.Tensor) -> th.Tensor:
        pass

    def forward_critic(self, obs: th.Tensor) -> th.Tensor:
        pass

    def reset_policy_networks(self):
        pass

    def reset_value_network(self):
        pass


class BaseCustomActorCriticPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        use_sde: bool = False,# Should be here for PPO (without it, it will throw an error)
        *args,
        **kwargs,
    ):
        # Remove custom_policy_params from kwargs
        custom_policy_params_dict = kwargs.pop('custom_policy_params', None)
        custom_policy_params = DictionableDataclass.create(myoassist_config.TrainSessionConfigBase.PolicyParams.CustomPolicyParams,
                                                           custom_policy_params_dict)
        
        # Initialize base class
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        print(f"{custom_policy_params=}")
        self.policy_network = self._build_policy_network(observation_space,
                                                    action_space,
                                                    custom_policy_params)
        
        self.action_dist = DiagGaussianDistribution(action_space)

        self.log_std = nn.Parameter(th.ones(self.action_space.shape[0], device=self.device) * custom_policy_params.log_std_init, requires_grad=True)
        
        print(f"{self.parameters()=}")
        self.apply(self.init_weights)
        
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
    def _build_policy_network(self, observation_space: spaces.Space,
                              action_space: spaces.Space,
                              custom_policy_params: myoassist_config.TrainSessionConfigBase.PolicyParams.CustomPolicyParams) -> BasePPOCustomNetwork:
        raise NotImplementedError("_build_policy_network should return a BasePPOCustomNetwork or it's subclass")
    # def _build_mlp_extractor(self) -> None:
    #     """This is called by the parent class but we don't need it"""
    #     pass

    # def _build(self, lr_schedule: Schedule) -> None:
    #     """This is called by the parent class but we don't need it"""
    #     pass

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        """
        # Get policy and value outputs
        mean_actions = self.policy_network.forward_actor(obs)
        value = self.policy_network.forward_critic(obs)
        
        # Create distribution and get actions
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return actions, value, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy
        """
        # Get policy and value outputs
        # Call forward once to get both mean_actions and value
        mean_actions, value = self.policy_network(obs)
        # This is more efficient than calling forward_actor and forward_critic separately
        
        # Create distribution and evaluate actions
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return value, log_prob, entropy
    

    # Override get_distribution and predict_values methods
    def get_distribution(self, obs: th.Tensor) -> DiagGaussianDistribution:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :return: the action distribution.
        """
        mean_actions = self.policy_network.forward_actor(obs)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        return self.policy_network.forward_critic(obs)
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)
    
