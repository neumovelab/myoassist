import torch
import torch.nn as nn
import torch.nn.init as init

from abc import ABC, abstractmethod
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
from typing import Any, ClassVar, Optional, TypeVar, Union
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict

from gymnasium import spaces
import torch as th
from torch import nn
from typing import Callable, Tuple

import gymnasium as gym

from myoassist_rl.rl_train.utils.data_types import DictionableDataclass
import myoassist_rl.rl_train.utils.config as myoassist_config
import myoassist_rl.envs.myoassist_leg_imitation_exo as myoassist_env
from myoassist_rl.rl_train.rl_agents.network_index_handler import NetworkIndexHandler
import torch
torch.autograd.set_detect_anomaly(True)

from myoassist_rl.rl_train.rl_agents.rl_agent_base import BasePPOCustomNetwork, BaseCustomActorCriticPolicy

class CustomNetworkHumanExo(BasePPOCustomNetwork):
    def forward_actor(self, obs: th.Tensor) -> th.Tensor:
        human_obs = self.network_index_handler.map_observation_to_network(obs, "human_actor")
        exo_obs = self.network_index_handler.map_observation_to_network(obs, "exo_actor")
        network_output_dict = {"human_actor": self.human_policy_net(human_obs), "exo_actor": self.exo_policy_net(exo_obs)}
        return self.network_index_handler.map_network_to_action(network_output_dict)

    def forward_critic(self, obs: th.Tensor) -> th.Tensor:
        value_obs = self.network_index_handler.map_observation_to_network(obs, "common_critic")
        return self.value_net(value_obs)
    
    def reset_policy_networks(self):
        self.network_index_handler = NetworkIndexHandler(self.net_indexing_info, self.observation_space, self.action_space)
        layers = []

        last_dim = self.network_index_handler.get_observation_num("human_actor")
        for dim in self.net_arch["human_actor"]:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.Tanh())
            last_dim = dim

        layers.append(nn.Linear(last_dim, self.network_index_handler.get_action_num("human_actor")))
        layers.append(nn.Tanh())
        self.human_policy_net = nn.Sequential(*layers)


        layers = []
        last_dim = self.network_index_handler.get_observation_num("exo_actor")
        
        for dim in self.net_arch["exo_actor"]:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.Tanh())
            last_dim = dim
            

        layers.append(nn.Linear(last_dim, self.network_index_handler.get_action_num("exo_actor")))
        layers.append(nn.Tanh())
        self.exo_policy_net = nn.Sequential(*layers)
    def reset_value_network(self):
        value_layers = []
        value_last_dim = self.network_index_handler.get_observation_num("common_critic")
        
        for dim in self.net_arch["common_critic"]:
            value_layers.append(nn.Linear(value_last_dim, dim))
            value_layers.append(nn.Tanh())
            value_last_dim = dim
            
        value_layers.append(nn.Linear(value_last_dim, 1))
        
        self.value_net = nn.Sequential(*value_layers)

class HumanExoActorCriticPolicy(BaseCustomActorCriticPolicy):
    def _build_policy_network(self, observation_space: spaces.Space,
                              action_space: spaces.Space,
                              custom_policy_params: myoassist_config.TrainSessionConfigBase.PolicyParams.CustomPolicyParams) -> BasePPOCustomNetwork:
        return CustomNetworkHumanExo(observation_space,
                                    action_space,
                                    custom_policy_params)
    def reset_network(self, reset_shared_net: bool = False, reset_policy_net: bool = False, reset_value_net: bool = False):
        """Reset the networks if specified"""
        if reset_shared_net or reset_policy_net or reset_value_net:
            self.policy_network.reset_networks()

