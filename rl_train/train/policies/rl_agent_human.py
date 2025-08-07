from rl_train.train.policies.rl_agent_base import BasePPOCustomNetwork, BaseCustomActorCriticPolicy
import torch as th
from torch import nn
from gymnasium import spaces
import rl_train.train.train_configs.config as myoassist_config
import rl_train.train.train_configs.config_imitation as myoassist_config_imitation
class HumanPPOCustomNetwork(BasePPOCustomNetwork):

    def forward_actor(self, obs: th.Tensor) -> th.Tensor:
        return self.policy_net(obs)

    def forward_critic(self, obs: th.Tensor) -> th.Tensor:
        return self.value_net(obs)
    
    def reset_policy_networks(self):
        """Reset both networks to their initial state"""
        layers = []
        last_dim = self.observation_space.shape[0]
        
        for dim in self.net_arch["human_actor"]:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.Tanh())
            last_dim = dim
            

        layers.append(nn.Linear(last_dim, self.action_space.shape[0]))
        layers.append(nn.Tanh())
        self.policy_net = nn.Sequential(*layers)
    def reset_value_network(self):
        value_layers = []
        value_last_dim = self.observation_space.shape[0]
        
        for dim in self.net_arch["common_critic"]:
            value_layers.append(nn.Linear(value_last_dim, dim))
            value_layers.append(nn.Tanh())
            value_last_dim = dim
            
        value_layers.append(nn.Linear(value_last_dim, 1))
        
        self.value_net = nn.Sequential(*value_layers)

class HumanActorCriticPolicy(BaseCustomActorCriticPolicy):
    def _get_custom_policy_type(self):
        return myoassist_config_imitation.ImitationTrainSessionConfig.PolicyParams.CustomPolicyParams
    def _build_policy_network(self, observation_space: spaces.Space,
                              action_space: spaces.Space,
                              custom_policy_params: myoassist_config.TrainSessionConfigBase.PolicyParams.CustomPolicyParams) -> BasePPOCustomNetwork:
        return HumanPPOCustomNetwork(observation_space,
                                                    action_space,
                                                    custom_policy_params)
    # Custom reset network
    def reset_network(self, reset_shared_net: bool = False, reset_policy_net: bool = False, reset_value_net: bool = False):
        """Reset the networks if specified"""
        if reset_policy_net:
            self.policy_network.reset_policy_networks()
        if reset_value_net:
            self.policy_network.reset_value_network()
        

