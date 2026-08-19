import torch
import torch.nn as nn


from gymnasium import spaces
import torch as th


import rl_train.train.train_configs.config as myoassist_config
from rl_train.train.policies.network_index_handler import NetworkIndexHandler
from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
from rl_train.train.policies.rl_agent_base import BasePPOCustomNetwork, BaseCustomActorCriticPolicy

# Runtime autograd flag, read during backward -- unaffected by import order.
torch.autograd.set_detect_anomaly(True)

# Right first, so "the right leg's own ordering" is the one the shared network is sized from.
_SHARED_SIDE_EXO_NETS = ("exo_actor_r", "exo_actor_l")


class CustomNetworkHumanExo(BasePPOCustomNetwork):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        custom_policy_params: ExoImitationTrainSessionConfig.PolicyParams.CustomPolicyParams,
    ):
        super().__init__(observation_space, action_space, custom_policy_params)

    def forward_actor(self, obs: th.Tensor) -> th.Tensor:
        network_output_dict = {
            name: net(self.network_index_handler.map_observation_to_network(obs, name)) for name, net in self.actor_nets.items()
        }
        return self.network_index_handler.map_network_to_action(network_output_dict)

    def forward_critic(self, obs: th.Tensor) -> th.Tensor:
        value_obs = self.network_index_handler.map_observation_to_network(obs, "common_critic")
        return self.value_net(value_obs)

    def _build_actor_mlp(self, net_name: str) -> nn.Sequential:
        """Tanh MLP sized from the config's index mappings for `net_name`."""
        layers = []
        last_dim = self.network_index_handler.get_observation_num(net_name)
        for dim in self.net_arch[net_name]:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.Tanh())
            last_dim = dim
        layers.append(nn.Linear(last_dim, self.network_index_handler.get_action_num(net_name)))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def _uses_shared_side_exo(self) -> bool:
        """True when the config asks for one exo network applied to each leg in turn.

        A single `exo_actor` reading both legs and emitting both exo commands is free to
        ignore the left/right structure, and measurably does: in the best 30M Tutorial run
        the two exo commands correlate near zero lag in absolute time, so one command lands
        at push-off on one leg and at early stance on the other, half a cycle away.

        Declaring `exo_actor_r` and `exo_actor_l` instead builds *one* network and applies it
        to each leg with that leg's own inputs first, so `Exo_L(s) == Exo_R(mirror(s))` holds
        by construction rather than being asked for by a penalty that reward can outbid. This
        is the `NET` arm of Abdolhosseini et al. 2019, *On Learning Symmetric Locomotion*.
        """
        present = [name in self.net_indexing_info for name in _SHARED_SIDE_EXO_NETS]
        assert len(set(present)) == 1, (
            f"declare both of {_SHARED_SIDE_EXO_NETS} or neither; got "
            f"{[n for n, p in zip(_SHARED_SIDE_EXO_NETS, present) if p]}"
        )
        if present[0]:
            assert "exo_actor" not in self.net_indexing_info, (
                "config declares both the single exo_actor and the per-side exo actors; "
                "the exo action slots would be written twice"
            )
        return present[0]

    def reset_policy_networks(self):
        self.network_index_handler = NetworkIndexHandler(self.net_indexing_info, self.observation_space, self.action_space)
        self.human_policy_net = self._build_actor_mlp("human_actor")
        actor_nets = {"human_actor": self.human_policy_net}

        if self._uses_shared_side_exo():
            right, left = _SHARED_SIDE_EXO_NETS
            # Necessary for the two sides to be each other's mirror: same inputs, reordered.
            # The full check -- that the left order really is the mirror permutation of the
            # right -- needs the composed model's names and lives in the config generator.
            index_sets = {
                name: sorted(i for block in self.net_indexing_info[name]["observation"] for i in block["index"])
                for name in _SHARED_SIDE_EXO_NETS
            }
            assert index_sets[right] == index_sets[left], (
                f"per-side exo actors must read the same observations in mirrored order; "
                f"right reads {index_sets[right]}, left reads {index_sets[left]}"
            )
            assert self.network_index_handler.get_action_num(right) == self.network_index_handler.get_action_num(left), (
                "per-side exo actors must emit the same number of commands"
            )
            if left in self.net_arch:
                assert self.net_arch[left] == self.net_arch[right], (
                    f"per-side exo actors share weights, so they cannot have different widths: "
                    f"{self.net_arch[right]} vs {self.net_arch[left]}"
                )
            # One module object under both names: `parameters()` deduplicates by identity, so
            # the optimizer sees a single copy and both legs are driven by the same weights.
            shared_exo_net = self._build_actor_mlp(right)
            self.exo_policy_net_right = shared_exo_net
            self.exo_policy_net_left = shared_exo_net
            actor_nets[right] = shared_exo_net
            actor_nets[left] = shared_exo_net
        else:
            self.exo_policy_net = self._build_actor_mlp("exo_actor")
            actor_nets["exo_actor"] = self.exo_policy_net

        self.actor_nets = actor_nets

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
    def _get_custom_policy_type(self):
        return ExoImitationTrainSessionConfig.PolicyParams.CustomPolicyParams

    def _build_policy_network(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        custom_policy_params: myoassist_config.TrainSessionConfigBase.PolicyParams.CustomPolicyParams,
    ) -> BasePPOCustomNetwork:
        return CustomNetworkHumanExo(observation_space, action_space, custom_policy_params)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """Forward pass that pins the action slots the config declares constant."""
        mean_actions = self.policy_network.forward_actor(obs)
        value = self.policy_network.forward_critic(obs)

        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)

        actions = distribution.get_actions(deterministic=deterministic)
        actions = self.policy_network.network_index_handler.mask_default_value(actions)

        log_prob = distribution.log_prob(actions)
        return actions, value, log_prob

    def reset_network(self, reset_shared_net: bool = False, reset_policy_net: bool = False, reset_value_net: bool = False):
        """Reset the networks if specified"""
        if reset_policy_net:
            print("Resetting policy network")
            self.policy_network.reset_policy_networks()
        if reset_value_net:
            print("Resetting value network")
            self.policy_network.reset_value_network()
