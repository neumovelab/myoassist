import torch as th
from gymnasium import spaces
class NetworkIndexHandler:
    def __init__(self, net_indexing_info: dict,
                        observation_space: spaces.Space,
                        action_space: spaces.Space,):
        self.net_indexing_info = net_indexing_info
        self.observation_space = observation_space
        self.action_space = action_space
    def get_observation_num(self, net_name: str):
        human_observation_num = 0
        for net_indexing_info in self.net_indexing_info[net_name]["observation"]:
            if net_indexing_info["type"] == "range":
                start_inclusive, end_exclusive = net_indexing_info["range"]
                human_observation_num += end_exclusive - start_inclusive
            elif net_indexing_info["type"] == "index":# specifying index of observation
                human_observation_num += len(net_indexing_info["index"])
        return human_observation_num
    def get_action_num(self, net_name: str):
        human_action_num = 0
        for net_indexing_info in self.net_indexing_info[net_name]["action"]:
            if net_indexing_info["type"] == "range_mapping":
                start_inclusive_net, end_exclusive_net = net_indexing_info["range_net"]
                start_inclusive_action, end_exclusive_action = net_indexing_info["range_action"]
                human_action_num += end_exclusive_action - start_inclusive_action
            elif net_indexing_info["type"] == "index_mapping":# specifying index of action
                human_action_num += len(net_indexing_info["index"])
        return human_action_num
    
    def map_observation_to_network(self, observation: th.Tensor, net_name: str):
        observation_num = self.get_observation_num(net_name)
        # Create result tensor on the same device as observation
        result = th.zeros(observation.shape[0], observation_num, device=observation.device)
        current_index = 0
        for net_indexing_info in self.net_indexing_info[net_name]["observation"]:
            if net_indexing_info["type"] == "range":
                start_inclusive, end_exclusive = net_indexing_info["range"]
                result[:,current_index:current_index + end_exclusive - start_inclusive] = observation[:,start_inclusive:end_exclusive]
                current_index += end_exclusive - start_inclusive
            elif net_indexing_info["type"] == "index":# specifying index of observation
                result[:,current_index] = observation[:,current_index]
                current_index += 1
        if observation.shape[1] != observation_num:
            raise ValueError(f"Observation length {observation.shape[1]} does not match expected length {observation_num}")
        return result
    def map_network_to_action(self, network_output_dict: dict[str, th.Tensor]):
        action_num = 0
        batch_size = 0
        device = None
        # print(f'{network_output_dict=}')
        # First pass: validate all outputs are on the same device
        for network_name, network_output in network_output_dict.items():
            if device is None:
                device = network_output.device
            elif device != network_output.device:
                raise ValueError(f"All network outputs must be on the same device. Found {device} and {network_output.device}")
            
            action_num += self.get_action_num(network_name)
            if batch_size == 0:
                batch_size = network_output.shape[0]
            else:
                if batch_size != network_output.shape[0]:
                    raise ValueError(f"Batch size {network_output.shape[0]} does not match expected batch size {batch_size}")
                    
        # Create result tensor on the same device as network outputs
        result = th.zeros(batch_size, self.action_space.shape[0], device=device)
        
        for network_name, network_output in network_output_dict.items():
            for net_indexing_info in self.net_indexing_info[network_name]["action"]:
                if net_indexing_info["type"] == "range_mapping":
                    start_inclusive_net, end_exclusive_net = net_indexing_info["range_net"]
                    start_inclusive_action, end_exclusive_action = net_indexing_info["range_action"]
                    result[:,start_inclusive_action:end_exclusive_action] = network_output[:,start_inclusive_net:end_exclusive_net]
                elif net_indexing_info["type"] == "index_mapping":
                    result[:,net_indexing_info["index"]] = network_output[:,net_indexing_info["index"]]
        return result