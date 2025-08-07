from rl_train.train.train_configs.config import TrainSessionConfigBase
from dataclasses import dataclass, field
@dataclass
class ImitationTrainSessionConfig(TrainSessionConfigBase):
    @dataclass
    class AutoRewardAdjustParams:
        learning_rate: float = 0.001
    auto_reward_adjust_params: AutoRewardAdjustParams = field(default_factory=AutoRewardAdjustParams)

    @dataclass
    class EnvParams(TrainSessionConfigBase.EnvParams):
        @dataclass
        class RewardWeights(TrainSessionConfigBase.EnvParams.RewardWeights):
            qpos_imitation_rewards:dict = field(default_factory=dict)
            qvel_imitation_rewards:dict = field(default_factory=dict)
            
            end_effector_imitation_reward: float = 0.3

            

        reward_keys_and_weights: RewardWeights = field(default_factory=RewardWeights)

        flag_random_ref_index: bool = False
        out_of_trajectory_threshold: float = 1
        reference_data_path: str = ""

        reference_data_keys: list[str] = field(default_factory=list[str])
    env_params: EnvParams = field(default_factory=EnvParams)