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
            qpos_imitation_rewards: dict = field(default_factory=dict)
            qvel_imitation_rewards: dict = field(default_factory=dict)

        reward_keys_and_weights: RewardWeights = field(default_factory=RewardWeights)

        flag_random_ref_index: bool = False
        out_of_trajectory_threshold: float = 1
        reference_data_path: str = ""

        reference_data_keys: list[str] = field(default_factory=list[str])

        # Advance the reference at the target velocity instead of one frame per control step.
        # Off by default: every existing config plays it at its recorded 1.281 m/s cadence, which
        # is right when the target velocity is also 1.25. It stops being right under a speed
        # curriculum -- at a 0.3 m/s target the qpos terms still demand full-stride angles at
        # full cadence, which cannot coexist with 0.3 m/s of forward travel, so the imitation and
        # forward terms pull against each other. Scaling the playback keeps the stride and slows
        # the cadence, and speed = stride x cadence then follows.
        scale_reference_playback: bool = False

        # Joints the out-of-trajectory termination watches. Empty -> every key in
        # qpos_imitation_rewards, which is what the intact configs have always done. Set it to
        # decouple "guide this joint" from "end the episode when this joint drifts": an amputee
        # config wants the residual limb shaped by the reward without a healthy-gait deviation
        # there killing the episode. See MyoAssistLegImitation._get_out_of_trajectory_diff.
        out_of_trajectory_joint_keys: list[str] = field(default_factory=list[str])

    env_params: EnvParams = field(default_factory=EnvParams)
