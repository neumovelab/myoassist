from dataclasses import dataclass
from myoassist_rl.rl_train.utils.handlers.train_checkpoint_data import TrainCheckpointData

@dataclass
class ImitationTrainCheckpointData(TrainCheckpointData):
    reward_accumulate:dict = None
    reward_weights:dict = None