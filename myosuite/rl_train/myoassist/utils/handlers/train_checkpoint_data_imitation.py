from dataclasses import dataclass
from myosuite.rl_train.myoassist.utils.handlers.train_checkpoint_data import TrainCheckpointData

@dataclass
class ImitationTrainCheckpointData(TrainCheckpointData):
    reward_accumulate:dict = None
    reward_weights:dict = None