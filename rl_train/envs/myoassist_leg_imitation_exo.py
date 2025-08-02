import collections
import numpy as np
from dataclasses import dataclass, field
from rl_train.envs.myoassist_leg_imitation import MyoAssistLegImitation
from rl_train.envs.myoassist_leg_imitation import ImitationTrainSessionConfig
from rl_train.envs.myoassist_leg_imitation import ImitationCustomLearningCallback


from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils import train_log_handler
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_train.utils.learning_callback import BaseCustomLearningCallback

class MyoAssistLegImitationExo(MyoAssistLegImitation):
    # override
    def step(self, a, **kwargs):
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        
        return (next_obs, reward, terminated, truncated, info)
        