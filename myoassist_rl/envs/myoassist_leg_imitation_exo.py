import collections
import numpy as np
from dataclasses import dataclass, field
from myoassist_rl.envs.myoassist_leg_imitation import MyoAssistLegImitation
from myoassist_rl.envs.myoassist_leg_imitation import ImitationTrainSessionConfig
from myoassist_rl.envs.myoassist_leg_imitation import ImitationCustomLearningCallback


from myoassist_rl.rl_train.utils.config import TrainSessionConfigBase
from myoassist_rl.rl_train.utils.data_types import DictionableDataclass
from myoassist_rl.rl_train.utils.handlers import train_log_handler
from stable_baselines3.common.vec_env import SubprocVecEnv
from myoassist_rl.rl_train.utils.learning_callback import BaseCustomLearningCallback

class MyoAssistLegImitationExo(MyoAssistLegImitation):
    # override
    def step(self, a, **kwargs):
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        
        return (next_obs, reward, terminated, truncated, info)
        