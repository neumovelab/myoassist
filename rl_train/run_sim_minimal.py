from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase

env = MyoAssistLegBase(model_path="models/22muscle_2D/myoLeg22_2D_TUTORIAL.xml",
                       env_params=TrainSessionConfigBase.EnvParams())
env.mujoco_render_frames = True

obs, info = env.reset(seed=1)
for timestep in range(150):
    random_action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(random_action)

