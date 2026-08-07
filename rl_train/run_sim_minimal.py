from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from myoassist_utils.compose import compose_env_model

# A10: compose the model (human MSK + assistive device + a flat default ground)
# via the shared pipeline instead of loading a bundled static XML. Swap in a
# terrains JSON config path for `terrain=` to run on structured terrain.
model_xml = compose_env_model("myolegs22", "DephyExoBoot_L1", terrain=None)

env = MyoAssistLegBase(model_path=model_xml, env_params=TrainSessionConfigBase.EnvParams())
env.mujoco_render_frames = True

obs, info = env.reset(seed=1)
for timestep in range(150):
    random_action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(random_action)
