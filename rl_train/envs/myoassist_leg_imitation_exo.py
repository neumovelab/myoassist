from rl_train.envs.myoassist_leg_imitation import MyoAssistLegImitation


class MyoAssistLegImitationExo(MyoAssistLegImitation):
    # override
    def step(self, a, **kwargs):
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)

        return (next_obs, reward, terminated, truncated, info)
