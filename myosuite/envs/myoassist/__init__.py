from myosuite.utils import gym; register=gym.register
from myosuite.envs.env_variants import register_env_variant

def register_env_myoassist(id, entry_point, max_episode_steps, kwargs):
    # register_env_with_variants base env
    register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs
    )
    #register variants env with sarcopenia
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'sarcopenia'},
            variant_id=id[:3]+"Sarc"+id[3:],
            silent=True
        )
    #register variants with fatigue
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'fatigue'},
            variant_id=id[:3]+"Fati"+id[3:],
            silent=True
        )

    #register variants with tendon transfer
    if id[:7] == "myoHand":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'reafferentation'},
            variant_id=id[:3]+"Reaf"+id[3:],
            silent=True
        )

# curr_dir = os.path.dirname(os.path.abspath(__file__))

register_env_myoassist(id='myoLeg18Imitation-v0',
        entry_point='myosuite.envs.myoassist.myo_leg_18_imitation:MyoLeg18Imitation',
        max_episode_steps=1000,
        kwargs={},
    )

register_env_myoassist(id='myoLeg18Base-v0',
        entry_point='myosuite.envs.myoassist.myo_leg_18_base:MyoLeg18Base',
        max_episode_steps=1000,
        kwargs={},
    )

register_env_myoassist(id='myoLeg18Guided-v0',
        entry_point='myosuite.envs.myoassist.myo_leg_18_guided:MyoLeg18Guided',
        max_episode_steps=1000,
        kwargs={},
    )
register_env_myoassist(id='myoLeg18ImitationDephy-v0',
        entry_point='myosuite.envs.myoassist.myo_leg_18_imitation_dephy:myoLeg18ImitationDephy',
        max_episode_steps=1000,
        kwargs={},
    )
# register_env_myoassist(id='myoLeg18RewardPerStep-v0',
#         entry_point='myosuite.envs.myoassist.myo_leg_18_reward_per_step:myoLeg18RewardPerStep',
#         max_episode_steps=1000,
#         kwargs={},
#     )