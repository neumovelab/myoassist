import numpy as np
from dataclasses import dataclass
from myosuite.utils import gym
from myosuite.envs import env_base
from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.utils.data_types import DictionableDataclass
import collections
import mujoco
from myoassist_utils.hfield_manager import HfieldManager
from enum import Enum
import random
import numpy as np


class MyoAssistLegBase(env_base.MujocoEnv):
    MYO_CREDIT = """\
    NeuMove MyoLegBase
    """

    class VelocityMode(Enum):
        UNIFORM = 0
        SINUSOIDAL = 1
        STEP = 2

    JOINT_LIMIT_SENSOR_NAMES = [
        "r_knee_sensor",
        "l_knee_sensor",
        "r_hip_sensor",
        "l_hip_sensor",
        "r_ankle_sensor",
        "l_ankle_sensor",
        "r_mtp_sensor",
        "l_mtp_sensor",
    ]


    DEFAULT_OBS_KEYS = ['qpos',
                        'qvel',
                        'act',
                        'sensor',
                        'target_velocity',
                        ]
    
    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        print(f"=================environment seed: {seed}=====================")
        print(f"=================environment model_path: {model_path}=====================")
        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)
    def _setup(self,*,
               env_params:TrainSessionConfigBase.EnvParams,
                **kwargs):
        
        self.is_evaluate_mode = kwargs.pop("is_evaluate_mode", False)

        self.sim.model.opt.timestep = 1 / env_params.physics_sim_framerate
        self._safe_height = env_params.safe_height

        self._min_target_velocity = env_params.min_target_velocity
        self._max_target_velocity = env_params.max_target_velocity
        self._min_target_velocity_period = env_params.min_target_velocity_period
        self._max_target_velocity_period = env_params.max_target_velocity_period
        self._change_mode_and_target_velocity_randomly()

        self._step_count_per_episode = 0
        self.CUSTOM_MAX_EPISODE_STEPS = env_params.custom_max_episode_steps

        self._prev_muscle_activations_for_reward = None

        self._enable_lumbar_joint = env_params.enable_lumbar_joint
        self._lumbar_joint_fixed_angle = env_params.lumbar_joint_fixed_angle
        self._lumbar_joint_damping_value = env_params.lumbar_joint_damping_value

        self.observation_joint_pos_keys = env_params.observation_joint_pos_keys
        self.observation_joint_vel_keys = env_params.observation_joint_vel_keys
        self.observation_joint_sensor_keys = env_params.observation_joint_sensor_keys

        # Safely check whether the joint named "lumbar_extension" exists in the model.
        try:
            lumbar_joint_id = self.sim.model.joint("lumbar_extension").id  # Raises if joint is absent
            has_lumbar_extension = True
        except (KeyError, ValueError, TypeError):
            has_lumbar_extension = False

        if not self._enable_lumbar_joint:
            if 'lumbar_extension' in self.observation_joint_pos_keys:
                self.observation_joint_pos_keys.remove('lumbar_extension')
            if 'lumbar_extension' in self.observation_joint_vel_keys:
                self.observation_joint_vel_keys.remove('lumbar_extension')
            if has_lumbar_extension:
                # Fix the lumbar joint to a constant position and (optionally) remove it from observations
                self.sim.data.joint("lumbar_extension").qpos[0] = self._lumbar_joint_fixed_angle
                self.sim.model.jnt_range[lumbar_joint_id] = [
                    self._lumbar_joint_fixed_angle,
                    self._lumbar_joint_fixed_angle + 1e-6,
                ]


                # Adjust damping (whether the joint is fixed or not)
                dof_adr = self.sim.model.jnt_dofadr[lumbar_joint_id]
                joint_type = self.sim.model.jnt_type[lumbar_joint_id]
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    dof_count = 6
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    dof_count = 3
                elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                    dof_count = 1
                elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    dof_count = 1
                else:
                    dof_count = 0  # Currently unused

                self.sim.model.dof_damping[dof_adr] = self._lumbar_joint_damping_value
            else:
                self.sim.model.body("torso").quat = [1, 0, 0, self._lumbar_joint_fixed_angle]
        
        
        #phys: 1000hz
        # control 50hz : 50 * 20 = 1000hz
        # ref 50hz: 500hz 10skip: 20 * 500 / 1000

        frame_skip = env_params.physics_sim_framerate // env_params.control_framerate
        original_reward_dict = DictionableDataclass.to_dict(env_params.reward_keys_and_weights)
        self.rwd_keys_wt = {}
        for key, value in original_reward_dict.items():
            if type(value) == dict:
                weight_sum = sum(value.values())
                self.rwd_keys_wt[key] = weight_sum
            else:
                self.rwd_keys_wt[key] = value

        self._initialize_pose()

        # reward per step
        self._reset_heel_strike_buffer()
        self._reset_reward_per_step()
        self._reset_properties_per_step()

        
        
        # self.renderer = self.sim._create_renderer(self.sim)



        super()._setup(obs_keys=self.DEFAULT_OBS_KEYS,
                weighted_reward_keys=self.rwd_keys_wt,
                frame_skip=frame_skip,
                **kwargs,
                )
        
        # Check if the keys in DEFAULT_OBS_KEYS are in the keys of the observation dictionary
        obs_dict_keys = list(self.get_obs_dict(self.sim).keys())
        assert (set(self.DEFAULT_OBS_KEYS + ['time'])) == set(obs_dict_keys), f"DEFAULT_OBS_KEYS != get_obs_dict.keys. DEFAULT_OBS_KEYS: {self.DEFAULT_OBS_KEYS}, get_obs_dict keys: {obs_dict_keys}"
        actual_reward_keys = list(self.get_reward_dict(self.sim).keys())
        assert (set(list(self.rwd_keys_wt.keys()) + ['dense', 'sparse', 'solved', 'done'])) == set(actual_reward_keys), f"rwd_keys_wt != actual_reward_keys. rwd_keys_wt: {self.rwd_keys_wt}, actual_reward_keys keys: {actual_reward_keys}"
        
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = self.sim.model.key_qvel[0]

        # find geometries with ID == 1 which indicates the skins
        geom_1_indices = np.where(self.sim.model.geom_group == 1)
        # Change the alpha value to make it transparent
        self.sim.model.geom_rgba[geom_1_indices, 3] = 0

        # move heightfield down if not used
        # self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 0.0
        # self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, -10])

        self._terrain_type = env_params.terrain_type
        self._terrain_params = env_params.terrain_params
        self._hfield_manager = HfieldManager(self.sim, "terrain", self.np_random)
        self._hfield_manager.set_hfield(self._terrain_type, self._terrain_params)

        observation, _reward, done, *_, _info = self.step(np.zeros(self.sim.model.nu))
        # if qpos set to all zero, joint looks weird, 30 steps will make it normal
        for _ in range(30):
            super().step(a=np.zeros(self.sim.model.nu))

    # override from MujocoEnv
    def get_obs_dict(self, sim):
        # TODO observation - tx exclude
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time]) # they use time separately like t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)

        qpos = []
        for key in self.observation_joint_pos_keys:
            qpos.append(sim.data.joint(f"{key}").qpos[0].copy())
        qvel = []
        for key in self.observation_joint_vel_keys:
            qvel.append(sim.data.joint(f"{key}").qvel[0].copy())
        obs_dict['qpos'] = np.array(qpos) # 7 + 1 elements
        obs_dict['qvel'] = np.array(qvel) # 7 + 2 elements
        if sim.model.na>0:
            # BaseV0 Add the key like this: obs_keys.append("act")
            obs_dict['act'] = sim.data.act[:].copy() # 22 elements
        obs_dict['sensor'] = []
        for key in self.observation_joint_sensor_keys:
            sensor_data = sim.data.sensor(f"{key}").data.copy()
            if "foot" in key or "toes" in key:
                model_mass = np.sum(self.sim.model.body_mass)
                sensor_data = sensor_data / (model_mass * 9.81)
            obs_dict['sensor'].extend(sensor_data)
        obs_dict['sensor'] = np.array(obs_dict['sensor'])

        obs_dict['target_velocity'] = np.array([self._target_velocity])

        return obs_dict
    def _calculate_reward_per_step(self, obs_dict, muscle_activations):
        self._footstep_delta_time += self.dt
        self._delta_velocity_sum += self.dt * (self.sim.data.joint("pelvis_tx").qvel[0].copy() - self._target_velocity)
        self._activation_square_sum += np.sum(np.square(muscle_activations)) * self.dt
        
        time_passed = self.sim.data.time - self._prev_step_time
        if self._detect_heel_strike() and time_passed > 0:
            travel_distance = self.sim.data.body('pelvis').xpos[0] - self._prev_pelvis_tx_pos
            self.reward_muscle_activation_penalty_per_step = self.dt * (-self._activation_square_sum)
            leg_length = 1 # see https://github.com/stanfordnmbl/osim-rl/blob/master/osim/env/osim.py
            self.reward_average_velocity_per_step = self.dt * (-np.abs(self._delta_velocity_sum)) / leg_length
            self.reward_footstep_delta_time = self.dt * self._footstep_delta_time

            self._reset_properties_per_step()
        else:
            pass
            # reward_muscle_activation_penalty_per_step = 0.0
            # reward_average_velocity_per_step = 0.0
            # reward_footstep_delta_time = 0.0
        reward_per_steps = {
            'muscle_activation_penalty_per_step': float(self.reward_muscle_activation_penalty_per_step),
            'average_velocity_per_step': float(self.reward_average_velocity_per_step),
            'footstep_delta_time': float(self.reward_footstep_delta_time),
        }
        info = {}
        return reward_per_steps, info
    def _calculate_base_reward(self, obs_dict):
        model_mass = np.sum(self.sim.model.body_mass)
        # print(f"DEBUG:: model_mass: {model_mass}")
        model_weight = model_mass * 9.81 # in Newtons

        forward_reward = self.dt * np.exp(-5 * np.square(self.sim.data.joint("pelvis_tx").qvel[0].copy() - self._target_velocity))

        muscle_activations = self._get_muscle_activation()
        muscle_activation_penalty = - self.dt * np.mean(muscle_activations)

        joint_constraint_force_penalty = - self.dt * self._get_max_joint_constraint_force() / (model_weight)

        # TODO: take off muscle activation penalty from imitation rewards
        reward_per_steps, info = self._calculate_reward_per_step(obs_dict, muscle_activations)    

        if self._prev_muscle_activations_for_reward is not None:
            muscle_activation_diff_penalty = self.dt * np.mean(np.exp(-4 * np.square(self._prev_muscle_activations_for_reward - muscle_activations)))
        else:
            muscle_activation_diff_penalty = 0
        self._prev_muscle_activations_for_reward = muscle_activations

        
        normalized_foot_force_sum = (np.abs(self._get_foot_force('r')) + np.abs(self._get_foot_force('l'))) / model_weight
        # print(f"DEBUG:: normalized_foot_force_sum: {normalized_foot_force_sum}")
        # e^(-max(0, f/w - 1))
        # foot_force_penalty = self.dt * np.exp(-np.maximum(0, normalized_foot_force_sum - 1))
        foot_force_penalty = - self.dt * max(0, normalized_foot_force_sum - 1.2)
        # print(f"DEBUG:: foot_force_penalty: {foot_force_penalty}")

        base_reward = {
            'forward_reward': forward_reward,
            'muscle_activation_penalty': muscle_activation_penalty,
            'muscle_activation_diff_penalty': muscle_activation_diff_penalty,
            'foot_force_penalty': foot_force_penalty,
            'joint_constraint_force_penalty': joint_constraint_force_penalty,
        }
        # Update base_reward with reward_per_steps
        base_reward.update(reward_per_steps)

        info = {
            "muscle_activations": muscle_activations,
        }
        return base_reward, info
    # override from MujocoEnv
    def get_reward_dict(self, obs_dict):

        base_reward, info = self._calculate_base_reward(obs_dict)

        # Automatically add all base_reward items to rwd_dict
        rwd_dict = collections.OrderedDict((key, base_reward[key]) for key in base_reward)

        # Add additional fixed keys
        rwd_dict.update({
            'sparse': 0,
            'solved': False,
            'done': self._get_done(),  # env will use this to determine if the episode is over (see _forward in env_base.py)
        })
        # rwd_keys_wt: from MujocoEnv
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict
    
    def step(self, a, **kwargs):
        self._modulate_target_velocity()
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        self._step_count_per_episode += 1
        is_over_time_limit = self._step_count_per_episode >= self.CUSTOM_MAX_EPISODE_STEPS
        
        return (next_obs, reward, terminated, truncated or is_over_time_limit, info)
    def just_forward(self):
        self.sim.forward()
    def set_target_velocity_mode_manually(self, mode:VelocityMode, starting_phase:float, initial_target_velocity:float, target_velocity_period:float = None):
        self._velocity_mode_for_this_episode = mode
        self._starting_phase = starting_phase
        if mode == MyoAssistLegBase.VelocityMode.SINUSOIDAL and target_velocity_period is None:
            raise ValueError("target_velocity_period must be provided for sinusoidal mode")
        self._target_velocity_period = target_velocity_period
        # self._modulate_target_velocity()
        self._target_velocity = initial_target_velocity
        self._prev_step_changed_time = self.sim.data.time
    def _change_mode_and_target_velocity_randomly(self):
        velocity_mode_for_this_episode = random.choice(list(MyoAssistLegBase.VelocityMode))
        starting_phase = random.uniform(0, 2 * np.pi)
        target_velocity_period = random.uniform(self._min_target_velocity_period, self._max_target_velocity_period) # maximum acc/dec is self._target_velocity_period / 2
        self.set_target_velocity_mode_manually(velocity_mode_for_this_episode, self._min_target_velocity, starting_phase, target_velocity_period)
        if self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.UNIFORM:
            self._target_velocity = random.uniform(self._min_target_velocity, self._max_target_velocity)
        elif self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.SINUSOIDAL:
            self._target_velocity = self._calc_sinusoidal_target_velocity(self._starting_phase,
                                                                          self._target_velocity_period,
                                                                          self._min_target_velocity,
                                                                          self._max_target_velocity)
        elif self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.STEP:
            self._target_velocity = np.random.uniform(self._min_target_velocity, self._max_target_velocity)


    def _calc_sinusoidal_target_velocity(self, phase:float, period:float, min_velocity:float, max_velocity:float):
        return min_velocity\
            + (max_velocity - min_velocity)\
                  * (np.sin(phase + 2 * np.pi * self.sim.data.time / (period)) + 1) / 2
    def _modulate_target_velocity(self):
        if self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.UNIFORM:
            # print(f"DEBUG:: {self.is_evaluate_mode} (mode:{self._velocity_mode_for_this_episode}, starting_phase:{self._starting_phase}, target_velocity_period:{self._target_velocity_period})")
            pass
        elif self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.SINUSOIDAL:
            self._target_velocity = self._calc_sinusoidal_target_velocity(self._starting_phase,
                                                                          self._target_velocity_period,
                                                                          self._min_target_velocity,
                                                                          self._max_target_velocity)
        elif self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.STEP:
            if self.sim.data.time - self._prev_step_changed_time > self._target_velocity_period:
                self._target_velocity = np.random.uniform(self._min_target_velocity, self._max_target_velocity)
                self._prev_step_changed_time = self.sim.data.time
    def reset(self, **kwargs):
        self._step_count_per_episode = 0
        if not self.is_evaluate_mode:
            self._change_mode_and_target_velocity_randomly()
        self.sim.data.joint("pelvis_tx").qvel[0] = self._target_velocity

        self.sim.forward()
        # sync targets to sim_obsd
        self.robot.sync_sims(self.sim, self.sim_obsd)

        self._reset_heel_strike_buffer()
        self._reset_reward_per_step()
        self._reset_properties_per_step()

        # generate resets
        # obs = super().reset(reset_qpos= self.sim.data.qpos, reset_qvel=self.sim.data.qvel, **kwargs)
        obs = super().reset(**kwargs)
        return obs
    
    def _get_done(self):
        pelvis_height = self.sim.data.joint('pelvis_ty').qpos[0].copy()
        if pelvis_height < self._safe_height:
            return True
        return False
    
    def _get_muscle_activation(self):
        if not self._enable_lumbar_joint:
            return self.sim.data.act[:].copy()
        muscle_activations_with_lumbar = np.concatenate((self.sim.data.act[:].copy(), 
                                              np.array([self.sim.data.actuator('lumbar_extension_motor').ctrl[0].copy()]).reshape(1,)))
        return muscle_activations_with_lumbar
    def _get_max_joint_constraint_force(self):
        max_constraint_force = 0
        for sensor_name in self.JOINT_LIMIT_SENSOR_NAMES:
            sensor_data = self.sim.data.sensor(sensor_name).data[0].copy()
            max_constraint_force = max(max_constraint_force, np.max(np.abs(sensor_data)))
        return max_constraint_force
    # ============ Custon Function ==============
    def _reset_properties_per_step(self):
        self._prev_pelvis_tx_pos = self.sim.data.body('pelvis').xpos[0]
        self._prev_step_time = self.sim.data.time
        self._activation_square_sum = 0
        self._footstep_delta_time = 0
        self._delta_velocity_sum = 0
    def _reset_reward_per_step(self):
        # prev reward save for non-sparse reward ( helpful for training? )
        self.reward_muscle_activation_penalty_per_step = 0
        self.reward_average_velocity_per_step = 0
        self.reward_footstep_delta_time = 0
        
    def _reset_heel_strike_buffer(self):
        self._r_heel_striking_value_buffer = []
        self._l_heel_striking_value_buffer = []
        self._last_heel_strike_foot = ""
    def _detect_heel_strike(self):
        r_foot_force = self._get_foot_force("r")
        l_foot_force = self._get_foot_force("l")

        self._r_heel_striking_value_buffer.append( r_foot_force)
        self._l_heel_striking_value_buffer.append(l_foot_force)

        if len(self._r_heel_striking_value_buffer) > 2:
            last_three_min = min(self._r_heel_striking_value_buffer[-3:])
            if last_three_min > 0.1 and self._last_heel_strike_foot != "right":
                self._last_heel_strike_foot = "right"
                # print("DEBUG:: right heel strike")
                return True
        if len(self._l_heel_striking_value_buffer) > 2:
            last_three_min = min(self._l_heel_striking_value_buffer[-3:])
            if last_three_min > 0.1 and self._last_heel_strike_foot != "left":
                self._last_heel_strike_foot = "left"
                # print("DEBUG:: left heel strike")
                return True
        return False
    def _get_foot_force(self, foot_side_alphabet:str):
        foot_force = self.sim.data.sensor(f"{foot_side_alphabet}_foot").data.copy()[0] + self.sim.data.sensor(f"{foot_side_alphabet}_toes").data.copy()[0]
        return foot_force

    # To override
    def _initialize_pose(self):
        self.sim.data.qpos[:] = self.sim.model.key_qpos[0][:]
        self.just_forward()