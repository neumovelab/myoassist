import collections
import numpy as np
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils import train_log_handler
from rl_train.utils.learning_callback import BaseCustomLearningCallback
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
################################################################


class ImitationCustomLearningCallback(BaseCustomLearningCallback):
    def __init__(
        self,
        *,
        log_rollout_freq: int,
        evaluate_freq: int,
        log_handler: train_log_handler.TrainLogHandler,
        original_reward_weights: ImitationTrainSessionConfig.EnvParams.RewardWeights,
        auto_reward_adjust_params: ImitationTrainSessionConfig.AutoRewardAdjustParams,
        verbose=1,
    ):
        super().__init__(
            log_rollout_freq=log_rollout_freq, evaluate_freq=evaluate_freq, log_handler=log_handler, verbose=verbose
        )
        self._reward_weights = original_reward_weights
        self._auto_reward_adjust_params = auto_reward_adjust_params

    def _init_callback(self):
        super()._init_callback()

        self.reward_accumulate = DictionableDataclass.create(ImitationTrainSessionConfig.EnvParams.RewardWeights)
        self.reward_accumulate = DictionableDataclass.to_dict(self.reward_accumulate)
        for key in self.reward_accumulate.keys():
            self.reward_accumulate[key] = 0

    # called after all envs step done
    def _on_step(self) -> bool:
        # print("======================self.locals    ======================")
        # # pprint.pprint(self.locals)
        # print(f"DEBUG:: {len(self.locals['infos'])=}")
        # for idx, info in enumerate(self.locals['infos']):
        #     print(f"DEBUG:: {idx=} {info['rwd_dict']=}")
        # print("======================self.locals    ======================")

        # print(f"DEBUG:: {subprocvec_env=}")
        # print(f"DEBUG:: {subprocvec_env.env_method('subproc_env_test', 'This is param from learning callback')=}")
        for info in self.locals["infos"]:
            for key in self.reward_accumulate.keys():
                self.reward_accumulate[key] += info["rwd_dict"][key]

        super()._on_step()

        return True

    def _on_rollout_start(self) -> None:
        super()._on_rollout_start()

    def _on_rollout_end(self, write_log: bool = True) -> "ImitationTrainCheckpointData":
        log_data_base = super()._on_rollout_end(write_log=False)
        if log_data_base is None:
            return
        log_data = ImitationTrainCheckpointData(
            **log_data_base.__dict__,
            reward_weights=DictionableDataclass.to_dict(self._reward_weights),
            reward_accumulate=self.reward_accumulate.copy(),
        )
        if write_log:
            self.train_log_handler.add_log_data(log_data)
            self.train_log_handler.write_json_file()

        self.rewards_sum = np.zeros(self.training_env.num_envs)
        self.episode_counts = np.zeros(self.training_env.num_envs)
        self.episode_length_counts = np.zeros(self.training_env.num_envs)

        ## ARA (Disabled)

        # print(f"DEBUG:: {self.reward_accumulate=}")
        # joint_rewards = {}
        # for key in self.reward_accumulate.keys():
        #     # print(f"DEBUG:: {key=} {self.reward_accumulate[key]=}")
        #     if MyoLeg18Imitation.Q_POS_DIFF_REWARD_KEY_PREFIX in key:
        #         joint_rewards[key] = self.reward_accumulate[key]
        #     self.reward_accumulate[key] = 0
        # reward_mean = 0
        # for key in joint_rewards.keys():
        #     reward_mean += joint_rewards[key]
        # reward_mean /= len(joint_rewards)
        # joint_reward_deviations = {key: (joint_rewards[key] - reward_mean)/reward_mean for key in joint_rewards.keys()}

        # for key in joint_reward_deviations.keys():
        #     new_reward_weight = getattr(self._reward_weights, key) - self._auto_reward_adjust_params.learning_rate * joint_reward_deviations[key]
        #     setattr(self._reward_weights, key, new_reward_weight)
        # subprocvec_env:SubprocVecEnv = self.model.get_env()
        # subprocvec_env.env_method('set_reward_weights', self._reward_weights)
        # print(f"DEBUG:: {self._reward_weights=}")


##############################################################################


class MyoAssistLegImitation(MyoAssistLegBase):
    # Standing-height excess above which the reference's pelvis height is corrected for the
    # device's under-foot thickness. Measured: devices that add nothing under the foot sit within
    # 1 cm of the reference, STRIDE_L2's sole pads put it 5.5 cm above it.
    REFERENCE_HEIGHT_CORRECTION_THRESHOLD = 0.02

    # automatically inherit from MyoAssistLegBase
    # DEFAULT_OBS_KEYS = ['qpos',
    #                     'qvel',
    #                     'act',
    #                     'target_velocity',
    #                     ]

    def _setup(
        self,
        *,
        env_params: ImitationTrainSessionConfig.EnvParams,
        reference_data: dict | None = None,
        loop_reference_data: bool = False,
        **kwargs,
    ):
        self._flag_random_ref_index = env_params.flag_random_ref_index
        self._out_of_trajectory_threshold = env_params.out_of_trajectory_threshold
        self._out_of_trajectory_joint_keys = env_params.out_of_trajectory_joint_keys
        self.reference_data_keys = env_params.reference_data_keys
        self._reset_keyframe_joint_keys = env_params.reset_keyframe_joint_keys
        self._loop_reference_data = loop_reference_data
        self._reward_keys_and_weights: ImitationTrainSessionConfig.EnvParams.RewardWeights = env_params.reward_keys_and_weights

        self.setup_reference_data(data=reference_data)

        super()._setup(
            env_params=env_params,
            **kwargs,
        )

    def set_reward_weights(self, reward_keys_and_weights: TrainSessionConfigBase.EnvParams.RewardWeights):
        self._reward_keys_and_weights = reward_keys_and_weights

    # override from MujocoEnv
    def get_obs_dict(self, sim):
        return super().get_obs_dict(sim)

    def _get_qpos_diff(self) -> dict:

        def get_qpos_diff_one(key: str):
            diff = (
                self.sim.data.joint(f"{key}").qpos[0].copy()
                - self._reference_data["series_data"][f"q_{key}"][self._imitation_index]
            )
            return diff

        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qpos_imitation_rewards:
            name_diff_dict[q_key] = get_qpos_diff_one(q_key)
        return name_diff_dict

    def _get_qvel_diff(self):
        speed_ratio_to_target_velocity = (
            self._target_velocity / self._reference_data["series_data"]["dq_pelvis_tx"][self._imitation_index]
        )

        def get_qvel_diff_one(key: str):
            diff = (
                self.sim.data.joint(f"{key}").qvel[0].copy()
                - self._reference_data["series_data"][f"dq_{key}"][self._imitation_index] * speed_ratio_to_target_velocity
            )
            return diff

        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qvel_imitation_rewards:
            # joint_weight = self._reward_keys_and_weights.qvel_imitation_rewards[q_key]
            name_diff_dict[q_key] = get_qvel_diff_one(q_key)
        return name_diff_dict

    def _get_qpos_diff_nparray(self):
        return np.array([diff for diff in self._get_qpos_diff().values()])

    def _get_out_of_trajectory_diff(self) -> np.ndarray:
        """Tracking errors the episode-ending check looks at.

        By default every joint the imitation reward names, which is what the intact configs
        rely on. `out_of_trajectory_joint_keys` narrows it, which is what lets a reward term
        exist without also being a hard terminator.

        An amputee needs that split. The residual limb wants a weak posture term -- without one
        the knee simply folds, since nothing else in the reward cares about it -- but it cannot
        also be a terminator: it is the joint that deviates most from a healthy walker, and on
        the first 30M runs `knee_angle_r` was the single most frequent cause of episode
        termination.
        """
        if not self._out_of_trajectory_joint_keys:
            return self._get_qpos_diff_nparray()
        diffs = self._get_qpos_diff()
        missing = [k for k in self._out_of_trajectory_joint_keys if k not in diffs]
        assert not missing, (
            f"out_of_trajectory_joint_keys names {missing}, which the imitation reward does not "
            f"track, so there is no reference to compare against. Tracked: {sorted(diffs)}"
        )
        return np.array([diffs[k] for k in self._out_of_trajectory_joint_keys])

    def _get_end_effector_diff(self):
        # body_pos = self.sim.data.body('pelvis').xpos.copy()
        # diff_array = []
        # for mapping in self.ANCHOR_SIM_TO_REF.values():
        #     sim_anchor = self.sim.data.joint(mapping.sim_name).xanchor.copy() - body_pos
        #     ref_anchor = self._reference_data[mapping.ref_name][self._imitation_index]
        #     diff = np.linalg.norm(sim_anchor - ref_anchor)
        #     diff_array.append(diff)
        # return diff_array
        return np.array([0])

    def _calculate_imitation_rewards(self, obs_dict):
        base_reward, base_info = super()._calculate_base_reward(obs_dict)

        q_diff_dict = self._get_qpos_diff()
        dq_diff_dict = self._get_qvel_diff()
        anchor_diff_array = self._get_end_effector_diff()

        # Calculate joint position rewards
        q_reward_dict = {}
        for joint_name, diff in q_diff_dict.items():
            q_reward_dict[joint_name] = self.dt * np.exp(-8 * np.square(diff))

        dq_reward_dict = {}
        for joint_name, diff in dq_diff_dict.items():
            dq_reward_dict[joint_name] = self.dt * np.exp(-8 * np.square(diff))

        # Calculate end effector reward
        anchor_reward = self.dt * np.mean(np.exp(-5 * np.square(anchor_diff_array)))

        # Calculate joint imitation rewards sum
        qpos_imitation_rewards = np.sum(
            [q_reward_dict[key] * self._reward_keys_and_weights.qpos_imitation_rewards[key] for key in q_reward_dict.keys()]
        )
        qvel_imitation_rewards = np.sum(
            [dq_reward_dict[key] * self._reward_keys_and_weights.qvel_imitation_rewards[key] for key in dq_reward_dict.keys()]
        )

        # Add new key-value pairs to the base_reward dictionary
        base_reward.update(
            {
                "qpos_imitation_rewards": qpos_imitation_rewards,
                "qvel_imitation_rewards": qvel_imitation_rewards,
                "end_effector_imitation_reward": anchor_reward,
            }
        )

        # Use the updated base_reward as imitation_rewards
        imitation_rewards = base_reward
        info = base_info
        return imitation_rewards, info

    # override from MujocoEnv
    def get_reward_dict(self, obs_dict):
        # Calculate common rewards
        imitation_rewards, info = self._calculate_imitation_rewards(obs_dict)

        # Construct reward dictionary
        # Automatically add all imitation_rewards items to rwd_dict
        rwd_dict = collections.OrderedDict((key, imitation_rewards[key]) for key in imitation_rewards)

        # Add additional fixed keys
        rwd_dict.update(
            {
                "sparse": 0,
                "solved": False,
                "done": self._get_done(),
            }
        )
        # Calculate final reward
        rwd_dict["dense"] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items() if key in rwd_dict], axis=0)

        return rwd_dict

    def _reset_keyframe_joints(self):
        """Restore the joints the reference cannot supply to the model's standing keyframe.

        `reset` seeds the next episode from `sim.data.qpos`, so a DOF the reference does not
        write keeps whatever value it held when the last episode ended -- which is normally the
        value it held while the model was falling. On an intact model the only such DOFs are the
        passive toe joints, and the configs have always run that way.

        An amputee model makes the same carry-over a real problem: the prosthesis' own joint is
        the one the device actuator drives, and the reference is a healthy walker with no
        trajectory for it. Left alone it starts each episode wherever the previous fall left it,
        routinely outside its own limit -- the OSL ankle is limited to +-0.52 rad and was
        measured starting successive episodes at +1.43 and +1.94 rad, so every step after the
        first fall ran against a large limit-constraint force that has nothing to do with gait.

        Opt-in through `reset_keyframe_joint_keys` rather than applied to every joint the
        reference omits, so the intact configs -- and the results already trained from them --
        keep the reset behaviour they were trained under.
        """
        for key in self._reset_keyframe_joint_keys:
            joint_id = self.sim.model.joint(key).id
            self.sim.data.joint(key).qpos = self.sim.model.key_qpos[0][self.sim.model.jnt_qposadr[joint_id]]
            self.sim.data.joint(key).qvel = self.sim.model.key_qvel[0][self.sim.model.jnt_dofadr[joint_id]]

    def _follow_reference_motion(self, is_x_follow: bool):
        for key in self.reference_data_keys:
            self.sim.data.joint(f"{key}").qpos = self._reference_data["series_data"][f"q_{key}"][self._imitation_index]
            if not is_x_follow and key == "pelvis_tx":
                self.sim.data.joint(f"{key}").qpos = 0
            # if key == 'pelvis_ty':
            #     self.sim.data.joint(f"{key}").qpos += 0.05
        speed_ratio_to_target_velocity = (
            self._target_velocity / self._reference_data["series_data"]["dq_pelvis_tx"][self._imitation_index]
        )
        for key in self.reference_data_keys:
            self.sim.data.joint(f"{key}").qvel = (
                self._reference_data["series_data"][f"dq_{key}"][self._imitation_index] * speed_ratio_to_target_velocity
            )

    def imitation_step(self, is_x_follow: bool, specific_index: int | None = None):
        if specific_index is None:
            self._imitation_index += 1
            if self._imitation_index >= self._reference_data_length:
                self._imitation_index = 0
        else:
            self._imitation_index = specific_index
        self._follow_reference_motion(is_x_follow)
        # should call this but I don't know why
        # next_obs, reward, terminated, truncated, info = super().step(np.zeros(self.sim.model.nu))
        # return (next_obs, reward, False, False, info)
        self.forward()
        return self._imitation_index
        # pass

    # override
    def step(self, a, **kwargs):
        if self._imitation_index is not None:
            self._imitation_index += 1
            if self._imitation_index < self._reference_data_length:
                is_out_of_index = False
            else:
                if self._loop_reference_data:
                    self._imitation_index = 0
                    is_out_of_index = False
                else:
                    is_out_of_index = True
                    self._imitation_index = self._reference_data_length - 1
        else:
            is_out_of_index = True

        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        if is_out_of_index:
            reward = 0
            truncated = True
        else:
            q_diff_nparray: np.ndarray = self._get_out_of_trajectory_diff()
            is_out_of_trajectory = np.any(np.abs(q_diff_nparray) > self._out_of_trajectory_threshold)
            terminated = terminated or is_out_of_trajectory

        return (next_obs, reward, terminated, truncated, info)

    def setup_reference_data(self, data: dict | None):
        if data is None:
            raise ValueError("Reference data is not set")
        self._reference_data = self._height_corrected_reference(data)
        self._imitation_index = None
        self._reference_data_length = self._reference_data["metadata"]["resampled_data_length"]

    def _standing_pelvis_height(self) -> float | None:
        """The `pelvis_ty` qpos at which this composed model's feet reach the ground.

        Normally the keyframe. assist_sim authors it per composition with the feet on the ground,
        and reading it there keeps every device that satisfies that assumption exactly as it was.

        The fallback is a guard, and every shipped composition currently takes the fast path. It
        exists because two did not: `myolegs22 + OpenSourceLeg_A_L1` and `+ OpenSourceLeg_KA_L1`
        opened 9 cm above the floor with no ground contact at all, because
        `compose._model_ground_candidates` read a horizontal capsule's half-length as its
        downward extent and seated the model on a point that was not there. That is fixed at the
        source, but the failure was invisible from the RL side -- a config that trains, on a model
        that never touches the ground -- so the check stays: no contact at the keyframe pose means
        the keyframe is not a standing pose, and the height is then measured by lowering the
        pelvis until the feet touch.

        Returns None when the model has no keyframe to start from.
        """
        model, data = self.sim.model, self.sim.data
        if model.nkey == 0:
            return None

        ty_adr = model.jnt_qposadr[model.joint("pelvis_ty").id]
        keyframe_height = float(model.key_qpos[0][ty_adr])

        saved_qpos, saved_qvel = data.qpos.copy(), data.qvel.copy()
        try:
            data.qpos[:] = model.key_qpos[0]
            data.qvel[:] = 0
            self.sim.forward()
            if data.ncon > 0:
                return keyframe_height

            # Bisect on "is anything touching": above the answer nothing is, below it something
            # is. 0.4 m is more than the largest gap any shipped composition produces.
            low, high = keyframe_height - 0.4, keyframe_height
            for _ in range(40):
                mid = 0.5 * (low + high)
                data.qpos[ty_adr] = mid
                self.sim.forward()
                if data.ncon > 0:
                    low = mid
                else:
                    high = mid
            standing = 0.5 * (low + high)
            print(
                f"Keyframe pose has no ground contact, so its pelvis height {keyframe_height:.3f} m "
                f"is not a standing height; measured {standing:.3f} m by lowering onto the floor."
            )
            return standing
        finally:
            data.qpos[:] = saved_qpos
            data.qvel[:] = saved_qvel
            self.sim.forward()

    def _height_corrected_reference(self, data: dict) -> dict:
        """Shift the reference's pelvis height so the composed model's feet reach the ground.

        The reference is motion capture of a bare musculoskeletal model, so its `q_pelvis_ty`
        assumes the human foot is the ground-contacting surface. A device that puts material
        under the foot breaks that assumption: `STRIDE_L2` adds sole pads and stands 5.5 cm
        higher, above the reference's entire range (0.859-0.945 m), so every reset placed it
        below its own soles and the imitation term then demanded a pelvis height its geometry
        cannot reach. It never learned to walk.

        The shift is read off the composed model rather than configured or tabulated, because it
        is a geometric consequence of the device and nobody should have to look it up when adding
        one. The model's keyframe is authored per composition with the feet on the ground, so the
        gap between that standing height and the reference's own mean height is exactly the
        thickness the device introduced.

        Applied to a copy of the reference array, once, so every downstream reader -- the pose
        `_follow_reference_motion` writes, the imitation reward, the out-of-trajectory check --
        sees the corrected target without knowing about the shift.
        """
        series = data["series_data"]
        if "q_pelvis_ty" not in series or self.sim.model.nkey == 0:
            return data

        standing_height = self._standing_pelvis_height()
        if standing_height is None:
            return data
        reference_height = float(np.mean(series["q_pelvis_ty"]))
        offset = standing_height - reference_height

        # Compared on magnitude, not sign. A device that adds material under the foot needs the
        # reference raised; a composition that stands *below* the reference needs it lowered, and
        # the two OpenSourceLeg models do -- their standing height is 0.823 m against the
        # reference's 0.906 m mean, so every reset placed the pelvis 8.3 cm above the height at
        # which their feet reach the floor, and the imitation term then rewarded holding it there.
        # Devices that add nothing under the foot land within a centimetre of the reference, and
        # the existing configs train correctly at that residual, so leave them exactly as they
        # were rather than perturbing every run for a rounding difference.
        if abs(offset) < self.REFERENCE_HEIGHT_CORRECTION_THRESHOLD:
            return data

        corrected = dict(data)
        corrected["series_data"] = dict(series)
        corrected["series_data"]["q_pelvis_ty"] = np.asarray(series["q_pelvis_ty"], dtype=float) + offset
        print(
            f"Reference pelvis height shifted by {offset:+.4f} m: this composition stands at "
            f"{standing_height:.3f} m against the reference's {reference_height:.3f} m"
        )
        return corrected

    def reset(self, **kwargs):
        rng = np.random.default_rng()  # TODO: refactoring random to use seed

        if self._flag_random_ref_index:
            self._imitation_index = rng.integers(0, int(self._reference_data_length * 0.8))
        else:
            self._imitation_index = 0
        # generate random targets
        # new_qpos = self.generate_qpos()# TODO: should set qvel too.
        # self.sim.data.qpos = new_qpos
        self._reset_keyframe_joints()
        self._follow_reference_motion(False)

        obs = super().reset(reset_qpos=self.sim.data.qpos, reset_qvel=self.sim.data.qvel, **kwargs)
        return obs

    # override
    def _initialize_pose(self):
        super()._initialize_pose()
