from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from datetime import datetime
from myoassist_rl.rl_train.utils.handlers import train_log_handler
from myoassist_rl.rl_train.analyzer.train_analyzer import TrainAnalyzer
from myoassist_rl.rl_train.utils.handlers.train_checkpoint_data import TrainCheckpointData
import multiprocessing
from multiprocessing import Pool
import functools

def _analyze_process(log_dir):
    """
    Independent function for analysis process
    """
    train_analyzer = TrainAnalyzer()
    train_analyzer.analyze_in_sequence(log_dir, show_plot=False)

class BaseCustomLearningCallback(BaseCallback):
    def __init__(self, *,
                 log_rollout_freq: int,
                 evaluate_freq: int,
                 log_handler:train_log_handler.TrainLogHandler,
                 verbose=1):
        super().__init__(verbose)
        self.log_rollout_freq = log_rollout_freq
        self.evaluate_freq = evaluate_freq
        self.train_log_handler:train_log_handler.TrainLogHandler = log_handler
        self.log_count = 0
        
        # Move the analyze_process function to class level
        # self.analyze_process = functools.partial(_analyze_process)
        self.pool = None

    def _init_callback(self):
        self.rewards_sum = np.zeros(self.training_env.num_envs)
        self.current_episode_rewards = np.zeros(self.training_env.num_envs)
        self.episode_counts = np.zeros(self.training_env.num_envs)
        self.episode_length_counts = np.zeros(self.training_env.num_envs)
        self.current_episode_length_counts = np.zeros(self.training_env.num_envs)

        self.current_reward_dict_sum = [{} for _ in range(self.training_env.num_envs)]
        self.episode_reward_dict_sum = [{} for _ in range(self.training_env.num_envs)]

        self.prev_logging_timestep = 0

        
    #called after all envs step done
    def _on_step(self) -> bool:
        self.current_episode_rewards += self.locals["rewards"]
        for idx, done in enumerate(self.locals["dones"]):
            self.current_episode_length_counts[idx] += 1

            # ------------------------------------------------------------------
            # Safeguard: 'info' may be None or may not contain 'rwd_dict'.
            # In such cases we skip per-key reward accumulation instead of
            # triggering a `TypeError: 'NoneType' object is not subscriptable`.
            # ------------------------------------------------------------------
            info_dict = None
            try:
                info_dict = self.locals["infos"][idx]
            except (IndexError, KeyError):
                # Defensive: infos array shape mismatch → just ignore for now
                info_dict = None

            if info_dict and isinstance(info_dict, dict) and "rwd_dict" in info_dict:
                for key, val in info_dict["rwd_dict"].items():
                    if key not in self.current_reward_dict_sum[idx]:
                        self.current_reward_dict_sum[idx][key] = 0
                    self.current_reward_dict_sum[idx][key] += val
            if done:
                self.rewards_sum[idx] += self.current_episode_rewards[idx]
                self.episode_counts[idx] += 1
                self.current_episode_rewards[idx] = 0.0
                self.episode_length_counts[idx] += self.current_episode_length_counts[idx]
                self.current_episode_length_counts[idx] = 0

                # Aggregate episode-level reward dictionary safely
                for key, val in self.current_reward_dict_sum[idx].items():
                    if key not in self.episode_reward_dict_sum[idx]:
                        self.episode_reward_dict_sum[idx][key] = 0
                    self.episode_reward_dict_sum[idx][key] += val

             
        return True
    def _on_rollout_start(self) -> None:
        super()._on_rollout_start()

    def _on_rollout_end(self, write_log:bool=True) -> TrainCheckpointData|None:
        super()._on_rollout_end()

        self.prev_logging_timestep = self.num_timesteps
        
        log_data = None
        if self.log_count % self.log_rollout_freq == 0:

            model_path = self.train_log_handler.get_path2save_model(self.model.num_timesteps)
            self.model.save(model_path)
            
            def get_logger_value(key, default=-1):
                value = self.logger.name_to_value.get(key, default)
                return value.item() if hasattr(value, 'item') else value
            
            # average_reward_dict_per_episode=[{key:self.episode_reward_dict_sum[idx][key] / self.episode_counts[idx] for key in self.episode_reward_dict_sum[idx].keys()} for idx in range(self.training_env.num_envs)]
            # Calculate the average reward for each key across all environments
            # average_reward_dict_per_episode={key:sum(average_reward_dict_per_episode[idx][key] for idx in range(self.training_env.num_envs)) / self.training_env.num_envs for key in average_reward_dict_per_episode[0].keys()}
            # If there are no keys in self.episode_reward_dict_sum[idx], set all values to 0
            for idx in range(self.training_env.num_envs):
                if not self.episode_reward_dict_sum[idx]:
                    self.episode_reward_dict_sum[idx] = {key: 0 for key in self.episode_reward_dict_sum[0].keys()}

            average_reward_dict_per_episode = {
                key: (
                    sum(
                        self.episode_reward_dict_sum[idx][key]
                        for idx in range(self.training_env.num_envs)
                    ) / sum(self.episode_counts)
                )
                for key in self.episode_reward_dict_sum[0].keys()
            }



            log_data = TrainCheckpointData(
                approx_kl=get_logger_value('train/approx_kl'),
                clip_fraction=get_logger_value('train/clip_fraction'),
                clip_range=get_logger_value('train/clip_range'),
                clip_range_vf=get_logger_value('train/clip_range_vf'),
                entropy_loss=get_logger_value('train/entropy_loss'),
                explained_variance=get_logger_value('train/explained_variance'),
                learning_rate=get_logger_value('train/learning_rate'),
                loss=get_logger_value('train/loss'),
                n_updates=get_logger_value('train/n_updates'),
                policy_gradient_loss=get_logger_value('train/policy_gradient_loss'),
                std=get_logger_value('train/std'),
                value_loss=get_logger_value('train/value_loss'),
                num_timesteps=self.model.num_timesteps,
                average_num_timestep=(np.sum(self.episode_length_counts) / np.sum(self.episode_counts) if np.sum(self.episode_counts) != 0 else np.array(0)).item(),
                average_reward_per_episode=(np.sum(self.rewards_sum) / np.sum(self.episode_counts) if np.sum(self.episode_counts) != 0 else np.array(0)).item(),
                average_reward_dict_per_episode=average_reward_dict_per_episode,
                time=f"{datetime.now().strftime('%Y%m%d-%H%M%S.%f')}",
            )
            if write_log:
                self.train_log_handler.add_log_data(log_data)
                self.train_log_handler.write_json_file()
            
            self.rewards_sum = np.zeros(self.training_env.num_envs)
            self.episode_counts = np.zeros(self.training_env.num_envs)
            self.episode_length_counts = np.zeros(self.training_env.num_envs)
            self.episode_reward_dict_sum = [{} for _ in range(self.training_env.num_envs)]
            self.current_reward_dict_sum = [{} for _ in range(self.training_env.num_envs)]


        if self.log_count % self.evaluate_freq == 0 and self.log_count != 0:
            # Create new pool each time to prevent memory leaks
            pool = Pool(processes=1)
            try:
                pool.apply(
                    _analyze_process,
                    args=(self.train_log_handler.log_dir,)
                )
            finally:
                # Always cleanup pool resources, even if error occurs
                pool.close()  # No more tasks
                pool.join()   # Wait for worker processes to exit
                # Note: If error occurred in pool.apply(), it will still propagate after cleanup
        self.log_count += 1
        
        return log_data


#need refactoring
def _analyze_process(log_dir):
    train_analyzer = TrainAnalyzer()
    train_analyzer.analyze_in_sequence(log_dir, show_plot=False)