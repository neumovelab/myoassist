from myoassist_rl.rl_train.utils.handlers.train_log_handler import TrainLogHandler
import matplotlib.pyplot as plt
import os
import numpy as np

class TrainLogAnalyzer:
    def __init__(self, train_log_handler:TrainLogHandler):
        self._train_log_handler = train_log_handler

    def plot_reward(self, *, result_dir:str, show_plot:bool):
        time_steps = [log_data.num_timesteps for log_data in self._train_log_handler.log_datas]
        rewards = [log_data.average_reward_per_episode for log_data in self._train_log_handler.log_datas]
        
        fig, axes = plt.subplots(1,1,figsize=(15,15),dpi=300)
        axes.plot(time_steps, rewards)
        axes.set_title("Reward")
        axes.set_xlabel("Timesteps")
        axes.set_ylabel("Reward")
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, "reward.png"))
        if show_plot:
            plt.figure(fig.number)
            plt.show()

    def plot_reward_dict(self, *, result_dir:str, show_plot:bool, mult_weights:bool=False):
        start_index = 1 #TODO: Don't know why, but, the first average_reward_dict_per_episode data is empty dict
        modified_log_datas = self._train_log_handler.log_datas[start_index:]
        # Extract time steps and average reward dictionary per episode
        time_steps = [log_data.num_timesteps for log_data in modified_log_datas]
        
        # Defensive check: ensure we have valid data and keys
        if not modified_log_datas or not modified_log_datas[0].average_reward_dict_per_episode:
            print("Warning: No valid reward dictionary data found. Skipping reward dict plot.")
            return

        reward_dict_keys = modified_log_datas[0].average_reward_dict_per_episode.keys()
        # print(f"{reward_dict_keys=}")

        # Debug: Print reward_dict_keys to understand what keys are available
        # print(f"DEBUG: reward_dict_keys = {list(reward_dict_keys)}")
        # print(f"DEBUG: average_reward_dict_per_episode = {self._train_log_handler.log_datas[0].average_reward_dict_per_episode}")
        
        # # Check which keys are being filtered out
        # excluded_keys = ['sparse', 'solved', 'done', 'dense']
        # filtered_keys = [key for key in reward_dict_keys if key not in excluded_keys]
        # print(f"DEBUG: Keys after filtering (excluding {excluded_keys}): {filtered_keys}")

        original_total_reward = [log_data.average_reward_per_episode for log_data in modified_log_datas]
        
        # Prepare data for each key in the average_reward_dict_per_episode, excluding 'sparse', 'solved', 'done', 'dense'
        # Add defensive check for missing keys
        excluded_keys = ['sparse', 'solved', 'done', 'dense']
        valid_keys = [key for key in reward_dict_keys if key not in excluded_keys]
        
        if not valid_keys:
            print("Warning: No valid reward keys found after filtering. Skipping reward dict plot.")
            return

        reward_dict_data = {}
        for key in valid_keys:
            try:
                reward_dict_data[key] = [log_data.average_reward_dict_per_episode.get(key, 0.0) for log_data in modified_log_datas]
            except (KeyError, AttributeError) as e:
                # print(f"Warning: Error accessing key '{key}' in reward dict: {e}. Skipping this key.")
                continue

        # print(f"{reward_dict_data=}")
        
        # If mult_weights is True, multiply each reward by its corresponding weight
        if mult_weights:
            reward_weights = {}
            for key in valid_keys:
                if key in reward_dict_data:  # Only process keys that exist in reward_dict_data
                    # Check if the reward weight is a dictionary
                    try:
                        weight_value = modified_log_datas[0].reward_weights.get(key, 1.0)
                        if isinstance(weight_value, dict):
                            # Sum all values in the dictionary for each timestep
                            reward_weights[key] = [sum(weight_dict.values()) for weight_dict in 
                                                   (log_data.reward_weights.get(key, {}).values() for log_data in modified_log_datas)]
                        else:
                            # Directly use the weight if it's not a dictionary
                            reward_weights[key] = [log_data.reward_weights.get(key, 1.0) for log_data in modified_log_datas]
                    except (KeyError, AttributeError) as e:
                        # print(f"Warning: Error accessing weight for key '{key}': {e}. Using default weight 1.0.")
                        reward_weights[key] = [1.0] * len(modified_log_datas)
            
            # Multiply each reward by its corresponding weight
            for key in list(reward_dict_data.keys()):
                if key in reward_weights:
                    reward_dict_data[key] = [value * weight for value, weight in zip(reward_dict_data[key], reward_weights[key])]
        
        # Final check: ensure we still have data to plot
        if not reward_dict_data:
            print("Warning: No reward data available for plotting. Skipping reward dict plot.")
            return

        # Separate data into positive and negative
        positive_data = {k: v for k, v in reward_dict_data.items() if all(x >= 0 for x in v)}
        negative_data = {k: v for k, v in reward_dict_data.items() if any(x < 0 for x in v)}
        
        # Calculate the total reward for each timestep to determine proportions
        total_rewards = [sum(rewards) for rewards in zip(*reward_dict_data.values())]
        # print(f"{reward_dict_data=}")
        reward_error = np.array(original_total_reward) - np.array(total_rewards)
        # Print the max, min, and mean of reward_error
        print(f"Max reward error: {np.max(reward_error)}")
        print(f"Min reward error: {np.min(reward_error)}")
        print(f"Mean reward error: {np.mean(reward_error)}")
        
        fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
        
        # Define a color map for different keys
        color_map = plt.get_cmap('tab20')  # Use a colormap with distinct colors
        color_keys = list(reward_dict_data.keys())
        
        # Plot negative values first (from the bottom)
        bottom = [0] * len(time_steps)
        for idx, (key, data) in enumerate(negative_data.items()):
            ax.fill_between(time_steps, bottom, [b + d for b, d in zip(bottom, data)], 
                           label=key, color=color_map(idx / len(color_keys)), alpha=0.5)
            bottom = [b + d for b, d in zip(bottom, data)]
        
        # Plot positive values (on top of negative values)
        bottom = [0] * len(time_steps)
        for idx, (key, data) in enumerate(positive_data.items()):
            ax.fill_between(time_steps, bottom, [b + d for b, d in zip(bottom, data)], 
                           label=key, color=color_map(idx / len(color_keys)), alpha=0.5)
            bottom = [b + d for b, d in zip(bottom, data)]
        
        ax.set_title("Proportion of Average Reward per Episode")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Proportion of Reward")
        ax.legend(loc='upper left')
        
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, f"average_reward_dict_proportion_per_episode_{'mult_weights' if mult_weights else 'no_mult_weights'}.png"))
        if show_plot:
            plt.figure(fig.number)
            plt.show()

    def plot_imitation_reward_weights(self, *, result_dir:str):
        time_steps = [log_data.num_timesteps for log_data in self._train_log_handler.log_datas]
        reward_datas = {key:[data.reward_accumulate[key] for data in self._train_log_handler.log_datas] for key in self._train_log_handler.log_datas[0].reward_accumulate.keys()}
        reward_weights_datas = {key:[data.reward_weights[key] for data in self._train_log_handler.log_datas] for key in self._train_log_handler.log_datas[0].reward_weights.keys()}
        # print(reward_datas)

        fig, axes = plt.subplots(2,1,figsize=(15,15),dpi=300)
        for key, data in reward_datas.items():
            if "joint_imitation_reward_" in key:
                if "_r" == key[-2:]:
                    line_style = "-"
                elif "_l" == key[-2:]:
                    line_style = "--"
                else:
                    line_style = "-"
                axes[0].plot(time_steps, data, label=key, linestyle=line_style)
        for key, data in reward_weights_datas.items():
            if "joint_imitation_reward_" in key:
                if "_r" == key[-2:]:
                    line_style = "-"
                elif "_l" == key[-2:]:
                    line_style = "--"
                else:
                    line_style = "-"
                axes[1].plot(time_steps, data, label=key, linestyle=line_style)
        axes[0].set_title("Reward")
        axes[0].set_xlabel("Timesteps")
        axes[0].set_ylabel("Reward")
        axes[0].legend()
        axes[1].set_title("Reward Weight")
        axes[1].set_xlabel("Timesteps")
        axes[1].set_ylabel("Reward Weight")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, "joint_reward.png"))
        plt.figure(fig.number)
        plt.show()
