python rl_train/run_train.py ^
--config_file_path rl_train/train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json ^
--config.total_timesteps 12 ^
--config.env_params.num_envs 1 --config.ppo_params.n_steps 4 ^
--config.ppo_params.batch_size 4 --config.logger_params.logging_frequency 1 ^
--config.logger_params.evaluate_frequency 1 --flag_rendering ^